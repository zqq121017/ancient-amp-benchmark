"""Strict, offline, streaming conversion of author tar.gz NPZ features to C3.

Only call ``convert_stream`` with a fresh output directory. The caller owns and
closes the input binary stream (a local file or a caller-provided URL response).
This module has no networking or extraction implementation. Failed output is
preserved; consumers must require READY.json and verify its receipt digest.

Memory use is O(N) for the frozen sequence lookup/seen flags, one bounded NPZ,
and bounded I/O buffers. The N x 40 result is a disk-backed NPY memmap.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Mapping
import zipfile

import numpy as np


SOURCE_ARCHIVE_BYTES = {
    "gut": 1_885_799_347,
    "oral": 6_407_561_863,
    "mammoth": 5_011_578_188,
}
CANONICAL = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{10,50}\Z")
BLOCK = 512
IO_CHUNK = 1 << 20
SCHEMA = "ancient-quan-c3-stream-v2"


class ConversionError(ValueError):
    """A validation failure; the run is not ready for downstream use."""


@dataclass(frozen=True)
class Limits:
    max_tar_member_bytes: int = 256 * 1024
    max_npz_uncompressed_bytes: int = 128 * 1024
    max_npy_entry_bytes: int = 64 * 1024
    max_npy_header_bytes: int = 8192
    max_metadata_bytes: int = 16 * 1024
    # Fixed directory/global-metadata allowance, plus bounded per-sequence
    # extended headers (PAX archives may have one such header for every NPZ).
    max_metadata_members: int = 10_000
    max_extended_headers_per_sequence: int = 2
    max_trailing_zero_bytes: int = 16 * 1024 * 1024
    max_path_chars: int = 1024
    validation_chunk_rows: int = 4096


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    # Only status is replaced. Other targets are inside our newly owned run dir.
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _file_evidence(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(IO_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


class _HashingReader:
    """Non-seeking bounded reads; hash every compressed byte delivered by source."""

    def __init__(self, raw: BinaryIO, expected: int, progress):
        self.raw = raw
        self.expected = expected
        self.progress = progress
        self.sha256 = hashlib.sha256()
        self.count = 0
        self.reached_eof = False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = IO_CHUNK
        if size == 0:
            return b""
        size = min(size, IO_CHUNK)
        chunks = []
        remaining = size
        while remaining:
            data = self.raw.read(remaining)
            if not isinstance(data, bytes):
                raise ConversionError("input read() must return bytes")
            if len(data) > remaining:
                raise ConversionError("input read() returned more than requested")
            if not data:
                self.reached_eof = True
                break
            self.sha256.update(data)
            self.count += len(data)
            self.progress()
            if self.count > self.expected:
                raise ConversionError(
                    f"archive size exceeds expected: read {self.count}, expected {self.expected}"
                )
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)


def _read_exact(handle, size: int, label: str) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        part = handle.read(min(remaining, IO_CHUNK))
        if not part:
            raise ConversionError(f"truncated {label}: missing {remaining} bytes")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def _safe_tar_path(name: str, limits: Limits, *, directory=False) -> str:
    if len(name) > limits.max_path_chars or not name or "\x00" in name:
        raise ConversionError(f"invalid tar path: {name!r}")
    if name.startswith("/") or "\\" in name or ":" in name:
        raise ConversionError(f"unsafe tar path: {name!r}")
    parts = name.split("/")
    if ".." in parts:
        raise ConversionError(f"unsafe tar parent traversal: {name!r}")
    parts = [part for part in parts if part not in ("", ".")]
    if not parts and not directory:
        raise ConversionError(f"empty tar file path: {name!r}")
    return "/".join(parts)


def _pax_fields(data: bytes) -> dict[str, str]:
    """Parse bounded POSIX PAX records; no extraction or filesystem operations."""
    offset = 0
    result = {}
    while offset < len(data):
        space = data.find(b" ", offset)
        if space < 0:
            raise ConversionError("invalid PAX record length")
        try:
            length = int(data[offset:space])
        except ValueError as exc:
            raise ConversionError("invalid PAX record length") from exc
        if length <= space - offset + 2 or offset + length > len(data):
            raise ConversionError("invalid PAX record extent")
        record = data[space + 1:offset + length]
        if not record.endswith(b"\n") or b"=" not in record:
            raise ConversionError("invalid PAX record")
        key, value = record[:-1].split(b"=", 1)
        try:
            key, value = key.decode("utf-8"), value.decode("utf-8")
        except UnicodeError as exc:
            raise ConversionError("non-UTF8 PAX record") from exc
        if key in result:
            raise ConversionError(f"duplicate PAX key: {key}")
        if key.startswith("GNU.sparse") or key in ("linkpath", "GNU.dumpdir"):
            raise ConversionError(f"unsupported PAX field: {key}")
        result[key] = value
        offset += length
    return result


def _tar_records(handle, limits: Limits, expected_rows: int, stats: dict):
    """Strict sequential tar reader including mandatory two-block EOF marker.

    Allows ordinary files, directories, bounded PAX metadata, and GNU longnames.
    Every member is read in memory only after its individual size is bounded.
    Symlinks, hardlinks, devices, sparse files, appended archives, and nonzero
    trailing data are rejected. Header checksums are validated by stdlib tarfile.
    """
    global_pax = {}
    next_pax = {}
    longname = None
    meta_count = 0
    record_count = 0
    metadata_limit = limits.max_metadata_members + expected_rows * limits.max_extended_headers_per_sequence
    while True:
        header = _read_exact(handle, BLOCK, "tar header or EOF marker")
        stats["decompressed_tar_bytes"] += BLOCK
        if header == b"\0" * BLOCK:
            second = _read_exact(handle, BLOCK, "second tar EOF block")
            stats["decompressed_tar_bytes"] += BLOCK
            if second != b"\0" * BLOCK:
                raise ConversionError("tar EOF must contain two zero blocks")
            if next_pax or longname is not None:
                raise ConversionError("orphan tar extended header at EOF")
            trailing = 0
            while chunk := handle.read(IO_CHUNK):
                stats["decompressed_tar_bytes"] += len(chunk)
                trailing += len(chunk)
                if trailing > limits.max_trailing_zero_bytes:
                    raise ConversionError("tar trailing padding exceeds limit")
                if chunk.strip(b"\0"):
                    raise ConversionError("nonzero data after tar EOF marker")
            stats["tar_eof_valid"] = True
            return
        try:
            info = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        except (tarfile.TarError, UnicodeError, ValueError) as exc:
            raise ConversionError(f"invalid tar header: {exc}") from exc
        record_count += 1
        if record_count > expected_rows + metadata_limit:
            raise ConversionError("tar member count exceeds bounded record limit")
        kind = info.type
        is_metadata = kind in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME)
        if is_metadata:
            meta_count += 1
            if meta_count > metadata_limit:
                raise ConversionError("tar metadata count exceeds limit")
            size = info.size
            if size < 0 or size > limits.max_metadata_bytes:
                raise ConversionError("tar metadata size exceeds limit")
        else:
            pax = dict(global_pax)
            pax.update(next_pax)
            name = pax.get("path", longname if longname is not None else info.name)
            try:
                size = int(pax.get("size", info.size))
            except ValueError as exc:
                raise ConversionError("invalid PAX size") from exc
            next_pax = {}
            longname = None
            if kind not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
                raise ConversionError(f"unsupported tar member type {kind!r}: {name!r}")
            if size < 0 or size > limits.max_tar_member_bytes:
                raise ConversionError(f"tar member size exceeds limit: {name!r}: {size}")
            name = _safe_tar_path(name, limits, directory=kind == tarfile.DIRTYPE)
            if kind == tarfile.DIRTYPE and size != 0:
                raise ConversionError(f"nonempty directory member: {name!r}")
        payload = _read_exact(handle, size, "tar member payload")
        padding_size = (-size) % BLOCK
        padding = _read_exact(handle, padding_size, "tar member padding")
        stats["decompressed_tar_bytes"] += size + padding_size
        if padding.strip(b"\0"):
            raise ConversionError("nonzero tar member padding")
        if is_metadata:
            if kind == tarfile.GNUTYPE_LONGNAME:
                if longname is not None:
                    raise ConversionError("duplicate GNU longname header")
                try:
                    longname = payload.rstrip(b"\0").decode("utf-8")
                except UnicodeError as exc:
                    raise ConversionError("invalid GNU longname encoding") from exc
            elif kind == tarfile.XGLTYPE:
                global_pax.update(_pax_fields(payload))
            else:
                if next_pax:
                    raise ConversionError("duplicate pending PAX header")
                next_pax = _pax_fields(payload)
            continue
        if kind == tarfile.DIRTYPE:
            meta_count += 1
            stats["directory_members"] += 1
            if meta_count > metadata_limit or stats["directory_members"] > limits.max_metadata_members:
                raise ConversionError("tar directory/metadata count exceeds limit")
            continue
        stats["npz_members"] += 1
        yield name, payload


def _safe_npy(data: bytes, name: str, limits: Limits) -> np.ndarray:
    """Validate shape/data length before NumPy is permitted to allocate."""
    handle = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(
                handle, max_header_size=limits.max_npy_header_bytes
            )
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(
                handle, max_header_size=limits.max_npy_header_bytes
            )
        else:
            raise ConversionError(f"unsupported NPY format {version}: {name}")
    except (ValueError, EOFError) as exc:
        raise ConversionError(f"invalid NPY header for {name}: {exc}") from exc
    if dtype.hasobject:
        raise ConversionError(f"unsupported object dtype in {name}; pickle is prohibited")
    if name == "features.npy":
        if len(shape) != 2 or not 10 <= shape[0] <= 50 or shape[1] != 130:
            raise ConversionError(f"features must have shape (L,130), L in 10..50; got {shape}")
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ConversionError(f"features dtype mismatch: expected native float32/float64, got {dtype}")
    else:
        if shape != () or dtype.kind not in "USiu":
            raise ConversionError(f"seq_id must be a safe scalar string/integer; got {shape}, {dtype}")
    count = 1
    for dimension in shape:
        count *= dimension
    expected = count * dtype.itemsize
    if expected > limits.max_npy_entry_bytes or len(data) - handle.tell() != expected:
        raise ConversionError(f"NPY payload size mismatch or limit exceeded: {name}")
    return np.load(io.BytesIO(data), allow_pickle=False, max_header_size=limits.max_npy_header_bytes)


def _npz_c3(payload: bytes, sequence: str, limits: Limits):
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != 2 or set(names) != {"features.npy", "seq_id.npy"}:
                raise ConversionError(f"NPZ requires exactly features and seq_id entries; got {names}")
            if sum(entry.file_size for entry in entries) > limits.max_npz_uncompressed_bytes:
                raise ConversionError("NPZ uncompressed size exceeds limit")
            arrays = {}
            for entry in entries:
                if entry.flag_bits & 1 or entry.is_dir():
                    raise ConversionError("encrypted/directory NPZ entry is unsupported")
                if entry.file_size > limits.max_npy_entry_bytes:
                    raise ConversionError(f"NPZ entry exceeds size limit: {entry.filename}")
                with archive.open(entry) as handle:
                    data = handle.read(limits.max_npy_entry_bytes + 1)
                    if len(data) != entry.file_size or len(data) > limits.max_npy_entry_bytes:
                        raise ConversionError(f"NPZ entry length mismatch: {entry.filename}")
                    if handle.read(1):
                        raise ConversionError(f"NPZ entry overflow: {entry.filename}")
                arrays[entry.filename] = _safe_npy(data, entry.filename, limits)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
        raise ConversionError(f"invalid NPZ for {sequence}: {exc}") from exc
    features = arrays["features.npy"]
    if features.shape[0] != len(sequence):
        raise ConversionError(f"feature length {features.shape[0]} != sequence length {len(sequence)}")
    if not np.isfinite(features).all():
        raise ConversionError(f"nonfinite features for {sequence}")
    with np.errstate(over="raise", invalid="raise"):
        try:
            c3 = features[:, 40:80].mean(axis=0)
        except FloatingPointError as exc:
            raise ConversionError(f"nonfinite C3 mean for {sequence}") from exc
    if c3.dtype != features.dtype or not np.isfinite(c3).all():
        raise ConversionError(f"invalid C3 result for {sequence}")
    id_array = arrays["seq_id.npy"]
    seq_id = id_array.item()
    if isinstance(seq_id, bytes):
        try:
            seq_id = seq_id.decode("utf-8")
        except UnicodeError as exc:
            raise ConversionError("seq_id byte string is not UTF-8") from exc
    id_type = "string" if isinstance(seq_id, str) else "integer"
    id_meaning = "uninterpreted_scalar_identifier"
    if isinstance(seq_id, str) and CANONICAL.fullmatch(seq_id):
        if seq_id != sequence:
            raise ConversionError(f"canonical seq_id does not equal filename sequence: {sequence}")
        id_meaning = "canonical_sequence_equal_to_filename"
    # Preserve typed-value evidence for every ID without treating integer IDs
    # or noncanonical strings as sequences or inferred row indices.
    id_evidence = {"numpy_dtype": id_array.dtype.str, "scalar_type": id_type,
                   "meaning": id_meaning}
    encoded = json.dumps({**id_evidence, "value": seq_id}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    id_evidence["typed_value_sha256"] = hashlib.sha256(encoded).hexdigest()
    return c3, id_evidence


def _verify_final_arrays(out: Path, count: int, samples: dict, limits: Limits, progress) -> dict:
    """Reopen committed filenames, check every value/cast, and original samples."""
    author = np.load(out / "c3_author_mean.npy", mmap_mode="r", allow_pickle=False)
    model = None
    try:
        model = np.load(out / "c3.npy", mmap_mode="r", allow_pickle=False)
        if author.shape != (count, 40) or author.dtype != np.dtype("float64"):
            raise ConversionError("reopened author mean shape/dtype mismatch")
        if model.shape != (count, 40) or model.dtype != np.dtype("float32"):
            raise ConversionError("reopened model C3 shape/dtype mismatch")
        max_abs = 0.0
        max_relative = 0.0
        max_bound_ratio = 0.0
        half_subnormal = float(np.finfo(np.float32).smallest_subnormal) / 2.0
        for start in range(0, count, limits.validation_chunk_rows):
            stop = min(count, start + limits.validation_chunk_rows)
            before = np.asarray(author[start:stop])
            after = np.asarray(model[start:stop])
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                raise ConversionError("nonfinite value in reopened final arrays")
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                expected = before.astype(np.float32)
            if not np.array_equal(after, expected):
                raise ConversionError("reopened C3 does not equal exact float32 cast of author means")
            absolute = np.abs(after.astype(np.float64) - before)
            magnitude = np.abs(before)
            relative = np.divide(absolute, magnitude, out=np.zeros_like(absolute), where=magnitude != 0)
            # Conservative round-to-nearest error bound, including subnormals.
            bound = np.finfo(np.float32).eps * magnitude + half_subnormal
            if not np.all(absolute <= bound):
                raise ConversionError("float32 cast error exceeds documented rounding bound")
            max_abs = max(max_abs, float(absolute.max()))
            max_relative = max(max_relative, float(relative.max()))
            max_bound_ratio = max(max_bound_ratio, float((absolute / bound).max()))
            progress()
        for row, original_mean in samples.items():
            if not np.array_equal(author[row], original_mean):
                raise ConversionError(f"reopened author mean differs from original formula at row {row}")
        return {
            "reopened_final_files": True,
            "all_rows_finite": True,
            "all_rows_equal_exact_float32_cast": True,
            "original_formula_sample_rows": sorted(samples),
            "original_formula_samples_array_equal": True,
            "max_absolute_cast_error": max_abs,
            "max_relative_cast_error": max_relative,
            "relative_error_definition": "abs(float64(c3)-author_mean)/abs(author_mean); zero when author_mean is zero",
            "rounding_bound": "abs_error <= float32_eps * abs(author_mean) + float32_smallest_subnormal/2",
            "rounding_bound_verified_all_values": True,
            "max_fraction_of_rounding_bound": max_bound_ratio,
        }
    finally:
        author._mmap.close()
        if model is not None:
            model._mmap.close()


def convert_stream(
    stream: BinaryIO,
    *,
    sequences_path: str | Path,
    output_dir: str | Path,
    expected_archive_bytes: int,
    source_evidence: Mapping,
    expected_archive_sha256: str | None = None,
    limits: Limits = Limits(),
    progress_interval_seconds: float = 5.0,
) -> dict:
    """Convert one archive in frozen S50 sequence order; return ready receipt.

    A successfully created READY.json is the sole commit point. The caller must
    supply the author's expected archive byte count and source provenance. No
    input stream seek or close is performed. Existing output directories fail
    before any input is read or any output is changed.
    """
    if type(expected_archive_bytes) is not int or expected_archive_bytes <= 0:
        raise ValueError("expected_archive_bytes must be a positive integer")
    if progress_interval_seconds < 0:
        raise ValueError("progress interval cannot be negative")
    if expected_archive_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", expected_archive_sha256):
        raise ValueError("expected_archive_sha256 must be 64 hexadecimal characters")
    if any(type(value) is not int or value <= 0 for value in asdict(limits).values()):
        raise ValueError("all resource limits must be positive integers")
    provenance = json.loads(json.dumps(dict(source_evidence), allow_nan=False))
    if not provenance:
        raise ValueError("source_evidence must not be empty")
    sequences_path = Path(sequences_path).resolve()
    out = Path(output_dir).absolute()
    out.mkdir(parents=False, exist_ok=False)
    status = {
        "schema": SCHEMA,
        "state": "initializing",
        "started_utc": _now(),
        "processed_rows": 0,
        "expected_rows": None,
        "compressed_bytes_read": 0,
        "expected_archive_bytes": expected_archive_bytes,
        "output_dir": str(out),
    }
    reader = None
    matrix = None
    author_matrix = None
    id_handle = None
    frozen = None
    last_status = 0.0

    def progress(force=False):
        nonlocal last_status
        moment = time.monotonic()
        if force or moment - last_status >= progress_interval_seconds:
            if reader is not None:
                status["compressed_bytes_read"] = reader.count
            status["updated_utc"] = _now()
            _atomic_json(out / "status.json", status)
            last_status = moment

    try:
        progress(True)
        frozen_evidence = _file_evidence(sequences_path)
        frozen = np.load(sequences_path, mmap_mode="r", allow_pickle=False)
        if frozen.ndim != 1 or frozen.dtype != np.dtype("S50") or len(frozen) == 0:
            raise ConversionError(f"frozen sequences must be a nonempty 1D S50 NPY; got {frozen.shape}, {frozen.dtype}")
        lookup = {}
        for row, raw_sequence in enumerate(frozen):
            try:
                sequence = bytes(raw_sequence).decode("ascii")
            except UnicodeError as exc:
                raise ConversionError(f"non-ASCII frozen sequence at row {row}") from exc
            if not CANONICAL.fullmatch(sequence):
                raise ConversionError(f"invalid frozen canonical 10..50 sequence at row {row}")
            if sequence in lookup:
                raise ConversionError(f"duplicate frozen sequence at rows {lookup[sequence]} and {row}")
            lookup[sequence] = row
            progress()
        count = len(frozen)
        status["expected_rows"] = count
        status["state"] = "converting"
        seen = np.zeros(count, dtype=np.bool_)
        matrix_path = out / "c3.npy.partial"
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype="float32", shape=(count, 40))
        author_path = out / "c3_author_mean.npy.partial"
        author_matrix = np.lib.format.open_memmap(author_path, mode="w+", dtype="float64", shape=(count, 40))
        id_path = out / "seq_id_evidence.csv.partial"
        id_handle = id_path.open("x", encoding="utf-8", newline="")
        id_writer = csv.writer(id_handle, lineterminator="\n")
        id_columns = ["row_index", "sequence", "numpy_dtype", "scalar_type", "meaning", "typed_value_sha256"]
        id_writer.writerow(id_columns)
        stats = {"decompressed_tar_bytes": 0, "directory_members": 0, "npz_members": 0, "tar_eof_valid": False}
        seq_id_types = {"string": 0, "integer": 0}
        seq_id_meanings = {"canonical_sequence_equal_to_filename": 0, "uninterpreted_scalar_identifier": 0}
        source_dtypes = {"float32": 0, "float64": 0}
        sample_rows = {0, count // 2, count - 1}
        original_mean_samples = {}
        reader = _HashingReader(stream, expected_archive_bytes, progress)
        with gzip.GzipFile(fileobj=reader, mode="rb") as decompressed:
            for name, payload in _tar_records(decompressed, limits, count, stats):
                filename = name.rsplit("/", 1)[-1]
                if not filename.endswith(".npz"):
                    raise ConversionError(f"unexpected non-NPZ member: {name!r}")
                sequence = filename[:-4]
                if sequence not in lookup:
                    raise ConversionError(f"unknown sequence/member: {name!r}")
                row = lookup[sequence]
                if seen[row]:
                    raise ConversionError(f"duplicate sequence member: {sequence}")
                original_c3, id_evidence = _npz_c3(payload, sequence, limits)
                author_mean = original_c3.astype(np.float64, copy=False)
                try:
                    with np.errstate(over="raise", invalid="raise", under="ignore"):
                        c3 = author_mean.astype(np.float32)
                except FloatingPointError as exc:
                    raise ConversionError(f"float32 dtype bridge produced nonfinite output for {sequence}") from exc
                if not np.isfinite(c3).all():
                    raise ConversionError(f"float32 dtype bridge produced nonfinite output for {sequence}")
                author_matrix[row] = author_mean
                matrix[row] = c3
                if row in sample_rows:
                    original_mean_samples[row] = author_mean.copy()
                id_writer.writerow([row, sequence] + [id_evidence[key] for key in id_columns[2:]])
                seen[row] = True
                seq_id_types[id_evidence["scalar_type"]] += 1
                seq_id_meanings[id_evidence["meaning"]] += 1
                source_dtypes[str(original_c3.dtype)] += 1
                status["processed_rows"] += 1
                progress()
        # Gzip EOF has now validated all member CRC/ISIZE trailers. Require the
        # underlying stream's EOF too so no compressed bytes evade hash/count.
        while reader.read(IO_CHUNK):
            pass
        if reader.count != expected_archive_bytes or not reader.reached_eof:
            raise ConversionError(f"archive size mismatch: got {reader.count}, expected {expected_archive_bytes}")
        archive_sha256 = reader.sha256.hexdigest()
        if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256.lower():
            raise ConversionError("archive SHA-256 does not match expected digest")
        if not seen.all():
            missing = np.flatnonzero(~seen)
            examples = [bytes(frozen[row]).decode("ascii") for row in missing[:5]]
            raise ConversionError(f"missing {len(missing)} frozen sequences; examples: {examples}")
        status["state"] = "validating_outputs"
        progress(True)
        matrix.flush()
        matrix._mmap.close()
        matrix = None
        author_matrix.flush()
        author_matrix._mmap.close()
        author_matrix = None
        id_handle.flush()
        os.fsync(id_handle.fileno())
        id_handle.close()
        id_handle = None
        index_path = out / "sequence_index.csv.partial"
        with index_path.open("x", encoding="ascii", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["row_index", "sequence"])
            for row, raw_sequence in enumerate(frozen):
                writer.writerow([row, bytes(raw_sequence).decode("ascii")])
            handle.flush()
            os.fsync(handle.fileno())
        # The frozen input must still be identical at final commit.
        if _file_evidence(sequences_path) != frozen_evidence:
            raise ConversionError("frozen sequence input changed during conversion")
        os.rename(matrix_path, out / "c3.npy")
        os.rename(author_path, out / "c3_author_mean.npy")
        os.rename(index_path, out / "sequence_index.csv")
        os.rename(id_path, out / "seq_id_evidence.csv")
        dtype_bridge = _verify_final_arrays(out, count, original_mean_samples, limits, progress)
        outputs = {
            "c3.npy": {**_file_evidence(out / "c3.npy"), "dtype": "float32", "shape": [count, 40]},
            "c3_author_mean.npy": {**_file_evidence(out / "c3_author_mean.npy"), "dtype": "float64", "shape": [count, 40]},
            "sequence_index.csv": {**_file_evidence(out / "sequence_index.csv"), "columns": ["row_index", "sequence"], "rows": count},
            "seq_id_evidence.csv": {**_file_evidence(out / "seq_id_evidence.csv"), "columns": id_columns, "rows": count, "row_order": "archive arrival; row_index maps to frozen input"},
        }
        receipt = {
            "schema": SCHEMA,
            "state": "ready",
            "started_utc": status["started_utc"],
            "completed_utc": _now(),
            "source_evidence": provenance,
            "archive": {"bytes": reader.count, "expected_bytes": expected_archive_bytes, "sha256": archive_sha256, "expected_sha256": expected_archive_sha256, "gzip_crc_and_eof_valid": True, "raw_eof_observed": reader.reached_eof},
            "frozen_sequences": {"path": str(sequences_path), **frozen_evidence, "dtype": "S50", "shape": [count]},
            "conversion": {
                "formula": "features[:, 40:80].mean(axis=0)",
                "mean_computed_in_original_dtype": True,
                "source_feature_dtype_counts": source_dtypes,
                "author_mean_storage": "float64; original float32 means promoted exactly, original float64 means preserved",
                "model_dtype_bridge": "one float32 cast after original-dtype mean; numerical bridge, not a bitwise author-vector identity claim",
                "dtype_bridge_verification": dtype_bridge,
                "input_feature_width": 130,
                "feature_rows_equal_sequence_length": True,
                "all_input_features_finite": True,
                "pca_applied": False,
                "l2_normalization_applied": False,
                "minmax_scaling_applied": False,
                "row_order": "exact frozen sequences.npy order",
                "bijection_verified": True,
                "seq_id_scalar_types": seq_id_types,
                "seq_id_meanings": seq_id_meanings,
                "seq_id_typed_value_hash_encoding": "SHA256 of UTF-8 JSON with keys numpy_dtype,scalar_type,meaning,value; sort_keys=True, ensure_ascii=False, separators=(',',':')",
                "allow_pickle": False,
            },
            "tar": stats,
            "limits": asdict(limits),
            "outputs": outputs,
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        }
        _atomic_json(out / "receipt.json", receipt)
        ready = {"schema": SCHEMA, "state": "ready", "receipt": "receipt.json", "receipt_sha256": _file_evidence(out / "receipt.json")["sha256"]}
        # READY is written last, after all archive checks and complete outputs.
        status["state"] = "ready"
        progress(True)
        _atomic_json(out / "READY.json", ready)
        return receipt
    except BaseException as exc:
        status["state"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        try:
            progress(True)
        except Exception:
            pass
        raise
    finally:
        if matrix is not None:
            matrix.flush()
            matrix._mmap.close()
        if author_matrix is not None:
            author_matrix.flush()
            author_matrix._mmap.close()
        if id_handle is not None:
            id_handle.close()
        if isinstance(frozen, np.memmap):
            frozen._mmap.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="local tar.gz only")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="must not exist")
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--source-evidence-json", type=Path, default=Path(__file__).resolve().parents[1] / "configs/experiment.json")
    args = parser.parse_args(argv)
    with args.source_evidence_json.open(encoding="utf-8") as handle:
        evidence = json.load(handle)
        evidence = evidence.get("feature_source", evidence)
    with args.archive.open("rb") as stream:
        receipt = convert_stream(stream, sequences_path=args.sequences, output_dir=args.output_dir,
            expected_archive_bytes=args.expected_bytes, source_evidence=evidence,
            expected_archive_sha256=args.expected_sha256)
    print(json.dumps({"state": "ready", "rows": receipt["frozen_sequences"]["shape"][0], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
