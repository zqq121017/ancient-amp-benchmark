"""C3 input preparation and sequence descriptors."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]

CHUNK_ROWS = 4096

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()

def require_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Expected a lowercase SHA256 digest")
    return value

def checked_file(path, expected):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or sha(path) != require_digest(expected):
        raise ValueError("Input hash differs: " + str(path))
    return path

def validate_mapping(mapping_path, sequences):
    """Return target-row -> C3-row permutation; require a complete bijection."""
    if sequences.dtype != np.dtype("S50") or sequences.ndim != 1:
        raise ValueError("Frozen sequences must be a one-dimensional S50 array")
    lookup = {}
    for index, raw in enumerate(sequences):
        sequence = bytes(raw).decode("ascii")
        if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{10,50}", sequence):
            raise ValueError("Invalid frozen sequence")
        if sequence in lookup:
            raise ValueError("Duplicate frozen sequence")
        lookup[sequence] = index
    n = len(sequences)
    rowmap = np.full(n, -1, dtype=np.int64)
    seen_source = np.zeros(n, dtype=bool)
    count = 0
    with Path(mapping_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["row_index", "sequence"]:
            raise ValueError("Mapping requires exactly row_index,sequence columns")
        for row in rows:
            if set(row) != {"row_index", "sequence"}:
                raise ValueError("Malformed mapping row")
            raw_index = row["row_index"]
            if not isinstance(raw_index, str) or not re.fullmatch(r"0|[1-9][0-9]*", raw_index):
                raise ValueError("Mapping index must be an exact nonnegative integer")
            source_index = int(raw_index)
            target_index = lookup.get(row["sequence"])
            if (not 0 <= source_index < n or seen_source[source_index]
                    or target_index is None or rowmap[target_index] != -1):
                raise ValueError("Mapping has a duplicate, extra, or invalid row")
            rowmap[target_index] = source_index
            seen_source[source_index] = True
            count += 1
    if count != n or not seen_source.all() or np.any(rowmap < 0):
        raise ValueError("Mapping does not cover the complete frozen pool")
    return rowmap

def verify_conversion(conversion_dir, sequences_path, pin):
    """Read-only converter receipt/payload validation; also used with tiny tests."""
    conversion_dir = Path(conversion_dir).resolve(strict=True)
    ready_path = conversion_dir / "READY.json"
    ready_hash = sha(ready_path)
    ready = read(ready_path)
    if (ready.get("schema") != "ancient-quan-c3-stream-v2"
            or ready.get("state") != "ready" or ready.get("receipt") != "receipt.json"):
        raise ValueError("A successful pinned converter READY receipt is required")
    receipt_path = checked_file(conversion_dir / "receipt.json", ready.get("receipt_sha256"))
    receipt = read(receipt_path)
    if receipt.get("schema") != ready["schema"] or receipt.get("state") != "ready":
        raise ValueError("Conversion did not complete")
    archive = receipt.get("archive", {})
    if (archive.get("bytes") != pin["archive_bytes"]
            or archive.get("sha256") != pin["archive_sha256"]
            or archive.get("gzip_crc_and_eof_valid") is not True
            or archive.get("raw_eof_observed") is not True):
        raise ValueError("Conversion archive identity/completeness differs")
    frozen = receipt.get("frozen_sequences", {})
    if (frozen.get("sha256") != pin["sequences_sha256"]
            or frozen.get("dtype") != "S50" or frozen.get("shape") != [pin["n"]]):
        raise ValueError("Conversion used another sequence pool")
    conversion = receipt.get("conversion", {})
    if (conversion.get("formula") != "features[:, 40:80].mean(axis=0)"
            or conversion.get("mean_computed_in_original_dtype") is not True
            or conversion.get("bijection_verified") is not True
            or conversion.get("pca_applied") is not False
            or conversion.get("l2_normalization_applied") is not False
            or conversion.get("minmax_scaling_applied") is not False):
        raise ValueError("Conversion preprocessing differs")
    outputs = receipt.get("outputs", {})
    feature_info = outputs.get("c3.npy", {})
    mapping_info = outputs.get("sequence_index.csv", {})
    if (feature_info.get("sha256") != pin["c3_sha256"]
            or feature_info.get("dtype") != "float32" or feature_info.get("shape") != [pin["n"], 40]
            or mapping_info.get("sha256") != pin["mapping_sha256"]
            or mapping_info.get("rows") != pin["n"]):
        raise ValueError("C3 payload does not match the released experiment")
    feature_path = checked_file(conversion_dir / "c3.npy", pin["c3_sha256"])
    mapping_path = checked_file(conversion_dir / "sequence_index.csv", pin["mapping_sha256"])
    sequences_path = checked_file(sequences_path, pin["sequences_sha256"])
    sequences = np.load(sequences_path, allow_pickle=False, mmap_mode="r")
    if sequences.shape != (pin["n"],) or sequences.dtype != np.dtype("S50"):
        raise ValueError("Frozen sequence shape/dtype differs")
    features = np.load(feature_path, allow_pickle=False, mmap_mode="r")
    if features.shape != (pin["n"], 40) or features.dtype != np.dtype("float32"):
        raise ValueError("Converted C3 shape/dtype differs")
    rowmap = validate_mapping(mapping_path, sequences)
    for start in range(0, pin["n"], CHUNK_ROWS):
        if not np.isfinite(features[start:start + CHUNK_ROWS]).all():
            raise ValueError("Nonfinite C3 features")
    immutable = {
        ready_path: ready_hash, receipt_path: ready["receipt_sha256"],
        feature_path: pin["c3_sha256"], mapping_path: pin["mapping_sha256"],
        sequences_path: pin["sequences_sha256"],
    }
    return features, rowmap, immutable

def write_padded_actor(features, rowmap, path):
    """The original c3_inputs.py assignment order, with positive-zero padding."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Actor exists; do not overwrite")
    n = len(rowmap)
    if (features.shape != (n, 40) or features.dtype != np.dtype("float32")
            or rowmap.dtype != np.dtype("int64")
            or not np.array_equal(np.sort(rowmap), np.arange(n))):
        raise ValueError("Expected float32 C3 and a complete int64 permutation")
    actor = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, 64))
    try:
        for start in range(0, n, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n)
            source = features[rowmap[start:stop]]
            if not np.isfinite(source).all():
                raise ValueError("Nonfinite C3 features")
            actor[start:stop] = np.float32(0)
            actor[start:stop, :40] = source
        actor.flush()
    finally:
        del actor
    result = np.load(path, allow_pickle=False, mmap_mode="r")
    try:
        for start in range(0, n, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n)
            if (not np.array_equal(result[start:stop, :40].view(np.uint32),
                                   features[rowmap[start:stop]].view(np.uint32))
                    or np.any(result[start:stop, 40:].view(np.uint32) != 0)):
                raise ValueError("Feature bits or padding changed")
    finally:
        del result
    return sha(path)

def relative_path(path, base):
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # Windows paths on separate drives cannot be relative.
        return str(path)

def prepare(dataset, conversion_dir, cache_root, output_dir):
    pins = read(PACKAGE / "configs/experiment.json")["inputs"]
    if dataset not in pins:
        raise ValueError("Unknown dataset")
    pin = pins[dataset]
    conversion_dir, cache_root = Path(conversion_dir).resolve(), Path(cache_root).resolve()
    output = Path(output_dir).resolve()
    for protected in (PACKAGE, conversion_dir, cache_root):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Output must be a new separate directory from code and source inputs")
    if output.exists():
        raise FileExistsError("Output exists; preserve it and inspect")
    cache_manifest_path = checked_file(cache_root / "manifest.json", pin["cache_manifest_sha256"])
    cache_manifest = read(cache_manifest_path)
    if (cache_manifest.get("n") != pin["n"]
            or cache_manifest.get("files", {}).get("sequences.npy") != pin["sequences_sha256"]):
        raise ValueError("Wrong frozen final cache")
    sequences_path = cache_root / "sequences.npy"
    features, rowmap, immutable = verify_conversion(conversion_dir, sequences_path, pin)
    immutable[cache_manifest_path] = pin["cache_manifest_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    try:
        actor_path = output / "actor64.npy"
        actor_hash = write_padded_actor(features, rowmap, actor_path)
        if actor_hash != pin["actor_sha256"]:
            raise ValueError("Actor serialization/hash differs from the frozen experiment")
        for path, digest in immutable.items():
            checked_file(path, digest)
        inputs = dict(dataset=dataset,
            cache_root=relative_path(cache_root, output),
            cache_manifest_sha256=pin["cache_manifest_sha256"],
            actor_path="actor64.npy", actor_sha256=actor_hash,
            row_sequences_path=relative_path(sequences_path, output),
            row_sequences_sha256=pin["sequences_sha256"],
            c3_provenance={"source": "Quan C3",
                "preprocessing": "features[:,40:80].mean(axis=0); float32; zero-pad to64",
                "conversion_receipt_sha256": sha(conversion_dir / "receipt.json"),
                "pca_refitted": False, "normalization_applied": False})
        with (output / "inputs.json").open("x", encoding="utf-8") as stream:
            json.dump(inputs, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return inputs
    except BaseException as exc:
        with (output / "PREPARATION_FAILED.json").open("x", encoding="utf-8") as stream:
            json.dump({"complete": False, "error": repr(exc), "outputs_preserved": True}, stream, indent=2)
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("gut", "oral", "mammoth"))
    parser.add_argument("--conversion-dir", required=True, help="Complete output of convert_c3.py")
    parser.add_argument("--cache-root", required=True, help="Original final Porto cache directory")
    parser.add_argument("--output", required=True, help="New separate directory for actor64.npy and inputs.json")
    args = parser.parse_args()
    result = prepare(args.dataset, args.conversion_dir, args.cache_root, args.output)
    print(json.dumps({"dataset": args.dataset, "actor_sha256": result["actor_sha256"],
        "inputs": str(Path(args.output).resolve() / "inputs.json"), "training_started": False}, indent=2))

AA = "ACDEFGHIKLMNPQRSTVWY"

LOOKUP = {residue: index for index, residue in enumerate(AA)}

RAW_DIM = 462

DIV_DIM = 420

def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), np.float32(1e-8))

def extract(sequence):
    """Return 462 raw features, a 420-D unit diversity vector, and entropy.

    Raw columns: AAC[0:20], length/50[20], H/log2(20)[21], full directed
    DPC[22:422], first-four AAC[422:442], last-four AAC[442:462]. Diversity
    concatenates separately L2-normalized AAC and DPC, each weighted 1/sqrt(2).
    """
    if isinstance(sequence, (bytes, np.bytes_)):
        sequence = sequence.decode("ascii")
    if not 10 <= len(sequence) <= 50:
        raise ValueError("Expected a frozen original-pool sequence of length 10--50")
    try:
        ids = np.fromiter((LOOKUP[residue] for residue in sequence), dtype=np.int64, count=len(sequence))
    except KeyError as exc:
        raise ValueError("Nonstandard residue in original-pool sequence") from exc
    n = len(ids)
    aac = (np.bincount(ids, minlength=20) / n).astype(np.float32)
    nonzero = aac[aac > 0].astype(np.float64)
    entropy = float(-(nonzero * np.log2(nonzero)).sum() / math.log2(20))
    entropy = float(np.clip(entropy, 0.0, 1.0))
    dpc = (np.bincount(ids[:-1] * 20 + ids[1:], minlength=400) / (n - 1)).astype(np.float32)
    nterm = (np.bincount(ids[:4], minlength=20) / 4.0).astype(np.float32)
    cterm = (np.bincount(ids[-4:], minlength=20) / 4.0).astype(np.float32)
    raw = np.concatenate((aac, np.array([n / 50.0, entropy], dtype=np.float32), dpc, nterm, cterm))
    diversity = np.concatenate((normalize(aac), normalize(dpc))) / np.float32(math.sqrt(2))
    assert raw.shape == (RAW_DIM,) and diversity.shape == (DIV_DIM,)
    return raw, diversity, entropy

def extract_many(sequences):
    raw = np.empty((len(sequences), RAW_DIM), dtype=np.float32)
    diversity = np.empty((len(sequences), DIV_DIM), dtype=np.float32)
    entropy = np.empty(len(sequences), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        raw[index], diversity[index], entropy[index] = extract(sequence)
    return raw, diversity, entropy


if __name__ == "__main__":
    main()
