"""Sequence scores and equal-weight subset reward."""
import copy
import hashlib
import itertools
import json
import math
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
import numpy as np

CACHE_FILES = ("sequences.npy", "order64.npy", "div420.npy", "cluster.npy",
               "quality_S1.npy", "quality_S2.npy", "transform.npz")

COMPONENT_NAMES = ["Q", "D", "C", "U"]

REWARD_FORMULA = "(Q + D + C + 1 - U) / 4"

Q_SOURCES = {"S1": "preserved_sequence_entropy", "S2": "Porto2018_raw_mu_over_sumexp"}

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

def verify_cache_structure(cache, n):
    expected = {"sequences.npy": ((n,), np.dtype("S50")),
        "order64.npy": ((n, 64), np.dtype("float32")),
        "div420.npy": ((n, 420), np.dtype("float16")),
        "cluster.npy": ((n,), np.dtype("int16")),
        "quality_S1.npy": ((n,), np.dtype("float32")),
        "quality_S2.npy": ((n,), np.dtype("float32"))}
    for name, (shape, dtype) in expected.items():
        array = np.load(cache / name, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError("Unexpected cache shape/dtype: " + name)
        for start in range(0, n, 4096):
            piece = array[start:start+4096]
            if name == "sequences.npy":
                for raw in piece:
                    seq = raw.decode("ascii")
                    if not 10 <= len(seq) <= 50 or not set(seq) <= AA:
                        raise ValueError("Invalid ancient sequence")
            elif not np.isfinite(piece).all():
                raise ValueError("Non-finite prepared cache: " + name)
            if name.startswith("quality_") and (np.any(piece < 0) or np.any(piece > 1)):
                raise ValueError("Raw q is outside [0,1]; do not clip or normalize: " + name)
            if name == "cluster.npy" and (np.any(piece < 0) or np.any(piece >= 256)):
                raise ValueError("Invalid fixed coverage cluster")
    sequences = np.load(cache / "sequences.npy", mmap_mode="r", allow_pickle=False)
    if n > 1 and not np.all(sequences[1:] > sequences[:-1]):
        raise ValueError("Sequence cache must be unique and lexicographically ordered")

class PortoEqualPool:
    """Same index/group/representation environment, explicit new q/reward contract."""
    def __init__(self, cfg, route, device="cpu"):
        if route not in Q_SOURCES or device != "cpu":
            raise ValueError("Only CPU S1/S2 are authorized")
        import torch
        self.cfg, self.route, self.device = cfg, route, device
        cache = Path(cfg["cache_root"]).resolve()
        self.sequences = np.load(cache / "sequences.npy", mmap_mode="r", allow_pickle=False)
        self.q = np.load(cache / ("quality_" + route + ".npy"), mmap_mode="r", allow_pickle=False)
        self.div = np.load(cache / "div420.npy", mmap_mode="r", allow_pickle=False)
        self.cluster = np.load(cache / "cluster.npy", mmap_mode="r", allow_pickle=False)
        n = len(self.sequences)
        if n != cfg["pool_size"] or self.q.shape != (n,) or self.div.shape != (n, 420) or self.cluster.shape != (n,):
            raise ValueError("Invalid prepared Pool mapping")
        if not np.isfinite(self.q).all() or np.any(self.q < 0) or np.any(self.q > 1):
            raise ValueError("Raw q must be finite within [0,1]; no implicit scaling")
        self.ids = np.random.default_rng(cfg["feature_seed"] + 1).permutation(n)
        self.n, self.k, self.g = n, cfg["subset_size"], cfg["group_size"]
        if not 0 < self.k <= self.n:
            raise ValueError("Invalid subset size")
        self.ng = (n + self.g - 1) // self.g
        embedding = np.load(cache / "order64.npy", mmap_mode="r", allow_pickle=False)
        x = np.asarray(embedding[self.ids], dtype=np.float32)
        if x.shape != (n, 64) or not np.isfinite(x).all():
            raise ValueError("Invalid preserved actor representation")
        self.x, self.d = torch.as_tensor(x, device=device), 64
        padded = torch.zeros(self.ng * self.g, self.d, device=device)
        padded[:n] = self.x
        self.padded = padded.view(self.ng, self.g, self.d)
        sizes = np.minimum(self.g, n - np.arange(self.ng) * self.g)
        self.sizes = torch.tensor(sizes, device=device, dtype=torch.float32)
        self.gx = self.padded.sum(1) / self.sizes[:, None]
        self.clusters_local = self.cluster[self.ids]

    def score(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[None, :] if ids.ndim == 1 else ids
        if (ids.ndim != 2 or not ids.size or np.any(ids < 0) or np.any(ids >= self.n)
                or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0)):
            raise ValueError("Invalid subset IDs")
        original = self.ids[ids]
        quality = np.asarray(self.q[original], dtype=np.float32).mean(1)
        vectors = np.asarray(self.div[original], dtype=np.float32)
        similarities = np.clip(np.einsum("bkd,bjd->bkj", vectors, vectors), 0, 1)
        tri = np.triu_indices(ids.shape[1], 1)
        pairs = similarities[:, tri[0], tri[1]]
        diversity = (1 - pairs).mean(1) if pairs.shape[1] else np.zeros(len(ids))
        redundancy = np.clip((pairs - .92) / .08, 0, 1).mean(1) if pairs.shape[1] else np.zeros(len(ids))
        coverage = np.array([len(np.unique(c)) / len(c) for c in self.cluster[original]])
        components = np.column_stack([quality, diversity, coverage, redundancy]).astype(np.float32)
        reward = (components[:, 0] + components[:, 1] + components[:, 2]
                  + np.float32(1) - components[:, 3]) / np.float32(4)
        if not np.isfinite(components).all() or not np.isfinite(reward).all():
            raise FloatingPointError("Non-finite equal-weight reward")
        return reward.astype(np.float32), components

    def uniform(self, rng, batch):
        return np.array([rng.choice(self.n, self.k, replace=False) for _ in range(batch)], dtype=np.int64)

def replace_actor_only(pool, actor, torch_module):
    """No reward method replacement; preserve reward/index object identities."""
    identities = {name: getattr(pool, name) for name in
                  ("q", "div", "cluster", "ids", "sizes", "clusters_local")}
    probes = pool.uniform(np.random.default_rng(2026102701), 512)
    before_reward, before_components = pool.score(probes)
    x = np.asarray(actor[pool.ids], dtype=np.float32)
    if x.shape != (pool.n, 64) or not np.isfinite(x).all() or np.any(x[:, 40:] != 0):
        raise ValueError("Expected finite C3 features with exact zero padding")
    pool.x = torch_module.as_tensor(x, device="cpu")
    padded = torch_module.zeros(pool.ng * pool.g, pool.d, device="cpu")
    padded[:pool.n] = pool.x
    pool.padded = padded.view(pool.ng, pool.g, pool.d)
    pool.gx = pool.padded.sum(1) / pool.sizes[:, None]
    after_reward, after_components = pool.score(probes)
    if any(getattr(pool, name) is not value for name, value in identities.items()):
        raise AssertionError("Reward/index/group field was replaced")
    if before_reward.tobytes() != after_reward.tobytes() or before_components.tobytes() != after_components.tobytes():
        raise AssertionError("Reward or its components changed under actor replacement")
    return dict(reward_bitwise_identical=True, components_bitwise_identical=True,
        index_group_membership_identical=True, audit_subsets=512, audit_seed=2026102701,
        audit_excluded_from_training_and_candidates=True, reward_function_inherited_unchanged=True)

VERSION = "porto2018_raw_fitness_v1"

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

ANGLE_DEGREES = 100.0

FORMULA = "hypot(sum(H_i*cos(i*100*pi/180)),sum(H_i*sin(i*100*pi/180)))/sum(exp(a_i)); i=1..L"

H_VALUES = (
    .25, .04, -.72, -.62, .61, .16, -.40, .73, -1.10, .53,
    .26, -.64, -.07, -.69, -1.80, -.26, -.18, .54, .37, .02,
)

HELIX_COST_VALUES = (
    .00, .68, .69, .40, .54, 1.00, .61, .41, .26, .21,
    .24, .65, 3.16, .39, .21, .50, .66, .61, .49, .53,
)

HYDROPHOBICITY = MappingProxyType(dict(zip(AA_ORDER, H_VALUES)))

HELIX_COST = MappingProxyType(dict(zip(AA_ORDER, HELIX_COST_VALUES)))

_PARAMETERS = {
    "version": VERSION,
    "amino_acid_order": AA_ORDER,
    "hydrophobicity": list(H_VALUES),
    "helix_cost": list(HELIX_COST_VALUES),
    "angle_degrees": ANGLE_DEGREES,
    "formula": FORMULA,
    "residue_index_origin": 1,
    "transform": "none",
    "exponent_convention": "exp of published kcal/mol numerical values; no RT conversion",
}

PARAMETER_SHA256 = hashlib.sha256(json.dumps(
    _PARAMETERS, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode("utf-8")).hexdigest()

_PROVENANCE = {
    "parameters": _PARAMETERS,
    "parameter_sha256": PARAMETER_SHA256,
    "equation_source": {
        "doi": "10.1038/s41467-018-03746-3",
        "url": "https://www.nature.com/articles/s41467-018-03746-3",
        "location": "Methods, Fitness function, Equation 1 and following explanation",
    },
    "hydrophobicity_source": {
        "doi": "10.1039/FS9821700109",
        "url": "https://pubs.rsc.org/en/content/articlehtml/1982/fs/fs9821700109",
        "author_copy": "https://eisenberglab.mbi.ucla.edu/Reprints/40%20Eisenberg,%20Hydrophobic%20Moments%20and%20Protein%20Structure,%201982.pdf",
        "location": "Eisenberg et al. 1982, Table 1, p.111, final consensus column",
        "range": [-1.80, .73],
        "note": "Not Kyte--Doolittle, not the later normalized Eisenberg scale, and not the Arg=-1.76 variant",
    },
    "helix_cost_source": {
        "doi": "10.1016/S0006-3495(98)77529-0",
        "url": "https://pubmed.ncbi.nlm.nih.gov/9649402/",
        "location": "Pace and Scholtz 1998, abstract/Table 4 plus Pro=3.16 in Table 3, p.425",
        "unit_of_source_values": "kcal/mol relative to alanine",
        "single_value_convention": {"H": .61, "D": .69, "E": .40},
        "note": "Table 3 also provides ionization-state-specific alternatives; none are inferred from input sequences",
    },
    "interpretation": "Amphipathic alpha-helix-oriented heuristic, not AMP probability or MIC prediction",
    "transfer_limitations": [
        "Porto's original design used 20-residue Pg-AMP1 fragments; application to other full lengths is a new use",
        "No genetic algorithm, crossover, mutation, amino-acid removal, or CAMP-guided iteration selection is applied",
        "The source paper's full Perl parameter dictionary was not independently verified",
        "No clipping, pool min-max, additional division by length, or learned predictor is used",
    ],
    "computation": "float64 arithmetic; optional float32 conversion only after computing raw fitness",
}

def provenance():
    """Return a JSON-safe independent copy of the fixed scientific provenance."""
    return copy.deepcopy(_PROVENANCE)

def _sequence_text(sequence):
    if isinstance(sequence, bytes):
        try:
            sequence = sequence.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Sequence bytes must be ASCII canonical amino acids") from exc
    if not isinstance(sequence, str):
        raise TypeError("A sequence must be str or ASCII bytes")
    if not sequence:
        raise ValueError("A sequence must contain at least one residue")
    invalid = set(sequence).difference(AA_ORDER)
    if invalid:
        raise ValueError("Noncanonical amino acids or untrimmed input: " + repr(sorted(invalid)))
    return sequence

def porto_fitness(sequence):
    """Compute one raw fitness as a Python float using float64 arithmetic.

    Input must be a nonempty canonical, uppercase sequence.  No case folding,
    whitespace stripping, gap removal, truncation, or terminal modification is
    inferred.  The formula itself accepts any positive sequence length.
    """
    sequence = _sequence_text(sequence)
    theta = math.radians(ANGLE_DEGREES)
    cosine_sum = math.fsum(HYDROPHOBICITY[aa] * math.cos(i * theta)
                           for i, aa in enumerate(sequence, 1))
    sine_sum = math.fsum(HYDROPHOBICITY[aa] * math.sin(i * theta)
                         for i, aa in enumerate(sequence, 1))
    denominator = math.fsum(math.exp(HELIX_COST[aa]) for aa in sequence)
    result = math.hypot(cosine_sum, sine_sum) / denominator
    if not math.isfinite(result) or result < 0:
        raise ArithmeticError("Non-finite or negative raw Porto fitness")
    return result

def _output_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Output dtype must be float32 or float64")
    return dtype

def _chunk_values(sequences, dtype):
    sequences = [_sequence_text(sequence) for sequence in sequences]
    width = max(map(len, sequences))
    h = np.zeros((len(sequences), width), dtype=np.float64)
    exp_a = np.zeros_like(h)
    h_lookup = np.zeros(128, dtype=np.float64)
    exp_lookup = np.zeros(128, dtype=np.float64)
    for aa, value_h, value_a in zip(AA_ORDER, H_VALUES, HELIX_COST_VALUES):
        h_lookup[ord(aa)] = value_h
        exp_lookup[ord(aa)] = math.exp(value_a)
    for row, sequence in enumerate(sequences):
        indices = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        h[row, :len(indices)] = h_lookup[indices]
        exp_a[row, :len(indices)] = exp_lookup[indices]
    angles = np.arange(1, width + 1, dtype=np.float64) * math.radians(ANGLE_DEGREES)
    # Padding contributes zero to both sums; it is not an extra Ala (exp(0)=1).
    numerator = np.hypot(np.sum(h * np.cos(angles), axis=1, dtype=np.float64),
                         np.sum(h * np.sin(angles), axis=1, dtype=np.float64))
    denominator = np.sum(exp_a, axis=1, dtype=np.float64)
    result = numerator / denominator
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ArithmeticError("Non-finite or negative raw Porto fitness")
    return result.astype(dtype, copy=False)

def iter_porto_fitness_chunks(sequences, chunk_size=4096, dtype=np.float64):
    """Yield ``(start_index, 1D_values)`` in input order, without loading a pool.

    ``sequences`` may be an iterable, including a NumPy byte-string array or a
    generator.  Empty collections yield no chunks.  Invalid sequences raise an
    exception: a caller writing streamed caches should commit them atomically
    only after this iterator finishes successfully.
    """
    if isinstance(sequences, (str, bytes)):
        raise TypeError("Expected a collection of sequences, not a single string")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    dtype = _output_dtype(dtype)
    source = iter(sequences)
    start = 0
    while True:
        chunk = list(itertools.islice(source, int(chunk_size)))
        if not chunk:
            return
        values = _chunk_values(chunk, dtype)
        yield start, values
        start += len(chunk)

def porto_fitness_array(sequences, chunk_size=4096, dtype=np.float64):
    """Compute a 1D array in input order; default float64, optional float32."""
    dtype = _output_dtype(dtype)
    chunks = [values for _, values in iter_porto_fitness_chunks(sequences, chunk_size, dtype)]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)
