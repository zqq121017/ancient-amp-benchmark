"""PPO/GFlowNet search with Quan C3 inputs. Validation is the default action."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

# The CLI sets native-library limits before importing NumPy or Torch. Setting
# these only inside manual_run would be too late for an initialized BLAS pool.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
MODULES = Path(__file__).resolve().parent
COUNTS = {"gut": 118583, "oral": 387477, "mammoth": 287622}
EXPORT_COUNTS = {"gut": 111, "oral": 76, "mammoth": 50}
_MODULES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("A lowercase SHA256 digest is required")
    return value


def verify_source_snapshot():
    files = read(PACKAGE / "configs/experiment.json")["modules"]
    for name, digest in files.items():
        if Path(name).name != name or not name.endswith(".py") or sha(MODULES / name) != _digest(digest):
            raise ValueError("Search code changed: " + name)
    return files


def load_module(name):
    if name not in _MODULES:
        if name not in ("agent", "runner", "rewards", "outputs"):
            raise ValueError("Unknown search module")
        spec = importlib.util.spec_from_file_location("peptide_search_" + name, MODULES / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def dataset_config(dataset):
    settings = read(PACKAGE / "configs/experiment.json")
    return dict(settings["defaults"], **settings["datasets"][dataset])


def make_config(dataset, inputs_path, output_root=None):
    if dataset not in COUNTS:
        raise ValueError("Unknown dataset")
    cfg = dataset_config(dataset)
    inputs_path = Path(inputs_path).resolve()
    inputs = read(inputs_path)
    if inputs.get("dataset") != dataset:
        raise ValueError("Input manifest dataset differs")
    if cfg.get("pool_size") != COUNTS[dataset]:
        raise ValueError("The complete original pool is required")
    def resolve(field):
        value = inputs.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError("Missing input path: " + field)
        p = Path(value)
        return str((inputs_path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    cfg.update(cache_root=resolve("cache_root"), actor_path=resolve("actor_path"),
        row_sequences_path=resolve("row_sequences_path"),
        input_manifest_path=str(inputs_path), input_manifest_sha256=sha(inputs_path),
        input_manifest=inputs, source_project=str(MODULES),
        optimizer_engine=str(MODULES / "agent.py"),
        output_root=str(Path(output_root).resolve()) if output_root else None)
    return cfg


def verify_inputs(cfg):
    """Read-only full hash, schema, row mapping and actor checks; no optimizer."""
    sources = verify_source_snapshot()
    dataset = cfg["dataset"]
    n = COUNTS[dataset]
    # Every scientific configuration field is fixed by the supplied per-pool file.
    reference = dataset_config(dataset)
    if any(cfg.get(key) != value for key, value in reference.items()):
        raise ValueError("Configuration differs from the released experiment")
    if sha(cfg["input_manifest_path"]) != cfg["input_manifest_sha256"]:
        raise ValueError("Input manifest changed")
    manifest = cfg["input_manifest"]
    cache = Path(cfg["cache_root"])
    pinned = reference["cache_manifest_sha256"]
    if manifest.get("cache_manifest_sha256") != pinned or sha(cache / "manifest.json") != pinned:
        raise ValueError("Expected the frozen final Porto cache manifest")
    cache_manifest = read(cache / "manifest.json")
    base = load_module("rewards")
    if cache_manifest.get("n") != n or not set(base.CACHE_FILES).issubset(cache_manifest.get("files", {})):
        raise ValueError("Incomplete cache manifest")
    hashes = {}
    for filename in base.CACHE_FILES:
        expected = _digest(cache_manifest["files"][filename])
        if sha(cache / filename) != expected:
            raise ValueError("Frozen cache changed: " + filename)
        hashes[filename] = expected
    base.verify_cache_structure(cache, n)
    actor_path, rows_path = Path(cfg["actor_path"]), Path(cfg["row_sequences_path"])
    actor_sha = _digest(manifest.get("actor_sha256"))
    row_sha = _digest(manifest.get("row_sequences_sha256"))
    final_inputs = read(PACKAGE / "configs/experiment.json")["inputs"][dataset]
    if actor_sha != final_inputs["actor_sha256"] or row_sha != final_inputs["sequences_sha256"]:
        raise ValueError("C3/row hashes differ from the completed experiment")
    if sha(actor_path) != actor_sha or sha(rows_path) != row_sha:
        raise ValueError("C3 or row mapping hash mismatch")
    actor = np.load(actor_path, allow_pickle=False, mmap_mode="r")
    rows = np.load(rows_path, allow_pickle=False, mmap_mode="r")
    seqs = np.load(cache / "sequences.npy", allow_pickle=False, mmap_mode="r")
    if actor.shape != (n, 64) or actor.dtype != np.dtype("float32"):
        raise ValueError("Expected float32 C3 padded to shape (N,64)")
    if rows.shape != (n,) or rows.dtype != np.dtype("S50"):
        raise ValueError("C3 row mapping must be S50 with one sequence per row")
    for start in range(0, n, 4096):
        end = min(start + 4096, n)
        if not np.array_equal(rows[start:end], seqs[start:end]):
            raise ValueError("C3 sequence rows differ from the frozen cache")
        block = actor[start:end]
        if not np.isfinite(block).all() or np.any(block[:, 40:] != 0):
            raise ValueError("Invalid C3 values or nonzero padding")
    if cfg["output_root"] is not None:
        out = Path(cfg["output_root"])
        for protected in (PACKAGE, cache, actor_path.parent, rows_path.parent):
            if out == protected or protected in out.parents or out in protected.parents:
                raise ValueError("Output must be a separate new tree from code and inputs")
    return dict(dataset=dataset, pool_size=n, source_sha256=sources,
        cache_sha256=hashes, cache_manifest_sha256=pinned,
        actor_sha256=actor_sha, row_sequences_sha256=row_sha,
        representation="Quan C3 mean40; float32; zero-pad64; no refit or L2",
        reward_formula=base.REWARD_FORMULA, q_sources=base.Q_SOURCES,
        primary_output="queried_direct", primary_checkpoint=8192,
        export_prefix_count=EXPORT_COUNTS[dataset], no_predictor_scores_used=True,
        release_wrapper_sha256=sha(__file__),
        input_manifest_sha256=cfg["input_manifest_sha256"],
        configuration_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest())


def pool_class():
    """Retain the exact final Porto Pool and exact C3 actor replacement."""
    base = load_module("rewards")
    c3 = load_module("rewards")
    class QuanC3Pool(base.PortoEqualPool):
        def __init__(self, cfg, route, device="cpu"):
            super().__init__(cfg, route, device)
            import torch
            actor = np.load(cfg["actor_path"], allow_pickle=False, mmap_mode="r")
            self.representation_invariance = c3.replace_actor_only(self, actor, torch)
    return QuanC3Pool


def load_runtime():
    verify_source_snapshot()
    runner = load_module("runner")
    if getattr(runner, "_portable_wired", False):
        return runner
    cls = pool_class()
    def load_modules(cfg):
        import torch
        runner.TORCH = torch
        agent, outputs = load_module("agent"), load_module("outputs")
        outputs.Pool = cls
        agent.guard = runner.guard
        torch.set_num_threads(int(cfg["resource_limits"].get("cpu_threads", 2)))
        if hasattr(os, "nice"):
            current = os.nice(0)
            if current < 19:
                os.nice(19-current)
        return outputs, agent, agent
    runner.verify_inputs = verify_inputs
    runner.load_modules = load_modules
    runner._portable_wired = True
    return runner


def manual_run(cfg, route, method, seed, max_hours=2.0):
    """Explicit one-run entry only; called by --action run, never validation."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Training entry requires Linux for resource checks")
    if route not in ("S1", "S2") or method not in ("ppo", "tb", "random") or seed not in (17,29,43):
        raise ValueError("Only the frozen route/method/seed combinations are accepted")
    if not 0 < max_hours <= 2:
        raise ValueError("Per-run wall cap must be positive and at most two hours")
    if cfg["output_root"] is None:
        raise ValueError("--output is required for manual training")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    current = os.nice(0)
    if current < 19:
        os.nice(19-current)
    evidence = verify_inputs(cfg)
    output = Path(cfg["output_root"])
    runner = load_runtime()
    # One dataset root may accumulate its prescribed specs. Only a recognized
    # root with identical input/config evidence may be reused. Per-spec folders
    # remain exclusive in original run_one, including partial/failed attempts.
    if output.exists():
        marker = output / "RELEASE_RUN_INPUTS.json"
        if not marker.is_file() or read(marker) != evidence:
            raise FileExistsError("Existing output root is unrecognized or has different inputs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        runner.save(output / "RELEASE_RUN_INPUTS.json", evidence, exclusive=True)
    spec = dict(route=route, method=method, seed=seed,
                phase="controls" if method == "random" else "main")
    # Serialize writers to shared progress/status files. A crashed lock is kept
    # for inspection; this entry never removes a lock belonging to another run.
    lock_path = output / "RELEASE_RUN_LOCK.json"
    lock = dict(pid=os.getpid(), token=os.urandom(16).hex(), spec=spec)
    runner.save(lock_path, lock, exclusive=True)
    try:
        return runner.run_one(cfg, spec, time.time() + max_hours*3600)
    finally:
        if lock_path.is_file() and read(lock_path) == lock:
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("validate", "run"), default="validate")
    parser.add_argument("--dataset", choices=tuple(COUNTS), required=True)
    parser.add_argument("--inputs", required=True, help="External input manifest JSON")
    parser.add_argument("--output", help="New independent directory; required only for run")
    parser.add_argument("--route", choices=("S1", "S2"))
    parser.add_argument("--method", choices=("ppo", "tb", "random"), help="tb = trajectory-balance GFlowNet")
    parser.add_argument("--seed", type=int, choices=(17,29,43))
    parser.add_argument("--max-hours", type=float, default=2.0)
    args = parser.parse_args()
    cfg = make_config(args.dataset, args.inputs, args.output)
    if args.action == "validate":
        result = verify_inputs(cfg)
    else:
        if args.route is None or args.method is None or args.seed is None or args.output is None:
            parser.error("run requires --route, --method, --seed and --output")
        result = manual_run(cfg, args.route, args.method, args.seed, args.max_hours)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
