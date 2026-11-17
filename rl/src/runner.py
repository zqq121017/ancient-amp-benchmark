"""Training schedule and candidate checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROUTES = {"S1": "S1", "S2": "S2"}

TORCH = None

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()

def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False)
    if exclusive:
        with path.open("x", encoding="utf-8") as f:
            f.write(data)
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)

def schedule(cfg, spec):
    checkpoints = list(spec.get("checkpoints", cfg["checkpoints"]))
    initial = int(spec.get("initial_calls", cfg.get("initial_calls", 512)))
    probe = int(spec.get("final_calls", cfg.get("final_calls", 1024)))
    batch = int(cfg.get("batch_size", 32))
    if not checkpoints or checkpoints != sorted(set(checkpoints)) or min(initial, probe, batch) < 1:
        raise ValueError("Invalid query schedule")
    prior = initial
    for budget in checkpoints:
        if budget - probe <= prior or (budget - probe - prior) % batch:
            raise ValueError("Each checkpoint needs positive batch-aligned adaptive calls")
        prior = budget
    return checkpoints, initial, probe, batch

def specs(cfg):
    seeds = cfg.get("seeds", [17, 29, 43])
    main = [dict(route=r, method=m, seed=s, phase="main")
            for r in ROUTES for m in ("ppo", "tb") for s in seeds]
    random = [dict(route=r, method="random", seed=s, phase="controls")
              for r in ("S1", "S2") for s in seeds]
    return main + random

def run_name(spec):
    return "{}_{}_seed{}".format(spec["method"], spec["route"], int(spec["seed"]))

def normalize_spec(spec):
    result = dict(spec)
    result.setdefault("phase", "controls" if result.get("method") == "random" else "main")
    return result

def guard(cfg, deadline):
    limits, root = cfg["resource_limits"], Path(cfg["output_root"])
    while True:
        if time.time() >= deadline:
            raise TimeoutError("Stage deadline reached; partial results retained")
        free = shutil.disk_usage(root).free / 2 ** 30
        reasons = ["disk"] if free < limits.get("min_free_disk_gib", 80) else []
        load, available, rss = None, None, None
        if Path("/proc/meminfo").exists():
            mem = {s.split(":")[0]: int(s.split()[1]) for s in Path("/proc/meminfo").read_text().splitlines()}
            available, load = mem["MemAvailable"] / 2 ** 20, os.getloadavg()[0]
            if available < limits.get("min_available_ram_gib", 48):
                reasons.append("RAM")
            if load > limits.get("max_load1", 18):
                reasons.append("load")
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 2 ** 20
            if rss is not None and rss > limits.get("max_child_rss_gib", 6):
                raise MemoryError("Runner RSS cap exceeded")
        if not reasons:
            return
        save(root / "resource_wait.json", dict(time_unix=time.time(), reasons=reasons,
             free_disk_gib=free, load1=load, available_ram_gib=available, rss_gib=rss))
        time.sleep(10)

def record_selection(routes, pool, directory, name, chosen, spec, budget):
    rows = routes.candidate_rows(pool, chosen)
    csv_path, meta_path = directory / (name + ".csv"), directory / (name + ".json")
    routes.export_candidates(csv_path, rows, spec["route"], spec["method"], spec["seed"], role=name)
    save(meta_path, dict(run_spec=spec, budget=budget, output_rule=name,
         budget_unit="full_pool_sequences" if spec["method"] == "full_pool_direct" else "subset_reward_calls",
         actual_n=len(rows), requested_n=pool.cfg["candidate_count"], candidates=rows,
         no_predictor_scores_used_for_selection=True, experimentally_validated=False), exclusive=True)
    return dict(csv_sha256=sha(csv_path), json_sha256=sha(meta_path), actual_n=len(rows))

def run_one(cfg, spec, deadline):
    spec = normalize_spec(spec)
    if spec["route"] not in ROUTES or spec["method"] not in ("ppo", "tb", "random"):
        raise ValueError("Unrecognized route/method")
    if spec.get("phase", "main") not in ("main", "controls", "smoke"):
        raise ValueError("Unrecognized output phase")
    if spec.get("phase", "main") != "smoke":
        if spec not in specs(cfg):
            raise ValueError("Only prespecified scientific run specifications are accepted")
    evidence = verify_inputs(cfg)
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    checkpoints, initial_n, probe_n, batch = schedule(cfg, spec)
    destination = root / spec.get("phase", "main") / run_name(spec)
    destination.mkdir(parents=True, exist_ok=False)  # also prevents duplicate workers
    save(destination / "RUNNING.json", dict(pid=os.getpid(), spec=spec, started=time.time()), exclusive=True)
    routes, core, engine = load_modules(cfg)
    torch = TORCH
    guard(cfg, deadline)
    pool = routes.Pool(cfg, ROUTES[spec["route"]], "cpu")
    replacement = None
    rng = np.random.default_rng(int(spec["seed"]))
    torch.manual_seed(int(spec["seed"]))
    learned = spec["method"] != "random"
    initial = pool.uniform(rng, initial_n)
    reward, components = pool.score(initial)
    lw = float(cfg["reward"]["beta"]) * (reward.astype(np.float64) - .5)
    weights = np.exp(lw - lw.max())
    initial_logz = core.logchoose(pool.n, pool.k) + float(lw.max() + np.log(weights.mean()))
    model = core.Policy(pool) if learned else None
    effective_spec = dict(spec, tb_updates_per_batch=cfg.get("tb_updates_per_batch", 4),
                          logz_learning_rate=cfg.get("logz_learning_rate", .0003))
    optimizer = engine.optimizer_for(model, cfg, effective_spec) if learned else None
    if learned:
        with torch.no_grad():
            model.logz.fill_(initial_logz)
        actor_before = engine.actor_vector(model)
    ids_parts, reward_parts, component_parts = [initial], [reward], [components]
    phase_parts = [np.zeros(initial_n, dtype=np.uint8)]
    calls, updates, probes = initial_n, 0, 0
    start, summaries = time.time(), {}
    with (destination / "history.jsonl").open("x", encoding="utf-8") as history:
        for budget in checkpoints:
            last_update = None
            while calls < budget - probe_n:
                guard(cfg, deadline)
                actions = model.sample(batch) if learned else pool.uniform(rng, batch)
                reward, components = pool.score(actions)
                if learned:
                    last_update = engine.update_model(model, optimizer, spec["method"], actions, reward, cfg, effective_spec, core)
                    updates += last_update["optimizer_steps"]
                ids_parts.append(actions); reward_parts.append(reward); component_parts.append(components)
                phase_parts.append(np.ones(batch, dtype=np.uint8)); calls += batch
                row = dict(calls=calls, budget=budget, optimizer_steps=updates,
                           reward_mean=float(reward.mean()), seconds=time.time()-start, update=last_update)
                history.write(json.dumps(row, allow_nan=False) + "\n")
                if calls % 256 == 0:
                    history.flush()
                    save(root / "progress.json", dict(run=run_name(spec), **row))
            probe_seed = int(cfg.get("frozen_sampling_seed", 2026102301)) + int(spec["seed"])*100000 + budget
            rng_before = torch.random.get_rng_state().clone()
            final = engine.independently_seeded_draws(pool, model, probe_n, probe_seed, batch)
            if not torch.equal(rng_before, torch.random.get_rng_state()):
                raise AssertionError("Frozen probe altered training RNG")
            reward, components = pool.score(final)
            ids_parts.append(final); reward_parts.append(reward); component_parts.append(components)
            phase_parts.append(np.full(probe_n, 2, dtype=np.uint8)); calls += probe_n; probes += probe_n
            if calls != budget:
                raise AssertionError("Budget mismatch")
            current = destination / ("checkpoint_" + str(budget)); current.mkdir()
            all_ids = np.concatenate(ids_parts)
            np.savez_compressed(current / "query_ledger.npz", original_ids=pool.ids[all_ids].astype(np.uint32),
                 reward=np.concatenate(reward_parts), components=np.concatenate(component_parts), phase=np.concatenate(phase_parts))
            choices = {"native": routes.select_native(final, cfg["candidate_count"]),
                       "queried_direct": routes.select_direct(pool, all_ids, cfg["candidate_count"])}
            hashes = {key: record_selection(routes, pool, current, key, selected, spec, budget) for key, selected in choices.items()}
            audit = engine.fixed_audit(pool, model, cfg, core, deadline) if learned else None
            if learned:
                torch.save(dict(state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     torch_rng_state=torch.random.get_rng_state(), numpy_rng_state=rng.bit_generator.state,
                     spec=effective_spec, initial_logz=initial_logz, optimizer_steps=updates, evidence=evidence), current / "policy_and_optimizer.pt")
            artifacts = {"query_ledger.npz": sha(current / "query_ledger.npz")}
            if learned:
                artifacts["policy_and_optimizer.pt"] = sha(current / "policy_and_optimizer.pt")
            summary = dict(spec=effective_spec, budget=budget, pool_size=pool.n, candidate_hashes=hashes,
                 adaptive_calls=budget-initial_n-probes, cumulative_probe_calls=probes, optimizer_steps=updates,
                 unique_queried_sequences=len(np.unique(all_ids)), final_mean_reward=float(reward.mean()),
                 final_components_mean=components.mean(0).tolist(), diagnostics=audit, replacement_invariance=replacement,
                 actor_max_parameter_change=float((engine.actor_vector(model)-actor_before).abs().max()) if learned else None,
                 initial_logz=initial_logz, initial_mc_ess=float(weights.sum()**2/(weights@weights)),
                 last_update=last_update, seconds=time.time()-start, evidence=evidence,
                 audit_calls_separate=int(cfg.get("diagnostics", {}).get("audit_subsets", 512)) if learned else 0,
                 no_AMP_predictors_read=True, convergence_certified=False,
                 artifact_hashes=artifacts, query_ledger_sha256=artifacts["query_ledger.npz"],
                 checkpoint_sha256=artifacts.get("policy_and_optimizer.pt"))
            save(current / "summary.json", summary, exclusive=True)
            save(current / "CANDIDATES_FROZEN.json", dict(candidate_hashes=hashes, artifact_hashes=artifacts,
                 summary_sha256=sha(current/"summary.json")), exclusive=True)
            summaries[str(budget)] = sha(current / "summary.json")
    if verify_inputs(cfg) != evidence:
        raise AssertionError("Immutable inputs changed during training")
    result = dict(spec=spec, complete=True, seconds=time.time()-start, checkpoint_summaries=summaries,
                  evidence=evidence, excluded_from_main=spec.get("phase") == "smoke")
    save(destination / "summary.json", result, exclusive=True)
    return result
