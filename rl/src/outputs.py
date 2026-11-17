"""Candidate ranking and CSV export."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

def select_native(final_ids, limit=111):
    """First distinct final-policy draws. Never sort by quality or backfill."""
    seen, selected = set(), []
    for item in np.asarray(final_ids).ravel():
        item = int(item)
        if item not in seen:
            seen.add(item)
            selected.append(item)
        if len(selected) == limit:
            break
    return np.asarray(selected, dtype=np.int64)

def select_direct(pool, encountered, limit=111):
    ids = np.unique(np.asarray(encountered).ravel())
    ids = ids[np.argsort(pool.ids[ids], kind='stable')]
    order = np.argsort(-np.asarray(pool.q[pool.ids[ids]]), kind='stable')
    return ids[order[:limit]]

def candidate_rows(pool, chosen):
    rows = []
    for rank, index in enumerate(chosen, 1):
        original = int(pool.ids[index])
        sequence = pool.sequences[original]
        sequence = sequence.decode('ascii') if isinstance(sequence, (bytes, np.bytes_)) else str(sequence)
        rows.append({'rank': rank, 'pool_row_zero_based': original,
                     'original_row_zero_based': original,
                     'row_id': 'seq_' + hashlib.sha256(sequence.encode('ascii')).hexdigest()[:20],
                     'sequence': sequence, 'quality_prior': float(pool.q[original]),
                     'cluster': int(pool.cluster[original])})
    return rows

def export_candidates(path, rows, category, method, seed, role='native'):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['rank', 'row_id', 'sequence', 'category', 'method', 'seed', 'role',
              'pool_row_zero_based', 'quality_prior', 'cluster']
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, category=category, method=method, seed=seed, role=role))

COUNTS = {"gut": 111, "oral": 76, "mammoth": 50}

def selected_rows(meta, dataset, spec):
    if meta.get('budget') != 8192 or meta.get('output_rule') != 'queried_direct' or meta.get('run_spec') != spec:
        raise ValueError('Wrong fixed output rule, budget, or run')
    rows = meta['candidates']
    if meta.get('actual_n') != len(rows) or len(rows) > 111:
        raise ValueError('Candidate count mismatch')
    if len(rows) < COUNTS[dataset]:
        raise ValueError('Insufficient real candidates; do not backfill')
    sequences = [r['sequence'] for r in rows]
    if len(set(sequences)) != len(rows) or [r['rank'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Invalid frozen list order or duplicate sequence')
    for seq in sequences:
        if not 10 <= len(seq) <= 50 or not set(seq) <= set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError('Invalid ancient peptide')
    quality = [r['quality_prior'] for r in rows]
    if any(not isinstance(q,(int,float)) or not math.isfinite(q) or not 0 <= q <= 1 for q in quality):
        raise ValueError('Invalid fixed raw q')
    if any(a < b for a, b in zip(quality, quality[1:])):
        raise ValueError('Frozen list is not in queried-direct order')
    return sequences[:COUNTS[dataset]]

POOL_SIZES = {'gut':118583, 'oral':387477, 'mammoth':287622}

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b''):
            result.update(chunk)
    return result.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def verify_checkpoint(directory, spec, dataset):
    """Require a completed run and verify BOTH checkpoints before exporting."""
    run = read(directory / 'summary.json')
    if run.get('complete') is not True or run.get('spec') != spec:
        raise ValueError('Run is incomplete or specification differs: ' + str(directory))
    evidence = run.get('evidence', {})
    pin = read(Path(__file__).resolve().parents[1] / 'configs/experiment.json')['inputs'][dataset]
    # Bind the directory label to actual representation, pool and reward inputs.
    # Accept final historical evidence and this release's relocated evidence.
    historical = evidence.get('baseline_evidence', {})
    cache_hash = evidence.get('cache_manifest_sha256', historical.get('cache_manifest_sha256'))
    actor_hash = evidence.get('actor_sha256', evidence.get('representation_sha256'))
    cache_files = evidence.get('cache_sha256', historical.get('cache', {}))
    if (evidence.get('dataset') != dataset or evidence.get('pool_size') != POOL_SIZES[dataset]
            or cache_hash != pin['cache_manifest_sha256'] or actor_hash != pin['actor_sha256']
            or cache_files.get('sequences.npy') != pin['sequences_sha256']
            or evidence.get('reward_formula') != '(Q + D + C + 1 - U) / 4'):
        raise ValueError('Run dataset, C3 representation or reward cache differs from the final experiment')
    selected = None
    for budget in (4096, 8192):
        checkpoint = directory / f'checkpoint_{budget}'
        summary_file = checkpoint / 'summary.json'
        summary = read(summary_file)
        frozen = read(checkpoint / 'CANDIDATES_FROZEN.json')
        if (sha(summary_file) != run.get('checkpoint_summaries', {}).get(str(budget))
                or sha(summary_file) != frozen.get('summary_sha256')):
            raise ValueError('Checkpoint summary changed')
        effective = dict(spec, tb_updates_per_batch=4, logz_learning_rate=.0003)
        if (summary.get('spec') != effective or summary.get('budget') != budget
                or summary.get('evidence') != run.get('evidence')):
            raise ValueError('Checkpoint run/budget/evidence mismatch')
        required = {'query_ledger.npz', 'policy_and_optimizer.pt'}
        artifacts = frozen.get('artifact_hashes', {})
        if set(artifacts) != required or artifacts != summary.get('artifact_hashes'):
            raise ValueError('Missing ledger/checkpoint hashes')
        for name, digest in artifacts.items():
            if sha(checkpoint / name) != digest:
                raise ValueError('Changed artifact: ' + name)
        hashes = frozen.get('candidate_hashes', {})
        if set(hashes) != {'native', 'queried_direct'} or hashes != summary.get('candidate_hashes'):
            raise ValueError('Candidate hash manifest differs')
        for kind, digests in hashes.items():
            for suffix in ('json', 'csv'):
                if sha(checkpoint / f'{kind}.{suffix}') != digests.get(suffix + '_sha256'):
                    raise ValueError('Changed candidate file')
        if budget == 8192:
            source = checkpoint / 'queried_direct.json'
            selected = (selected_rows(read(source), dataset, spec), sha(source))
    return selected

def export(runs_root, destination):
    runs_root, destination = Path(runs_root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Output already exists; do not overwrite it')
    if destination == runs_root or destination in runs_root.parents or runs_root in destination.parents:
        raise ValueError('Export directory must be separate from run directories')
    records = []
    for dataset in COUNTS:
        for route in ('S1', 'S2'):
            for method in ('ppo', 'tb'):
                for seed in (17, 29, 43):
                    spec = dict(route=route, method=method, seed=seed, phase='main')
                    directory = runs_root / dataset / 'training/main' / f'{method}_{route}_seed{seed}'
                    sequences, digest = verify_checkpoint(directory, spec, dataset)
                    records.append((dataset, route, method, seed, sequences, digest))
    # All input checks finish before creating outputs.
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for dataset, route, method, seed, sequences, source_hash in records:
        folder = destination / ('data' + str(list(COUNTS).index(dataset) + 1))
        folder.mkdir(exist_ok=True)
        label = 'GFlowNet' if method == 'tb' else 'PPO'
        path = folder / f'{route}_{label}_seed{seed}.csv'
        with path.open('x', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['sequence'])
            writer.writerows([[seq] for seq in sequences])
        with path.open(encoding='utf-8', newline='') as handle:
            if [row['sequence'] for row in csv.DictReader(handle)] != sequences:
                raise ValueError('CSV readback failed')
        hashes[path.relative_to(destination).as_posix()] = dict(sha256=sha(path), n=len(sequences), source_json_sha256=source_hash)
    receipt = dict(complete=True, output_rule='queried_direct', budget=8192,
                   files=hashes, files_count=len(hashes), training_executed=False)
    with (destination / 'EXPORT_COMPLETE.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.runs_root, args.output)
    print(json.dumps({'files': result['files_count'], 'output': str(args.output)}))


if __name__ == "__main__":
    main()
