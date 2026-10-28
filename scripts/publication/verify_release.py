"""Audit public bytes and recompute saved arithmetic; standard library only.

Run in the published repository with Python 3. No torch, CUDA, network, private
inputs or holdout access. This verifies an archive, not a training replication.
"""
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {'.venv', 'artifacts', 'data', 'private', '__pycache__', '.git'}
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.onnx', '.parquet', '.arrow', '.npy', '.npz'}
PROFILE = re.compile(r'(?i)[a-z]:[\\/]+Users[\\/]+(?!USER(?:[\\/]|["\'\r\n]|$))([^\\/\r\n"\'<>]+)')
GPU_UUID = re.compile(r'\bGPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and not FORBIDDEN.intersection(p.parts), 'Forbidden path: ' + name)
    require(p.suffix.lower() not in FORBIDDEN_SUFFIXES, 'Forbidden binary: ' + name)


def privacy(data, name):
    if name.endswith('.zip'):
        return
    text = data.decode('utf-8')
    require(not PROFILE.search(text) and not GPU_UUID.search(text), 'Unredacted identifier: ' + name)


def verify_bytes(data, record, name):
    require(len(data) == record['bytes'] and digest(data) == record['sha256'], 'Byte/SHA mismatch: ' + name)


def main():
    publication = read('PUBLICATION_MANIFEST.json')
    files = publication['files']
    require('PUBLICATION_MANIFEST.json' not in files, 'Manifest self-reference')
    for name, record in files.items():
        safe(name)
        data = (ROOT / name).read_bytes()
        verify_bytes(data, record, name)
        privacy(data, name)
    zip_members = 0
    for phase in ('V01', 'V02'):
        name = phase + '_PUBLIC_REVIEW_BUNDLE.zip'
        with zipfile.ZipFile(ROOT / name) as z:
            require(z.testzip() is None, 'ZIP CRC failure: ' + name)
            manifest = json.loads(z.read('PUBLIC_BUNDLE_MANIFEST.json'))
            names = z.namelist()
            require(len(names) == len(set(names)), 'Duplicate ZIP path')
            require(set(names) == set(manifest['files']) | {'PUBLIC_BUNDLE_MANIFEST.json'}, 'ZIP manifest coverage')
            for member in names:
                safe(member)
                data = z.read(member)
                privacy(data, member)
                if member in manifest['files']:
                    verify_bytes(data, manifest['files'][member], name + ':' + member)
            zip_members += len(names)
        expected = (ROOT / (name + '.sha256')).read_text(encoding='ascii').split()[0]
        require(digest((ROOT / name).read_bytes()) == expected, 'External checksum mismatch')
    # These are saved scalar JSON records. No tensor or training code is loaded.
    v02 = read('runs/v02_20261022_193903/metrics_raw.json')
    v01 = read('runs/v01_20261022/metrics_raw.json')
    index = {(r['seed'], r['condition'], r['step']): r for r in v02}
    previous = {(r['seed'], r['condition'], r['step']): r for r in v01}
    require(len(v02) == len(index) == 24, 'Expected 24 unique V02 evaluations')
    differences = []
    final = []
    for seed in (1729, 2718):
        for condition in ('C', 'U', 'RS', 'S'):
            for step in (0, 100, 767):
                row = index[seed, condition, step]
                require(row['dev_valid_targets'] == 62601 and row['train_valid_targets'] == step * 508, 'Target denominator')
                require(row['nll'] == row['nll_sum'] / row['dev_valid_targets'], 'NLL numerator/denominator')
                require(row['context_gain'] == row['dev_unigram_nll'] - row['nll'], 'Unigram gain')
                if condition in ('C', 'S'):
                    old_condition = {'C': 'Fp', 'S': 'Sp'}[condition]
                    differences.append(abs(row['nll'] - previous[seed, old_condition, step]['nll']))
        values = {c: index[seed, c, 767]['nll'] for c in ('C', 'U', 'RS', 'S')}
        C, U, RS, S = (values[c] for c in ('C', 'U', 'RS', 'S'))
        d = {'seed': seed, 'G': S - C, 'D_r': RS - C, 'D_1': S - U,
             'N_cont': U - C, 'N_sign': S - RS, 'I': (S - U) - (RS - C)}
        require(d['G'] == d['N_sign'] + d['D_r'] == d['D_1'] + d['N_cont'], 'Difference identity')
        final.append(d)
    with (ROOT / 'reports/v02/contrasts.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 6, 'Expected six contrast rows')
    for expected in final:
        row = next(r for r in rows if int(r['seed']) == expected['seed'] and int(r['step']) == 767)
        require(all(float(row[k]) == v for k, v in expected.items() if k != 'seed'), 'Saved final contrasts')
    require(max(differences) == 0.0 and len(differences) == 12, 'V01/V02 bridge')
    screen = read('reports/v02/screening.json')
    require(screen['statuses']['NORM_DIRECTION'] == 'DIRECTION_RESIDUAL_AT_BOTH_NORMS', 'Screen label')
    require(all(r['D_r'] > .05 and r['D_1'] > .05 for r in final), 'Predeclared B predicates')
    budget = read('reports/v02/gpu_budget.json')
    require(math.isclose(budget['project_cumulative_seconds'], 332.12303610004624 + budget['new_cumulative_seconds'], rel_tol=0, abs_tol=1e-9), 'GPU ledger arithmetic')
    print(json.dumps({'status': 'PASS', 'public_files_verified': len(files), 'public_zip_members_verified': zip_members,
                      'v02_evaluations': len(v02), 'bridge_pairs': len(differences), 'bridge_max_absolute_error': max(differences),
                      'final_contrasts': final, 'scope': 'existing evidence only; no training, network, CUDA, tensors, private inputs or holdout'}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, StopIteration, zipfile.BadZipFile) as exc:
        print('FAIL: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
