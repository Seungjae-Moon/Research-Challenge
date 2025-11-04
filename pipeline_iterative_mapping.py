#!/usr/bin/env python3
"""
Iterative mapping pipeline
- Generates data with `digital_twin_engine.py` using provided knobs
- Trains/evaluates with `train_eval_fusion.py`
- Uses simple hill-climb to accept parameter updates that reduce MAE

Usage: adjust params below or run directly. This script runs subprocesses and
writes a per-iteration JSON log 'pipeline_iter_log.json'.
"""
from __future__ import annotations
import subprocess, json, time, os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
DT_SCRIPT = ROOT / 'src' / 'digital_twin_engine.py'
TRAIN_SCRIPT = ROOT / 'train_eval_fusion.py'
RUNS_DIR = ROOT / 'runs'

# Configuration
units = 200
seed = 42
max_iters = 5
patience_no_improve = 3
out_log = ROOT / 'pipeline_iter_log.json'

def run_generation(set_id: str, cov_blend_lam: float, sensor14_scale: float, sensor14_offset: float,
                   op3_compress: bool, op3_low: float, op3_high: float):
    cmd = ["python", str(DT_SCRIPT),
           "--units", str(units), "--seed", str(seed), "--set-id", set_id,
           "--cov-blend-lam", str(cov_blend_lam),
           "--fd004-train", str(ROOT / 'data' / 'cmapss_raw' / 'train_FD004.txt'),
           "--fd004-rul", str(ROOT / 'data' / 'cmapss_raw' / 'RUL_FD004.txt'),
           "--train-root", str(ROOT / 'data' / 'training'),
           "--test-root", str(ROOT / 'data' / 'test'),
           "--rul-root", str(ROOT / 'data' / 'rul')]
    # sensor14 overrides
    if sensor14_scale is not None:
        cmd += ["--sensor14-scale", str(sensor14_scale)]
    if sensor14_offset is not None:
        cmd += ["--sensor14-offset", str(sensor14_offset)]
    if op3_compress:
        cmd += ["--op3-compress"]
        if op3_low is not None:
            cmd += ["--op3-compress-low", str(op3_low)]
        if op3_high is not None:
            cmd += ["--op3-compress-high", str(op3_high)]
    print('\n[GEN] Running generation: ' + ' '.join(cmd))
    p = subprocess.run(cmd, cwd=str(ROOT))
    if p.returncode != 0:
        raise RuntimeError('Generation failed')


def run_train_eval(set_id: str, version: str = 'v1') -> dict:
    cmd = ["python", str(TRAIN_SCRIPT),
           "--set-id", set_id, "--version", version, "--run", "auto",
           "--train-src", "gen", "--test", "syn", "--vib", "1",
           "--epochs", "10", "--seq-len", "20", "--batch-size", "256"]
    print('\n[TRAIN] Running training/eval: ' + ' '.join(cmd))
    p = subprocess.run(cmd, cwd=str(ROOT))
    if p.returncode != 0:
        raise RuntimeError('Train/Eval failed')
    # after run, read eval_summary.json from runs/<tag>
    # tag format: {set_id}_{version}_run{run_idx}
    # find the newest matching folder
    tags = [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith(set_id + '_' + version + '_')]
    if not tags:
        raise RuntimeError('No run dir found after training')
    tagdir = sorted(tags, key=lambda p: p.stat().st_mtime)[-1]
    summary_path = tagdir / 'eval_summary.json'
    if not summary_path.exists():
        raise RuntimeError('eval_summary.json not found in ' + str(tagdir))
    return json.loads(summary_path.read_text(encoding='utf-8'))


def main():
    # initial params (EXTREME-inspired)
    cov = 0.98
    s14_scale = 50.0
    s14_offset = 1350.0
    op3_apply = True
    op3_low = 0.60
    op3_high = 0.85

    best_mae = float('inf')
    best_params = None
    no_improve = 0
    history = []

    for it in range(1, max_iters+1):
        set_id = f'extreme_iter_{it}'
        t0 = time.time()
        run_generation(set_id, cov, s14_scale, s14_offset, op3_apply, op3_low, op3_high)
        summary = run_train_eval(set_id, version='v1')
        syn = summary.get('synthetic_metrics') or {}
        fd = summary.get('fd004_metrics') or {}
        mae = None
        # prefer synthetic MAE if present, else FD004
        if syn and 'MAE' in syn:
            mae = syn['MAE']
        elif fd and 'MAE' in fd:
            mae = fd['MAE']
        else:
            raise RuntimeError('No MAE found in summary')

        dt = time.time() - t0
        print(f"[ITER {it}] MAE={mae:.4f} (time {dt/60:.1f} min)")
        entry = {'iter': it, 'mae': mae, 'cov': cov, 's14_scale': s14_scale, 's14_offset': s14_offset,
                 'op3_apply': op3_apply, 'op3_low': op3_low, 'op3_high': op3_high, 'time_s': dt}
        history.append(entry)
        # update best
        if mae < best_mae - 1e-6:
            best_mae = mae; best_params = (cov, s14_scale, s14_offset, op3_low, op3_high)
            no_improve = 0
            # propose a small random move for next iter (exploit)
            cov = min(0.999, max(0.5, cov + random.uniform(-0.02, 0.02)))
            s14_scale = max(1.0, s14_scale * random.uniform(0.9, 1.1))
            s14_offset = s14_offset * random.uniform(0.95, 1.05)
            op3_low = min(op3_low+0.0, max(0.4, op3_low + random.uniform(-0.03, 0.03)))
            op3_high = min(0.95, max(op3_high + random.uniform(-0.03, 0.03), op3_low+0.01))
        else:
            no_improve += 1
            # undo move / reduce step (explore less)
            cov = min(0.999, max(0.5, cov + random.uniform(-0.01, 0.01)))
            s14_scale = max(1.0, s14_scale * random.uniform(0.98, 1.02))
            s14_offset = s14_offset * random.uniform(0.99, 1.01)
            op3_low = min(op3_low+0.0, max(0.4, op3_low + random.uniform(-0.01, 0.01)))
            op3_high = min(0.95, max(op3_high + random.uniform(-0.01, 0.01), op3_low+0.01))

        # save history
        out = {'history': history, 'best_mae': best_mae, 'best_params': best_params}
        out_log.write_text(json.dumps(out, indent=2), encoding='utf-8')

        if no_improve >= patience_no_improve:
            print('[STOP] No improvement for several iterations; stopping.')
            break

    print('Done. Best MAE=', best_mae, 'best_params=', best_params)

if __name__ == '__main__':
    main()
