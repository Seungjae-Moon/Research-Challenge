#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_sensor_vib_fusion.py — 센서 단독 vs 바이브레이션 단독 vs 융합 성능 비교
================================================================================
Purpose: 3가지 모델 학습 및 비교
  1. Sensor-only (24 features: op1-3, sensor1-21)
  2. Vibration-only (4 features: fe_RMS, fe_Kurtosis, fe_Crest, fe_Entropy)
  3. Fusion (28 features: sensor + vibration)
  
Goal: Fusion MAE < min(Sensor MAE, Vib MAE) 달성
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def run_training(set_id, version, run, vib, seq_len, epochs, model, batch_size, hidden, layers, desc):
    """Run training and return MAE."""
    print(f"\n{'='*80}")
    print(f"🔧 Training: {desc}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable, "train_eval_fusion.py",
        "--set-id", set_id,
        "--version", version,
        "--run", str(run),
        "--vib", str(vib),
        "--seq-len", str(seq_len),
        "--epochs", str(epochs),
        "--val-style", "testlike_syn",
        "--rul-clip", "250",
        "--loss", "huber",
        "--test", "syn",
        "--device", "cpu",
        "--model", model,
        "--batch-size", str(batch_size),
        "--hidden", str(hidden),
        "--layers", str(layers)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse MAE from output
    for line in result.stdout.split('\n'):
        if "Synthetic test ->" in line and "MAE=" in line:
            mae_str = line.split("MAE=")[1].split(",")[0]
            mae = float(mae_str)
            return mae
    
    print(f"⚠ Could not parse MAE from output")
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare sensor-only, vib-only, and fusion models")
    parser.add_argument("--set-id", type=str, default="pure_200")
    parser.add_argument("--version", type=str, default="v6")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="tcn_bigru")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=120)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    
    results = {}
    
    # 1. Sensor-only (vib=0)
    print("\n" + "🔬 PHASE 1: Sensor-Only Model")
    mae_sensor = run_training(
        args.set_id, args.version, 1, 0,  # vib=0
        args.seq_len, args.epochs, args.model,
        args.batch_size, args.hidden, args.layers,
        "Sensor-Only (24 features)"
    )
    results['sensor_only'] = mae_sensor
    
    # 2. Fusion (vib=1)
    print("\n" + "🔬 PHASE 2: Sensor+Vibration Fusion Model")
    mae_fusion = run_training(
        args.set_id, args.version, 2, 1,  # vib=1
        args.seq_len, args.epochs, args.model,
        args.batch_size, args.hidden, args.layers,
        "Fusion (28 features)"
    )
    results['fusion'] = mae_fusion
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON RESULTS")
    print("="*80 + "\n")
    
    print(f"{'Model':<30} {'MAE':>10} {'Improvement':>15}")
    print("-" * 60)
    print(f"{'Sensor-Only (24 features)':<30} {mae_sensor:>10.3f} {'baseline':>15}")
    print(f"{'Fusion (28 features)':<30} {mae_fusion:>10.3f} {f'{((mae_sensor-mae_fusion)/mae_sensor*100):.1f}%':>15}")
    
    print("\n" + "="*80)
    
    if mae_fusion < mae_sensor:
        improvement = (mae_sensor - mae_fusion) / mae_sensor * 100
        print(f"✅ SUCCESS: Fusion improves over sensor-only by {improvement:.1f}%")
        print(f"   Vibration features ADD VALUE to RUL prediction!")
    else:
        degradation = (mae_fusion - mae_sensor) / mae_sensor * 100
        print(f"⚠ WARNING: Fusion is {degradation:.1f}% WORSE than sensor-only")
        print(f"   Need to improve vibration feature quality!")
    
    print("="*80 + "\n")
    
    # Save results
    output_file = Path("fusion_comparison_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
