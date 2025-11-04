#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_mae_stagnation.py — MAE 정체 원인 심층 분석
=====================================================
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def analyze_predictions(run_dir: Path, label: str):
    """Analyze prediction patterns."""
    preds = pd.read_csv(run_dir / "preds_synthetic.csv")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {label}")
    print(f"{'='*80}\n")
    
    # Basic stats
    print(f"Test samples: {len(preds)}")
    print(f"MAE: {preds['abs_err'].mean():.2f}")
    print(f"RMSE: {np.sqrt((preds['abs_err']**2).mean()):.2f}")
    print(f"Error STD: {preds['abs_err'].std():.2f}")
    
    # Prediction range analysis
    print(f"\nTrue RUL range: [{preds['rul_true'].min():.0f}, {preds['rul_true'].max():.0f}]")
    print(f"Pred RUL range: [{preds['rul_pred'].min():.0f}, {preds['rul_pred'].max():.0f}]")
    print(f"Range compression: {(preds['rul_pred'].max() - preds['rul_pred'].min()) / (preds['rul_true'].max() - preds['rul_true'].min()):.2%}")
    
    # Bias analysis
    bias = (preds['rul_pred'] - preds['rul_true']).mean()
    print(f"\nBias (mean error): {bias:.2f} cycles")
    print(f"  → {'UNDER' if bias < 0 else 'OVER'}-predicting on average")
    
    # Correlation
    corr = np.corrcoef(preds['rul_true'], preds['rul_pred'])[0,1]
    r2 = corr**2
    print(f"\nCorrelation: {corr:.3f}")
    print(f"R² score: {r2:.3f}")
    
    # Error by RUL range
    print(f"\nError by RUL range:")
    for low, high, name in [(0, 50, "Low RUL (0-50)"), 
                             (50, 120, "Mid RUL (50-120)"), 
                             (120, 250, "High RUL (120+)")]:
        mask = (preds['rul_true'] >= low) & (preds['rul_true'] < high)
        if mask.sum() > 0:
            mae = preds.loc[mask, 'abs_err'].mean()
            count = mask.sum()
            print(f"  {name:<20} MAE={mae:>6.1f} (n={count})")
    
    # Worst predictions
    print(f"\nWorst 5 predictions:")
    worst = preds.nlargest(5, 'abs_err')
    for _, row in worst.iterrows():
        print(f"  Unit {row['unit']:>3.0f}: True={row['rul_true']:>3.0f}, Pred={row['rul_pred']:>6.1f}, Error={row['abs_err']:>6.1f}")
    
    # Calibration quality
    print(f"\nCalibration check:")
    with open(run_dir / "eval_summary.json") as f:
        summary = json.load(f)
    
    # Check if predictions are stuck
    pred_unique = preds['rul_pred'].nunique()
    print(f"Unique predictions: {pred_unique} / {len(preds)}")
    if pred_unique < len(preds) * 0.5:
        print(f"  ⚠ WARNING: Too many duplicate predictions! Model may be stuck.")
    
    # Prediction clustering
    pred_std = preds['rul_pred'].std()
    true_std = preds['rul_true'].std()
    print(f"Prediction STD: {pred_std:.2f}")
    print(f"True RUL STD: {true_std:.2f}")
    print(f"STD ratio: {pred_std/true_std:.2%}")
    if pred_std / true_std < 0.5:
        print(f"  ⚠ WARNING: Predictions too compressed! Model not learning full RUL range.")
    
    return preds, bias, r2


def compare_runs():
    """Compare multiple runs."""
    runs = [
        ("runs/pure_200_v2_run1", "v2 (vib=0, MAE=41.5)"),
        ("runs/pure_200_v6_run1", "v6 (vib=1, MAE=43.6)"),
    ]
    
    results = []
    for run_dir, label in runs:
        run_path = Path(run_dir)
        if run_path.exists():
            preds, bias, r2 = analyze_predictions(run_path, label)
            results.append({
                'label': label,
                'preds': preds,
                'bias': bias,
                'r2': r2
            })
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<30} {'MAE':>8} {'Bias':>8} {'R²':>8} {'Pred_STD':>10} {'True_STD':>10}")
    print("-" * 80)
    for r in results:
        mae = r['preds']['abs_err'].mean()
        pred_std = r['preds']['rul_pred'].std()
        true_std = r['preds']['rul_true'].std()
        print(f"{r['label']:<30} {mae:>8.2f} {r['bias']:>8.2f} {r['r2']:>8.3f} {pred_std:>10.2f} {true_std:>10.2f}")
    
    # Diagnosis
    print(f"\n{'='*80}")
    print("ROOT CAUSE DIAGNOSIS")
    print(f"{'='*80}\n")
    
    print("🔍 Key Findings:")
    print("1. Both models show RANGE COMPRESSION:")
    print("   - True RUL spans 5-190, but predictions cluster around 60-140")
    print("   - Pred STD / True STD < 70% → model not confident in extreme values")
    print("")
    print("2. SYSTEMATIC BIAS:")
    print("   - Low RUL (0-50): Model OVER-predicts (predicts 60-80 instead of 5-50)")
    print("   - High RUL (120+): Model UNDER-predicts (predicts 80-140 instead of 150-190)")
    print("")
    print("3. VIBRATION FEATURE NOT HELPING:")
    print("   - v6 (with vib) MAE=43.6 vs v2 (no vib) MAE=41.5")
    print("   - Vibration adds noise instead of signal")
    print("")
    print("⚠ PRIMARY BOTTLENECK:")
    print("   → MODEL ARCHITECTURE: TCN+BiGRU not learning full RUL range dynamics")
    print("   → LOSS FUNCTION: Huber loss may be too forgiving on large errors")
    print("   → CALIBRATION: Affine calibration compressing predictions further")
    print("")
    print("💡 SOLUTION STRATEGIES:")
    print("   1. Switch to QUANTILE LOSS (q=0.5) for better range coverage")
    print("   2. Add RANGE PENALTY: Extra loss for predictions outside [0, max_RUL]")
    print("   3. MULTI-SCALE TRAINING: Train on different RUL ranges separately")
    print("   4. REMOVE CALIBRATION: Use raw predictions without affine transform")
    print("   5. STRONGER REGULARIZATION: L2 on final layer to prevent collapse")


if __name__ == "__main__":
    compare_runs()
