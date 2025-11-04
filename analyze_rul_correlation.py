#!/usr/bin/env python3
"""Analyze RUL-feature correlation in training data"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

def read_data():
    # Read train data
    train_path = Path("data/training/pure_200/v7/train_1.txt")
    SENSOR_COLS = [f"sensor{i}" for i in range(1, 22)]
    COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + SENSOR_COLS
    train = pd.read_csv(train_path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    train.columns = COLS[:len(train.columns)]
    
    # Read RUL
    rul_path = Path("data/rul/pure_200/v7/RUL_1.txt")
    rul_vals = np.loadtxt(rul_path, dtype=int)
    
    # Add RUL to train (FIX: RUL file contains max_cycle, not RUL directly)
    max_cycles = {}
    for unit_id in train['unit'].unique():
        unit_data = train[train['unit'] == unit_id]
        max_cycle = unit_data['cycle'].max()
        max_cycles[unit_id] = max_cycle
    
    train['max_cycle'] = train['unit'].map(max_cycles)
    train['RUL'] = train['max_cycle'] - train['cycle']
    
    # Read vibration (v7 for latest)
    vib_path = Path("data/training/pure_200/v7/train_1_vib.csv")
    vib = pd.read_csv(vib_path)
    train = pd.concat([train, vib[['fe_RMS', 'fe_Kurtosis', 'fe_Crest', 'fe_Entropy']]], axis=1)
    
    return train

train = read_data()

print("\n" + "="*80)
print("RUL-FEATURE CORRELATION ANALYSIS")
print("="*80 + "\n")

features = ['op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)] + ['fe_RMS', 'fe_Kurtosis', 'fe_Crest', 'fe_Entropy']

correlations = []
for feat in features:
    if train[feat].std() > 0:
        corr, pval = pearsonr(train[feat], train['RUL'])
        correlations.append((feat, abs(corr), corr, pval))

correlations.sort(key=lambda x: x[1], reverse=True)

print("Top 10 HIGHEST RUL-correlated features:")
for i, (feat, abs_corr, corr, pval) in enumerate(correlations[:10], 1):
    print(f"{i:2}. {feat:<15} |r|={abs_corr:.4f}  r={corr:+.4f}  (p={pval:.2e})")

print("\nBottom 10 LOWEST RUL-correlated features:")
for i, (feat, abs_corr, corr, pval) in enumerate(correlations[-10:], 1):
    print(f"{i:2}. {feat:<15} |r|={abs_corr:.4f}  r={corr:+.4f}  (p={pval:.2e})")

print("\nVibration feature correlations:")
vib_feats = ['fe_RMS', 'fe_Kurtosis', 'fe_Crest', 'fe_Entropy']
for feat, abs_corr, corr, pval in correlations:
    if feat in vib_feats:
        print(f"  {feat:<15} |r|={abs_corr:.4f}  r={corr:+.4f}")

# Check monotonicity
print("\n" + "="*80)
print("MONOTONICITY CHECK (RUL bins)")
print("="*80 + "\n")

train['RUL_bin'] = pd.cut(train['RUL'], bins=[0, 50, 100, 150, 200, 300], labels=['0-50', '50-100', '100-150', '150-200', '200+'])

for feat in ['sensor11', 'sensor14', 'fe_RMS', 'fe_Kurtosis']:
    print(f"\n{feat}:")
    means = train.groupby('RUL_bin')[feat].mean()
    for bin_label, mean_val in means.items():
        print(f"  RUL {bin_label:>10}: {mean_val:>8.4f}")
    
    # Check if monotonic
    vals = means.values
    increasing = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
    decreasing = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
    if increasing:
        print(f"  → ✅ MONOTONIC INCREASING")
    elif decreasing:
        print(f"  → ✅ MONOTONIC DECREASING")
    else:
        print(f"  → ❌ NOT MONOTONIC (non-linear)")
