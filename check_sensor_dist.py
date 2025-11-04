#!/usr/bin/env python3
"""Check sensor distribution for pure_200 v6"""
import numpy as np
import pandas as pd
from pathlib import Path

def read_cmapss(path):
    SENSOR_COLS = [f"sensor{i}" for i in range(1, 22)]
    COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + SENSOR_COLS
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    df.columns = COLS[:len(df.columns)]
    return df

train = read_cmapss(Path("data/training/pure_200/v6/train_1.txt"))
test = read_cmapss(Path("data/test/pure_200/v6/test_1.txt"))

print("\n" + "="*80)
print("SENSOR DISTRIBUTION CHECK: pure_200 v6")
print("="*80 + "\n")

features = ['op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]
problems = []

for feat in features:
    train_mean = train[feat].mean()
    test_mean = test[feat].mean()
    train_std = train[feat].std()
    test_std = test[feat].std()
    
    delta_mu = abs(train_mean - test_mean)
    sigma_ratio = test_std / (train_std + 1e-9)
    
    if delta_mu > 0.1 or sigma_ratio < 0.7 or sigma_ratio > 1.4:
        problems.append(f"⚠ {feat}: Δμ={delta_mu:.4f}, σ_ratio={sigma_ratio:.3f}")

if problems:
    print("🔴 SENSOR DISTRIBUTION PROBLEMS:\n")
    for p in problems:
        print(p)
else:
    print("✅ All sensors well-aligned!")

print("\nTop-5 |Δμ| sensors:")
deltas = []
for feat in features:
    delta = abs(train[feat].mean() - test[feat].mean())
    deltas.append((feat, delta))
deltas.sort(key=lambda x: x[1], reverse=True)
for feat, delta in deltas[:5]:
    print(f"  {feat}: Δμ={delta:.4f}")
