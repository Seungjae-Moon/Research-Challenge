#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_vibration.py — Vibration Feature Train/Test Distribution 진단
=====================================================================
Purpose: 센서처럼 vibration feature도 train/test 분포 차이 분석
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def read_vib_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def analyze_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    """Train/Test 분포 비교"""
    print(f"\n{'='*80}")
    print("VIBRATION FEATURE DISTRIBUTION ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}\n")
    
    print(f"{'Feature':<20} {'Train_μ':>12} {'Test_μ':>12} {'Δμ':>10} {'Train_σ':>10} {'Test_σ':>10} {'σ_ratio':>10}")
    print("-" * 80)
    
    problems = []
    
    for feat in feature_cols:
        train_mean = train_df[feat].mean()
        test_mean = test_df[feat].mean()
        train_std = train_df[feat].std()
        test_std = test_df[feat].std()
        
        delta_mu = abs(train_mean - test_mean)
        sigma_ratio = test_std / (train_std + 1e-9)
        
        print(f"{feat:<20} {train_mean:>12.6f} {test_mean:>12.6f} {delta_mu:>10.6f} {train_std:>10.6f} {test_std:>10.6f} {sigma_ratio:>10.3f}")
        
        # Problem detection
        if delta_mu > 0.1:  # Mean shift > 0.1
            problems.append(f"⚠ {feat}: Δμ={delta_mu:.4f} (mean shift)")
        if sigma_ratio < 0.7 or sigma_ratio > 1.4:  # Variance mismatch
            problems.append(f"⚠ {feat}: σ_ratio={sigma_ratio:.3f} (variance mismatch)")
    
    print("\n" + "="*80)
    
    if problems:
        print("\n🔴 PROBLEMS DETECTED:\n")
        for p in problems:
            print(p)
    else:
        print("\n✅ All vibration features well-aligned!")
    
    print("\n" + "="*80 + "\n")
    
    # Statistical summary
    print("\nDETAILED STATISTICS:")
    print("\nTrain:")
    print(train_df[feature_cols].describe())
    print("\nTest:")
    print(test_df[feature_cols].describe())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-id", type=str, default="pure_200")
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()
    
    train_vib = Path("data") / "training" / args.set_id / args.version / f"train_{args.run}_vib.csv"
    test_vib = Path("data") / "test" / args.set_id / args.version / f"test_{args.run}_vib.csv"
    
    if not train_vib.exists():
        print(f"❌ Train vib not found: {train_vib}")
        return
    if not test_vib.exists():
        print(f"❌ Test vib not found: {test_vib}")
        return
    
    train_df = read_vib_csv(train_vib)
    test_df = read_vib_csv(test_vib)
    
    vib_cols = [c for c in train_df.columns if c.startswith("fe_")]
    
    analyze_distribution(train_df, test_df, vib_cols)

if __name__ == "__main__":
    main()
