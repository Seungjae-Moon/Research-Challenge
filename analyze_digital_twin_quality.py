#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_digital_twin_quality.py — 디지털 트윈 데이터 품질 심층 분석
========================================================================
MAE가 높은 근본 원인이 디지털 트윈 생성 로직에 있는지 진단
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import json

def read_cmapss_like(path: Path) -> pd.DataFrame:
    SENSOR_COLS = [f"sensor{i}" for i in range(1, 22)]
    COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + SENSOR_COLS
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    df.columns = COLS[:len(df.columns)]
    return df

def read_vib_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def add_rul(df: pd.DataFrame, rul_path: Path) -> pd.DataFrame:
    rul_vals = np.loadtxt(rul_path, dtype=int)
    rul_dict = {unit: rul_vals[unit-1] for unit in range(1, len(rul_vals)+1)}
    df['max_cycle'] = df['unit'].map(rul_dict)
    df['RUL'] = df['max_cycle'] - df['cycle']
    return df.drop(columns=['max_cycle'])

def analyze_monotonicity(df: pd.DataFrame, feature: str) -> dict:
    """각 unit별 feature의 monotonicity 측정"""
    results = []
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit].sort_values('cycle')
        if len(unit_df) < 5:
            continue
        
        # Spearman correlation (monotonic relationship)
        from scipy.stats import spearmanr
        corr, _ = spearmanr(unit_df['cycle'], unit_df[feature])
        results.append(corr)
    
    return {
        'mean_monotonicity': np.mean(results),
        'std_monotonicity': np.std(results),
        'min_monotonicity': np.min(results),
        'max_monotonicity': np.max(results),
        'pct_non_monotonic': np.mean(np.abs(results) < 0.3) * 100  # |rho| < 0.3은 약한 monotonic
    }

def analyze_noise_level(df: pd.DataFrame, feature: str) -> dict:
    """Feature의 noise level 분석"""
    noise_ratios = []
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit].sort_values('cycle')
        if len(unit_df) < 10:
            continue
        
        vals = unit_df[feature].values
        # Signal = smoothed trend (5-point moving average)
        from scipy.ndimage import uniform_filter1d
        if len(vals) >= 5:
            signal = uniform_filter1d(vals, size=5, mode='nearest')
            noise = vals - signal
            noise_ratio = np.std(noise) / (np.std(vals) + 1e-9)
            noise_ratios.append(noise_ratio)
    
    return {
        'mean_noise_ratio': np.mean(noise_ratios),
        'std_noise_ratio': np.std(noise_ratios)
    }

def analyze_degradation_pattern(df: pd.DataFrame, feature: str) -> dict:
    """RUL에 따른 degradation pattern 분석"""
    # RUL을 구간별로 나눔
    bins = [0, 50, 100, 150, 300]
    labels = ['0-50', '50-100', '100-150', '150+']
    df['rul_bin'] = pd.cut(df['RUL'], bins=bins, labels=labels, include_lowest=True)
    
    pattern = {}
    for bin_label in labels:
        bin_data = df[df['rul_bin'] == bin_label]
        if len(bin_data) > 0:
            pattern[bin_label] = {
                'mean': bin_data[feature].mean(),
                'std': bin_data[feature].std(),
                'count': len(bin_data)
            }
    
    # Check if there's clear trend across RUL bins
    means = [pattern[label]['mean'] for label in labels if label in pattern]
    if len(means) >= 3:
        # Linear fit to check trend
        x = np.arange(len(means))
        slope = np.polyfit(x, means, 1)[0]
        pattern['trend_slope'] = slope
        pattern['trend_direction'] = 'increasing' if slope > 0 else 'decreasing'
    else:
        pattern['trend_slope'] = 0
        pattern['trend_direction'] = 'unclear'
    
    return pattern

def main():
    print("\n" + "="*80)
    print("🔬 DIGITAL TWIN QUALITY ANALYSIS — Root Cause of High MAE")
    print("="*80 + "\n")
    
    # Load data
    set_id = "pure_200"
    version = "v6"
    run = 1
    
    train_path = Path(f"data/training/{set_id}/{version}/train_{run}.txt")
    test_path = Path(f"data/test/{set_id}/{version}/test_{run}.txt")
    rul_train_path = Path(f"data/rul/{set_id}/{version}/RUL_{run}.txt")
    vib_train_path = Path(f"data/training/{set_id}/{version}/train_{run}_vib.csv")
    vib_test_path = Path(f"data/test/{set_id}/{version}/test_{run}_vib.csv")
    
    print("Loading data...")
    train_df = read_cmapss_like(train_path)
    test_df = read_cmapss_like(test_path)
    train_df = add_rul(train_df, rul_train_path)
    
    # For test, use last cycle as RUL (approximation)
    test_rul_dict = {}
    for unit in test_df['unit'].unique():
        unit_test = test_df[test_df['unit'] == unit]
        last_cycle = unit_test['cycle'].max()
        test_rul_dict[unit] = last_cycle
    test_df['RUL'] = test_df.groupby('unit')['cycle'].transform('max') - test_df['cycle']
    
    # Merge vibration
    if vib_train_path.exists():
        vib_train = read_vib_csv(vib_train_path)
        vib_only = [c for c in vib_train.columns if c not in train_df.columns]
        train_df = pd.concat([train_df, vib_train[vib_only]], axis=1)
    
    if vib_test_path.exists():
        vib_test = read_vib_csv(vib_test_path)
        vib_only = [c for c in vib_test.columns if c not in test_df.columns]
        test_df = pd.concat([test_df, vib_test[vib_only]], axis=1)
    
    print(f"Train: {len(train_df)} samples, {train_df['unit'].nunique()} units")
    print(f"Test:  {len(test_df)} samples, {test_df['unit'].nunique()} units")
    print()
    
    # Analyze features
    sensor_cols = [f"sensor{i}" for i in range(1, 22)]
    vib_cols = [c for c in train_df.columns if c.startswith('fe_')]
    op_cols = ['op1', 'op2', 'op3']
    all_features = op_cols + sensor_cols + vib_cols
    
    print("="*80)
    print("ANALYSIS 1: RUL CORRELATION (피처가 RUL을 예측할 수 있는가?)")
    print("="*80 + "\n")
    
    correlations = {}
    for feat in all_features:
        if feat in train_df.columns:
            valid_data = train_df[[feat, 'RUL']].dropna()
            if len(valid_data) > 10 and valid_data[feat].std() > 1e-9:
                corr, pval = pearsonr(valid_data[feat], valid_data['RUL'])
                correlations[feat] = {'corr': corr, 'abs_corr': abs(corr), 'pval': pval}
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1]['abs_corr'], reverse=True)
    
    print("Top 15 features by |correlation| with RUL:")
    print(f"{'Feature':<15} {'Correlation':>12} {'|Corr|':>10} {'p-value':>12}")
    print("-" * 60)
    for feat, data in sorted_corr[:15]:
        print(f"{feat:<15} {data['corr']:>12.4f} {data['abs_corr']:>10.4f} {data['pval']:>12.6f}")
    
    # Check if ANY feature has strong correlation
    strong_corr = [f for f, d in correlations.items() if d['abs_corr'] > 0.7]
    moderate_corr = [f for f, d in correlations.items() if 0.5 <= d['abs_corr'] <= 0.7]
    weak_corr = [f for f, d in correlations.items() if 0.3 <= d['abs_corr'] < 0.5]
    very_weak = [f for f, d in correlations.items() if d['abs_corr'] < 0.3]
    
    print(f"\n📊 Correlation Summary:")
    print(f"  Strong (|r| > 0.7):    {len(strong_corr)} features")
    print(f"  Moderate (0.5-0.7):    {len(moderate_corr)} features")
    print(f"  Weak (0.3-0.5):        {len(weak_corr)} features")
    print(f"  Very weak (< 0.3):     {len(very_weak)} features")
    
    if len(strong_corr) == 0:
        print(f"\n🚨 CRITICAL PROBLEM: NO STRONG RUL CORRELATION!")
        print(f"   → 어떤 피처도 RUL과 강한 상관관계(|r|>0.7)가 없음")
        print(f"   → 모델이 RUL을 정확히 예측하기 불가능")
    
    print("\n" + "="*80)
    print("ANALYSIS 2: MONOTONICITY (피처가 시간에 따라 일관되게 변하는가?)")
    print("="*80 + "\n")
    
    print("Checking monotonicity for top correlated features...")
    print(f"{'Feature':<15} {'Mean ρ':>10} {'Std ρ':>10} {'% Non-Mono':>12}")
    print("-" * 60)
    
    for feat, _ in sorted_corr[:10]:
        mono_result = analyze_monotonicity(train_df, feat)
        print(f"{feat:<15} {mono_result['mean_monotonicity']:>10.3f} {mono_result['std_monotonicity']:>10.3f} {mono_result['pct_non_monotonic']:>11.1f}%")
    
    print("\n" + "="*80)
    print("ANALYSIS 3: NOISE LEVEL (신호 대비 노이즈 비율)")
    print("="*80 + "\n")
    
    print(f"{'Feature':<15} {'Noise Ratio':>12} {'Assessment':>15}")
    print("-" * 60)
    
    for feat, _ in sorted_corr[:10]:
        noise_result = analyze_noise_level(train_df, feat)
        assessment = "Low" if noise_result['mean_noise_ratio'] < 0.3 else "High" if noise_result['mean_noise_ratio'] > 0.6 else "Moderate"
        print(f"{feat:<15} {noise_result['mean_noise_ratio']:>12.3f} {assessment:>15}")
    
    print("\n" + "="*80)
    print("ANALYSIS 4: DEGRADATION PATTERN (RUL 구간별 피처 변화)")
    print("="*80 + "\n")
    
    print("Checking if features show clear degradation trend...")
    print(f"{'Feature':<15} {'Trend':>10} {'Slope':>10}")
    print("-" * 60)
    
    for feat, _ in sorted_corr[:10]:
        deg_pattern = analyze_degradation_pattern(train_df, feat)
        print(f"{feat:<15} {deg_pattern['trend_direction']:>10} {deg_pattern['trend_slope']:>10.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS 5: TRAIN vs TEST DISTRIBUTION (데이터 일관성)")
    print("="*80 + "\n")
    
    print(f"{'Feature':<15} {'Train μ':>12} {'Test μ':>12} {'Δμ':>10} {'Status':>10}")
    print("-" * 70)
    
    distribution_problems = []
    for feat, _ in sorted_corr[:15]:
        if feat in test_df.columns:
            train_mean = train_df[feat].mean()
            test_mean = test_df[feat].mean()
            delta = abs(train_mean - test_mean)
            status = "✓" if delta < 0.1 else "⚠" if delta < 0.2 else "✗"
            if delta >= 0.1:
                distribution_problems.append((feat, delta))
            print(f"{feat:<15} {train_mean:>12.4f} {test_mean:>12.4f} {delta:>10.4f} {status:>10}")
    
    print("\n" + "="*80)
    print("🎯 ROOT CAUSE DIAGNOSIS")
    print("="*80 + "\n")
    
    # Summarize findings
    issues = []
    
    if len(strong_corr) == 0:
        issues.append({
            'severity': 'CRITICAL',
            'category': 'Weak RUL Correlation',
            'description': f'NO feature has |r| > 0.7 with RUL. Best is {sorted_corr[0][0]} with |r|={sorted_corr[0][1]["abs_corr"]:.3f}',
            'impact': 'Model cannot learn accurate RUL prediction',
            'solution': 'Increase degradation effects in digital twin physics'
        })
    
    if len(distribution_problems) > 0:
        issues.append({
            'severity': 'HIGH',
            'category': 'Train/Test Distribution Mismatch',
            'description': f'{len(distribution_problems)} features have Δμ > 0.1',
            'impact': 'Model trained on different distribution than test',
            'solution': 'Fix calibration or health calculation in digital twin'
        })
    
    # Check monotonicity issues
    non_mono_count = sum(1 for feat, _ in sorted_corr[:10] 
                         if analyze_monotonicity(train_df, feat)['pct_non_monotonic'] > 30)
    if non_mono_count > 3:
        issues.append({
            'severity': 'MEDIUM',
            'category': 'Non-Monotonic Features',
            'description': f'{non_mono_count}/10 top features are not consistently monotonic',
            'impact': 'Sequence models struggle with non-monotonic signals',
            'solution': 'Add smoothing or increase health decay rate'
        })
    
    # Print issues
    for i, issue in enumerate(issues, 1):
        print(f"{i}. [{issue['severity']}] {issue['category']}")
        print(f"   Description: {issue['description']}")
        print(f"   Impact: {issue['impact']}")
        print(f"   Solution: {issue['solution']}")
        print()
    
    if len(issues) == 0:
        print("✅ No critical issues found in digital twin data quality")
        print("   → Problem likely in model architecture or training procedure")
    else:
        print(f"⚠️  Found {len(issues)} issues in digital twin")
        print(f"   → MAE 40+ is primarily due to these data quality problems")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
