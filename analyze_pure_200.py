#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_pure_200.py — pure_200 데이터셋 종합 분석 및 시각화
=============================================================
생성되는 시각화:
1. correlation_heatmap.png — 전체 feature correlation heatmap
2. rul_vs_features.png — RUL에 따른 주요 센서 변화 추이
3. vibration_sensor_correlation.png — Vibration-Sensor 상관관계 Top 10
4. feature_importance_by_rul.png — RUL 구간별 feature 중요도
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


def read_cmapss_like(path: Path) -> pd.DataFrame:
    """Read CMAPSS-style data file."""
    SENSOR_COLS = [f"sensor{i}" for i in range(1, 22)]
    COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + SENSOR_COLS
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    df.columns = COLS[:len(df.columns)]
    return df


def read_vib_csv(path: Path) -> pd.DataFrame:
    """Read vibration feature CSV."""
    return pd.read_csv(path)


def add_rul_to_df(df: pd.DataFrame, rul_dict: dict) -> pd.DataFrame:
    """Add RUL column based on unit max cycles."""
    df = df.copy()
    df['max_cycle'] = df['unit'].map(rul_dict)
    df['RUL'] = df['max_cycle'] - df['cycle']
    return df.drop(columns=['max_cycle'])


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path, title: str = "Feature Correlation Heatmap"):
    """Plot correlation heatmap for all features."""
    # Select numeric columns (exclude unit, cycle)
    feature_cols = [c for c in df.columns if c not in ['unit', 'cycle', 'max_cycle']]
    corr_matrix = df[feature_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Heatmap
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax, annot=False)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def plot_rul_vs_features(df: pd.DataFrame, output_path: Path, top_n: int = 8):
    """Plot RUL vs top correlated features."""
    # Calculate correlation with RUL
    feature_cols = [c for c in df.columns if c not in ['unit', 'cycle', 'RUL', 'max_cycle']]
    correlations = {}
    for col in feature_cols:
        if df[col].std() > 0:
            correlations[col] = abs(pearsonr(df[col], df['RUL'])[0])
    
    # Select top N features
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (feat, corr) in enumerate(top_features):
        ax = axes[idx]
        
        # Sample data for plotting (to avoid overplotting)
        sample_df = df.sample(min(5000, len(df)), random_state=42)
        
        # Scatter plot
        ax.scatter(sample_df['RUL'], sample_df[feat], alpha=0.3, s=1)
        
        # Trend line (with error handling)
        try:
            z = np.polyfit(sample_df['RUL'], sample_df[feat], 1)
            p = np.poly1d(z)
            rul_range = np.linspace(sample_df['RUL'].min(), sample_df['RUL'].max(), 100)
            ax.plot(rul_range, p(rul_range), 'r--', linewidth=2, alpha=0.7)
        except np.linalg.LinAlgError:
            pass  # Skip trend line if SVD doesn't converge
        
        ax.set_xlabel('RUL', fontsize=10)
        ax.set_ylabel(feat, fontsize=10)
        ax.set_title(f'{feat} (|r|={corr:.3f})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('RUL vs Top Correlated Features', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def plot_vibration_sensor_correlation(df: pd.DataFrame, output_path: Path, top_n: int = 10):
    """Plot top vibration-sensor correlations."""
    vib_cols = [c for c in df.columns if c.startswith('fe_')]
    sensor_cols = [c for c in df.columns if c.startswith('sensor')]
    
    if not vib_cols:
        print("⚠ No vibration features found, skipping vibration-sensor correlation plot")
        return
    
    # Calculate cross-correlations
    cross_corr = []
    for vib in vib_cols:
        for sensor in sensor_cols:
            corr = pearsonr(df[vib], df[sensor])[0]
            cross_corr.append((vib, sensor, abs(corr), corr))
    
    # Sort by absolute correlation
    cross_corr.sort(key=lambda x: x[2], reverse=True)
    top_pairs = cross_corr[:top_n]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = [f"{vib} ← {sensor}" for vib, sensor, _, _ in top_pairs]
    values = [corr for _, _, _, corr in top_pairs]
    colors = ['steelblue' if v > 0 else 'coral' for v in values]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Pearson Correlation', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Vibration-Sensor Correlations', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def plot_feature_importance_by_rul_stage(df: pd.DataFrame, output_path: Path):
    """Plot feature importance at different RUL stages."""
    # Define RUL stages
    stages = [
        ("초기 단계 (RUL > 150)", df[df['RUL'] > 150]),
        ("중간 단계 (80 < RUL ≤ 150)", df[(df['RUL'] > 80) & (df['RUL'] <= 150)]),
        ("말기 단계 (RUL ≤ 80)", df[df['RUL'] <= 80])
    ]
    
    feature_cols = [c for c in df.columns if c not in ['unit', 'cycle', 'RUL', 'max_cycle']]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (stage_name, stage_df) in enumerate(stages):
        if len(stage_df) == 0:
            continue
        
        ax = axes[idx]
        
        # Calculate std for each feature (variability as proxy for importance)
        feature_std = {col: stage_df[col].std() for col in feature_cols}
        top_features = sorted(feature_std.items(), key=lambda x: x[1], reverse=True)[:10]
        
        labels = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color='teal', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Standard Deviation', fontsize=10)
        ax.set_title(stage_name, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    fig.suptitle('Feature Variability by RUL Stage', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pure_200 dataset and generate visualizations")
    parser.add_argument("--set-id", type=str, default="pure_200", help="Dataset set ID")
    parser.add_argument("--version", type=str, default="v3", help="Dataset version")
    parser.add_argument("--run", type=int, default=1, help="Run index")
    parser.add_argument("--output-dir", type=str, default="poster_figures", help="Output directory")
    args = parser.parse_args()
    
    # Paths
    train_path = Path("data") / "training" / args.set_id / args.version / f"train_{args.run}.txt"
    vib_path = Path("data") / "training" / args.set_id / args.version / f"train_{args.run}_vib.csv"
    rul_path = Path("data") / "rul" / args.set_id / args.version / f"RUL_{args.run}.txt"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing Dataset: {args.set_id}/{args.version}/run{args.run}")
    print(f"{'='*60}\n")
    
    # Read data
    print("Reading data...")
    df = read_cmapss_like(train_path)
    print(f"  Train data: {len(df)} rows, {len(df.columns)} columns")
    
    # Read RUL
    rul_vals = np.loadtxt(rul_path, dtype=int)
    rul_dict = {unit: rul_vals[unit-1] for unit in range(1, len(rul_vals)+1)}
    df = add_rul_to_df(df, rul_dict)
    
    # Read vibration features
    if vib_path.exists():
        vib_df = read_vib_csv(vib_path)
        print(f"  Vibration features: {len(vib_df)} rows, {len(vib_df.columns)} columns")
        # Drop duplicate columns from vib_df
        vib_only_cols = [c for c in vib_df.columns if c not in df.columns]
        df = pd.concat([df, vib_df[vib_only_cols]], axis=1)
    else:
        print(f"  ⚠ Vibration file not found: {vib_path}")
    
    print(f"\nCombined data shape: {df.shape}")
    print(f"Features: {list(df.columns)}\n")
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    # 1. Correlation heatmap
    plot_correlation_heatmap(
        df, 
        output_dir / f"{args.set_id}_{args.version}_correlation_heatmap.png",
        title=f"Feature Correlation Heatmap — {args.set_id}/{args.version}"
    )
    
    # 2. RUL vs features
    plot_rul_vs_features(
        df,
        output_dir / f"{args.set_id}_{args.version}_rul_vs_features.png",
        top_n=8
    )
    
    # 3. Vibration-Sensor correlation
    plot_vibration_sensor_correlation(
        df,
        output_dir / f"{args.set_id}_{args.version}_vibration_sensor_corr.png",
        top_n=10
    )
    
    # 4. Feature importance by RUL stage
    plot_feature_importance_by_rul_stage(
        df,
        output_dir / f"{args.set_id}_{args.version}_feature_importance_by_stage.png"
    )
    
    print(f"\n{'='*60}")
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
