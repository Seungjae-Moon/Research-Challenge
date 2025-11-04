#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_poster_figures.py — 연구 포스터용 시각화 생성
================================================================
Usage:
    python generate_poster_figures.py --run-dir runs/200_v1_run1
    
생성되는 그림:
1. scatter_pred_vs_true.png — Predicted vs True RUL (산점도 + ±10% 오차 범위)
2. mae_comparison.png — 실험별 MAE 비교 막대 그래프
3. error_distribution.png — 예측 오차 분포 히스토그램
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic, Linux: Nanum)
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def load_run_data(run_dir: Path):
    """Load predictions CSV and eval summary JSON from run directory."""
    preds_syn = run_dir / "preds_synthetic.csv"
    eval_summary = run_dir / "eval_summary.json"
    
    if not preds_syn.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_syn}")
    if not eval_summary.exists():
        raise FileNotFoundError(f"Eval summary not found: {eval_summary}")
    
    df = pd.read_csv(preds_syn)
    with open(eval_summary, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    return df, summary


def figure1_scatter_plot(df: pd.DataFrame, mae: float, use_vib: bool, output_path: Path):
    """
    Figure 1: Predicted vs True RUL scatter plot
    - X-axis: True RUL
    - Y-axis: Predicted RUL
    - Diagonal line (y=x) with ±10% error bands
    - Color: blue if vib=1, orange if vib=0
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    true_rul = df['rul_true'].values
    pred_rul = df['rul_pred'].values
    
    # Color by vib usage
    color = 'tab:blue' if use_vib else 'tab:orange'
    label = 'With Vibration Features' if use_vib else 'Without Vibration Features'
    
    # Scatter plot
    ax.scatter(true_rul, pred_rul, alpha=0.6, s=50, c=color, edgecolors='k', linewidth=0.5, label=label)
    
    # Diagonal line (perfect prediction)
    min_val = min(true_rul.min(), pred_rul.min())
    max_val = max(true_rul.max(), pred_rul.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction (y=x)')
    
    # ±10% error bands
    ax.fill_between([min_val, max_val], 
                     [min_val * 0.9, max_val * 0.9], 
                     [min_val * 1.1, max_val * 1.1], 
                     alpha=0.15, color='gray', label='±10% Error Band')
    
    ax.set_xlabel('True RUL (cycles)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted RUL (cycles)', fontsize=14, fontweight='bold')
    ax.set_title(f'Predicted vs True RUL (MAE={mae:.1f} cycles)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def figure2_mae_comparison(output_path: Path):
    """
    Figure 2: MAE comparison bar chart
    - X-axis: ['FD004 (baseline)', 'Synthetic (no vib)', 'Synthetic (vib)']
    - Y-axis: MAE values
    - Show MAE numbers on top of each bar
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data (example values - replace with actual if available)
    labels = ['FD004\n(baseline)', 'Synthetic\n(no vib)', 'Synthetic\n(with vib)']
    mae_values = [18.5, 17.1, 14.2]  # Replace with actual values from experiments
    colors = ['#d62728', '#ff7f0e', '#1f77b4']
    
    bars = ax.bar(labels, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add MAE numbers on top of bars
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax.set_title('Effect of Vibration Feature on Model Accuracy', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(mae_values) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def figure3_error_distribution(df: pd.DataFrame, mae: float, output_path: Path):
    """
    Figure 3: Prediction error distribution histogram
    - X-axis: Error (Predicted - True)
    - Y-axis: Frequency
    - Show mean MAE and std in title
    - Vertical line at x=0 as reference
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    errors = df['rul_pred'].values - df['rul_true'].values
    std_err = np.std(errors)
    mean_err = np.mean(errors)
    
    # Histogram
    n, bins, patches = ax.hist(errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Vertical line at x=0
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error (x=0)')
    
    # Mean error line
    ax.axvline(x=mean_err, color='darkgreen', linestyle='-.', linewidth=2, label=f'Mean Error = {mean_err:.2f}')
    
    ax.set_xlabel('Prediction Error (Predicted - True RUL)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title(f'Prediction Error Distribution (MAE={mae:.1f}, Std={std_err:.1f})', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate poster figures from trained model run directory')
    parser.add_argument('--run-dir', type=str, default='runs/200_v1_run1',
                        help='Path to run directory containing preds_synthetic.csv and eval_summary.json')
    parser.add_argument('--output-dir', type=str, default='poster_figures',
                        help='Directory to save output figures')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📊 Generating Poster Figures")
    print(f"{'='*70}")
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Load data
    try:
        df, summary = load_run_data(run_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Extract metrics
    syn_metrics = summary.get('synthetic_metrics', {})
    mae = syn_metrics.get('MAE', 0.0)
    use_vib = summary.get('use_vib', True)
    
    print(f"✓ Loaded {len(df)} test predictions")
    print(f"✓ Synthetic MAE: {mae:.3f}")
    print(f"✓ Vibration features used: {use_vib}\n")
    
    # Generate figures
    print("Generating figures...")
    figure1_scatter_plot(df, mae, use_vib, output_dir / 'scatter_pred_vs_true.png')
    figure2_mae_comparison(output_dir / 'mae_comparison.png')
    figure3_error_distribution(df, mae, output_dir / 'error_distribution.png')
    
    print(f"\n{'='*70}")
    print(f"✅ All figures generated successfully!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
