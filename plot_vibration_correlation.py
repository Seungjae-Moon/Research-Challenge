#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_vibration_correlation.py — Vibration Feature Correlation Analysis
=======================================================================
Purpose:
  • Visualize correlations between vibration features and sensor readings
  • Generate correlation heatmaps and scatter plots
  • Support multiple datasets and comparison views

Usage:
  # Basic heatmap (vibration features only)
  python plot_vibration_correlation.py --set-id 200 --version v1 --run 1

  # Heatmap with sensors included
  python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --include-sensors

  # Scatter plots for specific pairs
  python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --plot-type scatter

  # Save to custom directory
  python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --output-dir ./plots
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Constants
VIB_COLS = ["fe_RMS", "fe_Kurtosis", "fe_Crest", "fe_Entropy"]
SENSOR_COLS = [f"sensor{i}" for i in range(1, 22)]
OP_COLS = ['op1', 'op2', 'op3']

BASE_TRAIN = Path("data") / "training"
BASE_TEST = Path("data") / "test"


def read_cmapss_like(path: Path) -> pd.DataFrame:
    """Read CMAPSS-style data file."""
    COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + SENSOR_COLS
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    df.columns = COLS
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_vibration_data(set_id: str, version: str, run: int, split: str = 'train') -> pd.DataFrame:
    """Load and merge vibration data with sensor data."""
    base_dir = BASE_TRAIN if split == 'train' else BASE_TEST
    data_dir = base_dir / str(set_id) / version
    
    # Load main sensor data
    sensor_file = data_dir / f"{split}_{run}.txt"
    if not sensor_file.exists():
        raise FileNotFoundError(f"Sensor file not found: {sensor_file}")
    
    df_sensor = read_cmapss_like(sensor_file)
    
    # Load vibration data
    vib_file = data_dir / f"{split}_{run}_vib.csv"
    if not vib_file.exists():
        raise FileNotFoundError(f"Vibration file not found: {vib_file}")
    
    df_vib = pd.read_csv(vib_file)
    
    # Merge on unit and cycle
    df_merged = df_sensor.merge(df_vib[['unit', 'cycle'] + VIB_COLS], 
                                  on=['unit', 'cycle'], 
                                  how='inner')
    
    return df_merged


def plot_correlation_heatmap(df: pd.DataFrame, 
                             feature_cols: List[str],
                             title: str = "Correlation Heatmap",
                             figsize: Tuple[int, int] = (12, 10),
                             output_path: Optional[Path] = None,
                             method: str = 'pearson'):
    """
    Generate correlation heatmap.
    
    Args:
        df: DataFrame with features
        feature_cols: List of column names to include
        title: Plot title
        figsize: Figure size (width, height)
        output_path: Path to save figure (if None, display only)
        method: Correlation method ('pearson', 'spearman', 'kendall')
    """
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr(method=method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"},
                ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Heatmap: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return corr_matrix


def plot_vibration_scatter_matrix(df: pd.DataFrame,
                                   vib_cols: List[str] = VIB_COLS,
                                   title: str = "Vibration Feature Scatter Matrix",
                                   output_path: Optional[Path] = None,
                                   sample_size: Optional[int] = 5000):
    """
    Generate scatter plot matrix for vibration features.
    
    Args:
        df: DataFrame with vibration features
        vib_cols: Vibration column names
        title: Plot title
        output_path: Path to save figure
        sample_size: Number of samples to plot (for performance)
    """
    # Sample data if too large
    if sample_size and len(df) > sample_size:
        df_plot = df[vib_cols].sample(n=sample_size, random_state=42)
        print(f"[INFO] Sampling {sample_size} points from {len(df)} total rows")
    else:
        df_plot = df[vib_cols]
    
    # Create pairplot
    g = sns.pairplot(df_plot, 
                     diag_kind='kde',
                     plot_kws={'alpha': 0.3, 's': 10},
                     diag_kws={'shade': True})
    
    g.fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Scatter matrix: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_vib_vs_sensors_top_correlations(df: pd.DataFrame,
                                          n_top: int = 5,
                                          title: str = "Top Vibration-Sensor Correlations",
                                          output_path: Optional[Path] = None,
                                          sample_size: int = 3000):
    """
    Plot scatter plots for top N correlations between vibration and sensors.
    
    Args:
        df: DataFrame with vibration and sensor features
        n_top: Number of top correlations to plot
        title: Plot title
        output_path: Path to save figure
        sample_size: Number of samples for scatter plots
    """
    # Calculate correlations between vibration and sensors
    vib_available = [c for c in VIB_COLS if c in df.columns]
    sensor_available = [c for c in SENSOR_COLS if c in df.columns]
    
    corr_pairs = []
    for vib in vib_available:
        for sensor in sensor_available:
            corr = df[[vib, sensor]].corr().iloc[0, 1]
            corr_pairs.append((vib, sensor, abs(corr), corr))
    
    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = corr_pairs[:n_top]
    
    # Sample data
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42)
    else:
        df_plot = df
    
    # Create subplots
    n_cols = min(3, n_top)
    n_rows = (n_top + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_top == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (vib, sensor, abs_corr, corr) in enumerate(top_pairs):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(df_plot[sensor], df_plot[vib], alpha=0.3, s=10)
        
        # Add trend line
        z = np.polyfit(df_plot[sensor].dropna(), df_plot[vib].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_plot[sensor].min(), df_plot[sensor].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
        
        ax.set_xlabel(sensor, fontsize=10)
        ax.set_ylabel(vib, fontsize=10)
        ax.set_title(f"r = {corr:.3f}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_top, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Top correlations: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return top_pairs


def plot_rul_vs_vibration(df: pd.DataFrame,
                          title: str = "RUL vs Vibration Features",
                          output_path: Optional[Path] = None,
                          sample_size: int = 5000):
    """
    Plot RUL correlation with vibration features.
    
    Args:
        df: DataFrame with RUL and vibration features
        title: Plot title
        output_path: Path to save figure
        sample_size: Number of samples to plot
    """
    if 'RUL' not in df.columns:
        print("[WARN] RUL column not found, skipping RUL correlation plot")
        return
    
    vib_available = [c for c in VIB_COLS if c in df.columns]
    
    # Sample data
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42)
    else:
        df_plot = df
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, vib_col in enumerate(vib_available[:4]):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(df_plot['RUL'], df_plot[vib_col], alpha=0.3, s=10)
        
        # Calculate correlation
        corr = df_plot[['RUL', vib_col]].corr().iloc[0, 1]
        
        ax.set_xlabel('RUL (cycles)', fontsize=10)
        ax.set_ylabel(vib_col, fontsize=10)
        ax.set_title(f"{vib_col} vs RUL (r = {corr:.3f})", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] RUL correlation: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_correlation_report(corr_matrix: pd.DataFrame, 
                                output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Generate a text report of correlations.
    
    Args:
        corr_matrix: Correlation matrix
        output_path: Path to save report (CSV format)
    
    Returns:
        DataFrame with correlation pairs sorted by strength
    """
    # Extract upper triangle (avoid duplicates)
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            pairs.append({
                'Feature_1': col1,
                'Feature_2': col2,
                'Correlation': corr_val,
                'Abs_Correlation': abs(corr_val)
            })
    
    df_report = pd.DataFrame(pairs).sort_values('Abs_Correlation', ascending=False)
    
    if output_path:
        df_report.to_csv(output_path, index=False)
        print(f"[SAVED] Correlation report: {output_path}")
    
    return df_report


def main():
    parser = argparse.ArgumentParser(description="Vibration Correlation Analysis")
    
    # Data selection
    parser.add_argument('--set-id', type=str, default='200',
                        help='Dataset ID (e.g., 200)')
    parser.add_argument('--version', type=str, default='v1',
                        help='Dataset version (e.g., v1, v2, v3, v4)')
    parser.add_argument('--run', type=int, default=1,
                        help='Run number')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Data split to use')
    
    # Plot options
    parser.add_argument('--plot-type', type=str, default='all', 
                        choices=['heatmap', 'scatter', 'top-corr', 'rul', 'all'],
                        help='Type of plot to generate')
    parser.add_argument('--include-sensors', action='store_true',
                        help='Include sensor features in correlation analysis')
    parser.add_argument('--n-top', type=int, default=6,
                        help='Number of top correlations to plot')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Sample size for scatter plots')
    parser.add_argument('--corr-method', type=str, default='pearson',
                        choices=['pearson', 'spearman', 'kendall'],
                        help='Correlation method')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./correlation_plots',
                        help='Directory to save plots')
    parser.add_argument('--show', action='store_true',
                        help='Display plots in addition to saving')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n[INFO] Loading data: set_id={args.set_id}, version={args.version}, run={args.run}, split={args.split}")
    df = load_vibration_data(args.set_id, args.version, args.run, args.split)
    print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Add RUL if training data
    if args.split == 'train' and 'RUL' not in df.columns:
        max_cycles = df.groupby('unit')['cycle'].transform('max')
        df['RUL'] = (max_cycles - df['cycle']).astype(float)
        print(f"[INFO] Computed RUL column")
    
    # Determine features to analyze
    vib_available = [c for c in VIB_COLS if c in df.columns]
    print(f"[INFO] Vibration features: {vib_available}")
    
    if args.include_sensors:
        sensor_available = [c for c in SENSOR_COLS if c in df.columns]
        feature_cols = vib_available + sensor_available
        print(f"[INFO] Including {len(sensor_available)} sensor features")
    else:
        feature_cols = vib_available
    
    tag = f"{args.set_id}_{args.version}_run{args.run}_{args.split}"
    
    # Generate plots
    if args.plot_type in ['heatmap', 'all']:
        print("\n[PLOT] Generating correlation heatmap...")
        output_path = output_dir / f"{tag}_corr_heatmap.png" if not args.show else None
        corr_matrix = plot_correlation_heatmap(
            df, feature_cols,
            title=f"Correlation Heatmap ({tag})",
            figsize=(14, 12) if args.include_sensors else (8, 7),
            output_path=output_path,
            method=args.corr_method
        )
        
        # Save correlation report
        report_path = output_dir / f"{tag}_corr_report.csv"
        generate_correlation_report(corr_matrix, report_path)
    
    if args.plot_type in ['scatter', 'all']:
        print("\n[PLOT] Generating scatter matrix...")
        output_path = output_dir / f"{tag}_scatter_matrix.png" if not args.show else None
        plot_vibration_scatter_matrix(
            df, vib_cols=vib_available,
            title=f"Vibration Scatter Matrix ({tag})",
            output_path=output_path,
            sample_size=args.sample_size
        )
    
    if args.plot_type in ['top-corr', 'all'] and args.include_sensors:
        print("\n[PLOT] Generating top vibration-sensor correlations...")
        output_path = output_dir / f"{tag}_top_vib_sensor_corr.png" if not args.show else None
        top_pairs = plot_vib_vs_sensors_top_correlations(
            df, n_top=args.n_top,
            title=f"Top Vibration-Sensor Correlations ({tag})",
            output_path=output_path,
            sample_size=args.sample_size
        )
        
        # Print top correlations
        print("\n[INFO] Top correlations:")
        for vib, sensor, abs_corr, corr in top_pairs:
            print(f"  {vib:15s} <-> {sensor:10s} : {corr:6.3f}")
    
    if args.plot_type in ['rul', 'all'] and 'RUL' in df.columns:
        print("\n[PLOT] Generating RUL vs vibration plots...")
        output_path = output_dir / f"{tag}_rul_vs_vibration.png" if not args.show else None
        plot_rul_vs_vibration(
            df,
            title=f"RUL vs Vibration Features ({tag})",
            output_path=output_path,
            sample_size=args.sample_size
        )
    
    print(f"\n[COMPLETE] All plots saved to: {output_dir}")
    print(f"[COMPLETE] Tag: {tag}\n")


if __name__ == '__main__':
    main()
