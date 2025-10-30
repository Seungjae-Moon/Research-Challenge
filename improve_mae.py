#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
improve_mae.py — MAE 개선 실험 스크립트
=========================================
여러 설정을 자동으로 시도하여 최적의 MAE를 찾습니다.

Usage:
    python improve_mae.py --set-id data200 --version v4
"""
import argparse
import subprocess
import json
from pathlib import Path
import pandas as pd

def run_experiment(args_dict, tag):
    """단일 실험 실행"""
    cmd = ["python", "train_eval_fusion.py"]
    for key, val in args_dict.items():
        cmd.extend([f"--{key}", str(val)])
    
    print(f"\n{'='*80}")
    print(f"[EXPERIMENT: {tag}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    # 결과 읽기
    run_tag = f"{args_dict['set-id']}_{args_dict['version']}_run{args_dict.get('run', 'auto')}"
    eval_path = Path("runs") / run_tag / "eval_summary.json"
    
    if eval_path.exists():
        with open(eval_path) as f:
            results = json.load(f)
        return results
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set-id', type=str, default='data200')
    parser.add_argument('--version', type=str, default='v4')
    parser.add_argument('--run', type=str, default='auto')
    args = parser.parse_args()
    
    experiments = []
    
    # =================== 실험 1: Baseline (현재 설정) ===================
    exp1 = {
        'set-id': args.set_id,
        'version': args.version,
        'run': args.run,
        'vib': 0,
        'seq-len': 20,
        'test': 'syn',
        'epochs': 40,
        'model': 'tcn_bigru',
        'loss': 'quantile',
        'quantile': 0.5,
        'lr': 1e-3,
        'batch-size': 256,
        'hidden': 160,
        'dropout': 0.2
    }
    # result1 = run_experiment(exp1, "Baseline (no vib, seq=20)")
    # if result1: experiments.append({'tag': 'Baseline', 'config': exp1, 'results': result1})
    
    # =================== 실험 2: Vibration 추가 ===================
    exp2 = exp1.copy()
    exp2['vib'] = 1
    result2 = run_experiment(exp2, "With Vibration (vib=1)")
    if result2: experiments.append({'tag': 'With_Vibration', 'config': exp2, 'results': result2})
    
    # =================== 실험 3: 더 긴 시퀀스 + Vibration ===================
    exp3 = exp2.copy()
    exp3['seq-len'] = 30
    result3 = run_experiment(exp3, "Longer Sequence (seq=30, vib=1)")
    if result3: experiments.append({'tag': 'Seq30_Vib', 'config': exp3, 'results': result3})
    
    # =================== 실험 4: 더 큰 모델 ===================
    exp4 = exp2.copy()
    exp4['hidden'] = 256
    exp4['seq-len'] = 30
    result4 = run_experiment(exp4, "Larger Model (hidden=256, seq=30, vib=1)")
    if result4: experiments.append({'tag': 'Large_Model', 'config': exp4, 'results': result4})
    
    # =================== 실험 5: Huber Loss ===================
    exp5 = exp2.copy()
    exp5['loss'] = 'huber'
    exp5['huber-delta'] = 5.0
    exp5['seq-len'] = 30
    result5 = run_experiment(exp5, "Huber Loss (delta=5.0, seq=30, vib=1)")
    if result5: experiments.append({'tag': 'Huber_Loss', 'config': exp5, 'results': result5})
    
    # =================== 실험 6: 낮은 학습률 + 더 긴 학습 ===================
    exp6 = exp2.copy()
    exp6['lr'] = 5e-4
    exp6['epochs'] = 60
    exp6['seq-len'] = 30
    result6 = run_experiment(exp6, "Lower LR (lr=5e-4, epochs=60, seq=30, vib=1)")
    if result6: experiments.append({'tag': 'Lower_LR', 'config': exp6, 'results': result6})
    
    # =================== 결과 요약 ===================
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    
    results_data = []
    for exp in experiments:
        tag = exp['tag']
        syn_metrics = exp['results'].get('synthetic_metrics', {})
        fd_metrics = exp['results'].get('fd004_metrics', {})
        
        row = {
            'Experiment': tag,
            'Vib': exp['config']['vib'],
            'Seq_Len': exp['config']['seq-len'],
            'Hidden': exp['config']['hidden'],
            'Loss': exp['config']['loss'],
            'LR': exp['config']['lr'],
            'Epochs': exp['config']['epochs'],
            'SYN_MAE': syn_metrics.get('MAE', -1),
            'SYN_RMSE': syn_metrics.get('RMSE', -1),
            'FD_MAE': fd_metrics.get('MAE', -1) if fd_metrics else -1,
            'FD_RMSE': fd_metrics.get('RMSE', -1) if fd_metrics else -1,
        }
        results_data.append(row)
        
        print(f"\n{tag}:")
        print(f"  Config: vib={row['Vib']}, seq={row['Seq_Len']}, hidden={row['Hidden']}, loss={row['Loss']}")
        print(f"  SYN -> MAE={row['SYN_MAE']:.2f}, RMSE={row['SYN_RMSE']:.2f}")
        if row['FD_MAE'] > 0:
            print(f"  FD  -> MAE={row['FD_MAE']:.2f}, RMSE={row['FD_RMSE']:.2f}")
    
    # 결과 저장
    df_results = pd.DataFrame(results_data)
    df_results = df_results.sort_values('SYN_MAE')
    
    output_path = Path("experiment_results.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\n[SAVED] Results: {output_path}")
    
    # 최고 성능
    best = df_results.iloc[0]
    print(f"\n{'='*100}")
    print(f"🏆 BEST RESULT: {best['Experiment']}")
    print(f"   SYN MAE: {best['SYN_MAE']:.2f}")
    print(f"   Config: vib={best['Vib']}, seq={best['Seq_Len']}, hidden={best['Hidden']}, loss={best['Loss']}")
    print(f"{'='*100}\n")

if __name__ == '__main__':
    main()
