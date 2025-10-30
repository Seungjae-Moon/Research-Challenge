#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_improve.py — 빠른 MAE 개선 실행
====================================
검증된 최적 설정으로 즉시 학습을 실행합니다.

Usage:
    # 기본 (권장 설정)
    python quick_improve.py
    
    # 커스텀 데이터셋
    python quick_improve.py --set-id data200 --version v3
"""
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set-id', type=str, default='data200')
    parser.add_argument('--version', type=str, default='v4')
    parser.add_argument('--run', type=str, default='auto')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # 최적화된 설정
    optimal_configs = [
        {
            'name': 'Best Config: Vib + Seq50 + Large Model',
            'params': {
                'set-id': args.set_id,
                'version': args.version,
                'run': args.run,
                'vib': 1,                    # Vibration 특성 사용
                'seq-len': 50,               # 더 긴 시퀀스 (더 많은 컨텍스트)
                'test': 'syn',
                'epochs': 50,                # 충분한 학습
                'model': 'tcn_bigru',
                'loss': 'mae',               # 직접적인 MAE 최적화
                'lr': 5e-4,                  # 안정적인 학습률
                'batch-size': 128,           # 더 작은 배치로 안정성
                'hidden': 256,               # 더 큰 모델
                'layers': 2,
                'dropout': 0.15,             # 약간 낮은 드롭아웃
                'rul-clip': 125.0,
                'grad-clip': 1.0,
                'ema': 1,
                'patience': 8,
                'device': args.device
            }
        }
    ]
    
    for config in optimal_configs:
        print(f"\n{'='*80}")
        print(f"🚀 Running: {config['name']}")
        print(f"{'='*80}\n")
        
        cmd = ['python', 'train_eval_fusion.py']
        for key, val in config['params'].items():
            cmd.extend([f'--{key}', str(val)])
        
        print(f"Command: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n❌ Training failed!")
            sys.exit(1)
        else:
            print(f"\n✅ Training completed successfully!")
    
    print(f"\n{'='*80}")
    print("🎉 All experiments completed!")
    print("Check runs/ directory for results")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
