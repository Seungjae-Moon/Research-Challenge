# src/ims_calib.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMS 참고데이터로 '보정값(calibration constants)'만 추정해서 calib.json 으로 저장.
- CMAPSS 파일은 건드리지 않음.
- 산출물: calib.json (k_* 비례상수 + feature 분위수들)

Usage:
  python -m src.ims_calib --ims_dir "D:/IMS" --out "D:/project/out/calib.json" --ims_fs 20000 --max_files 20
"""

from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.io import loadmat  # 추가


# ---------- 유틸 ----------
def welch_psd(x: np.ndarray, fs: float, nperseg: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    f, Pxx = welch(x, fs=fs, window="hann", nperseg=min(nperseg, x.size), noverlap=None, detrend="constant")
    return f, Pxx

def _read_ims_file(path: Path) -> Optional[np.ndarray]:
    """
    IMS 원시 파일 하나를 1채널 시계열로 읽는다.
    - 확장자 없어도 시도
    - 숫자 여러 열이면 첫 번째 숫자열 사용
    - 최소 길이 1024 미만이면 제외
    """
    try:
        # 1) numpy 빠른 시도 (공백 구분, 헤더 없음 가정)
        arr = np.loadtxt(path, dtype=float, ndmin=1)
        arr = np.asarray(arr).squeeze()
        # 2열 이상이면 첫 번째 열만 사용
        if arr.ndim == 2:
            arr = arr[:, 0]
        if arr.size >= 1024:
            return arr.astype(float)
    except Exception:
        pass

    # 2) pandas로 재시도 (공백 구분, 콤마 자동 감지)
    try:
        df = pd.read_csv(path, header=None, sep=None, engine="python")
        # 숫자열만 골라서 가장 왼쪽 숫자열 사용
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return None
        arr = df[num_cols[0]].dropna().to_numpy()
        return arr.astype(float) if arr.size >= 1024 else None
    except Exception:
        return None



def load_ims_signals(
    ims_dir: str | Path,
    max_files: int = 200,           # 전체 상한
    per_folder: int = 30,           # 폴더당 샘플 상한
    skip_dirs: Tuple[str, ...] = ("_normalized",),  # 스킵할 폴더명
) -> List[np.ndarray]:
    """
    ims_dir 아래 모든 하위폴더를 훑어서:
    - 확장자 유무와 관계없이 '파일'이면 시도
    - 폴더별로 최대 per_folder개까지 균형 샘플링
    - 전체 합계가 max_files를 넘지 않게 제한
    """
    ims_dir = Path(ims_dir)
    sigs: List[np.ndarray] = []
    total_candidates = 0

    # 1) 1단계: 폴더별 후보 수집
    subdirs = [p for p in ims_dir.rglob("*") if p.is_dir()]
    # 최상위도 포함
    subdirs = [ims_dir] + subdirs

    for d in subdirs:
        # 스킵 폴더 제외
        if any(s in d.parts for s in skip_dirs):
            continue
        # 폴더 내 파일(확장자 무관)
        files = [p for p in d.iterdir() if p.is_file()]
        if not files:
            continue
        total_candidates += len(files)

        # 정렬(타임스탬프 이름이면 시간순이 됨)
        files = sorted(files)
        # 균형 샘플링: 앞/중간/끝 골고루 뽑기
        if len(files) > per_folder:
            idxs = np.linspace(0, len(files)-1, per_folder, dtype=int)
            files = [files[i] for i in idxs]

        # 파일별 로딩
        for f in files:
            if len(sigs) >= max_files:
                break
            arr = _read_ims_file(f)
            if arr is not None:
                sigs.append(arr)
        if len(sigs) >= max_files:
            break

    print(f"[IMS] usable files: {len(sigs)}  (from {total_candidates} candidates, "
          f"per_folder={per_folder}, max_files={max_files})")
    return sigs



def estimate_1x_fr_from_psd(f: np.ndarray, Pxx: np.ndarray, fmin: float = 5.0, fmax: float = 200.0) -> Optional[float]:
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask): return None
    fr = float(f[mask][np.argmax(Pxx[mask])])
    return fr

def band_energy(f: np.ndarray, Pxx: np.ndarray, center: float, width: float = 0.1) -> float:
    if center <= 0: return 0.0
    lo, hi = center*(1-width), center*(1+width)
    m = (f>=lo)&(f<=hi)
    if not np.any(m): return 0.0
    return float(np.trapezoid(Pxx[m], f[m]))

# ---------- 결과 구조 ----------
@dataclass
class IMSCalibration:
    k_BPFO: float
    k_BPFI: float
    k_BSF: float
    k_FTF: float
    feat_p5: Dict[str, float]
    feat_p50: Dict[str, float]
    feat_p95: Dict[str, float]

# ---------- 핵심 로직 ----------
def fit_ims_calibration(
    ims_signals: List[np.ndarray],
    fs: float = 20000.0,
    max_windows_per_file: int = 4,
) -> IMSCalibration:
    if not ims_signals:
        # IMS가 없더라도 안전한 기본값
        return IMSCalibration(
            k_BPFO=7.2, k_BPFI=5.4, k_BSF=4.9, k_FTF=0.4,
            feat_p5={"RMS":0.01,"Kurtosis":2.4,"Crest":1.5,"Entropy":2.0},
            feat_p50={"RMS":0.05,"Kurtosis":3.0,"Crest":2.0,"Entropy":2.5},
            feat_p95={"RMS":0.20,"Kurtosis":6.0,"Crest":5.0,"Entropy":3.0},
        )

    fr_list = []
    feats = {"RMS": [], "Kurtosis": [], "Crest": [], "Entropy": []}

    for arr in ims_signals:
        N = arr.size
        wN = max(4096, min(16384, N // max_windows_per_file))
        step = max(1, (N - wN) // max_windows_per_file or 1)
        for s in range(0, N - wN + 1, step):
            x = arr[s:s+wN].astype(float)
            x = x - np.mean(x)
            f, Pxx = welch_psd(x, fs=fs, nperseg=min(8192, wN))
            fr = estimate_1x_fr_from_psd(f, Pxx, 5, 200)
            if fr: fr_list.append(fr)

            rms = float(np.sqrt(np.mean(x**2)))
            crest = float(np.max(np.abs(x)) / (rms + 1e-12))
            kur = float(np.mean(((x - np.mean(x)) / (np.std(x)+1e-12))**4))
            p = Pxx/(np.sum(Pxx)+1e-12)
            ent = float(-np.sum(p*np.log(p+1e-12)))
            feats["RMS"].append(rms); feats["Crest"].append(crest)
            feats["Kurtosis"].append(kur); feats["Entropy"].append(ent)

    if not fr_list: fr_list=[30.0]
    fr_med = float(np.median(fr_list))

    # 대표 PSD
    PSDs = []
    for arr in ims_signals[:min(6, len(ims_signals))]:
        x = arr[:min(32768, arr.size)].astype(float)
        x = x - np.mean(x)
        f, Pxx = welch_psd(x, fs=fs, nperseg=min(8192, x.size))
        PSDs.append((f, Pxx))
    f = PSDs[0][0]
    Pxx_mean = np.mean([P for _, P in PSDs], axis=0)

    def search_k(cands: np.ndarray) -> float:
        best_k, best_e = cands[len(cands)//2], -1.0
        for k in cands:
            e = band_energy(f, Pxx_mean, center=k*fr_med, width=0.08)
            if e > best_e: best_e, best_k = e, k
        return float(best_k)

    k_BPFO = search_k(np.arange(4.0, 9.5, 0.1))
    k_BPFI = search_k(np.arange(5.0, 12.5, 0.1))
    k_BSF  = search_k(np.arange(4.0, 9.5, 0.1))
    k_FTF  = search_k(np.arange(0.3, 0.7, 0.01))

    feat_p5  = {k: float(np.percentile(v, 5))  for k, v in feats.items() if len(v)>=5}
    feat_p50 = {k: float(np.percentile(v, 50)) for k, v in feats.items() if len(v)>=5}
    feat_p95 = {k: float(np.percentile(v, 95)) for k, v in feats.items() if len(v)>=5}

    # 누락 기본값 보강
    def fill(d, k, v): d.setdefault(k, v)
    for nm, base in [("RMS",0.05),("Kurtosis",3.0),("Crest",2.0),("Entropy",2.5)]:
        fill(feat_p5, nm, base*0.2); fill(feat_p50, nm, base); fill(feat_p95, nm, base*4.0)

    return IMSCalibration(k_BPFO, k_BPFI, k_BSF, k_FTF, feat_p5, feat_p50, feat_p95)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_folder", type=int, default=30)
    ap.add_argument("--max_files", type=int, default=200)
    ap.add_argument("--ims_dir", type=str, required=True, help="IMS 루트 디렉터리(참고자료)")
    ap.add_argument("--out", type=str, required=True, help="보정값 저장 경로 e.g., ./out/calib.json")
    ap.add_argument("--ims_fs", type=float, default=20000.0, help="IMS 샘플링(Hz)")
    args = ap.parse_args()

    ims_signals = load_ims_signals(args.ims_dir, max_files=args.max_files, per_folder=args.per_folder)
    calib = fit_ims_calibration(ims_signals, fs=args.ims_fs)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(asdict(calib), f, indent=2, ensure_ascii=False)
    print(f"[OK] saved → {outp}")

if __name__ == "__main__":
    main()
