#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Digital Twin Engine — CMAPSS-style
=====================================================================
- FD004 참조 통계(평균/공분산, RUL 분포) 자동 로드
- calib_mode: ["none","moments","cov"] → 실제 센서값에 보정 적용
- TEST는 항상 '고장 이전'에서 트렁케이션, 이제 FD004 RUL 분포에서 샘플링
- vib feature는 간단 버전(옵션) 유지
- set_id/version/run 인덱싱 + manifest 기록

CLI 예:
  python digital_twin_engine.py --units 200 --seed 7
  python digital_twin_engine.py --calib-mode cov --cov-blend-lam 0.85 --no-vib

경로/기본값은 Windows를 기준으로 설정되었으나, --fd004-train, --fd004-rul, --train-root 등으로 재지정 가능.
"""

from __future__ import annotations
import argparse, json, math, os, random, re, time, hashlib, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ========= 기본 경로 (필요 시 CLI로 override) ===============================
DEFAULT_TRAIN_ROOT = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\training")
DEFAULT_TEST_ROOT  = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\test")
DEFAULT_RUL_ROOT   = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\rul")

DEFAULT_FD004_TRAIN = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\cmapss_raw\train_FD004.txt")
DEFAULT_FD004_RUL   = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\cmapss_raw\RUL_FD004.txt")

DEFAULT_CALIB_MODE = "cov"     # ["none","moments","cov"]
DEFAULT_COV_LAM    = 0.85      # 0~1, 0.8~0.9 권장
# =========================================================================

COLS = ['unit','cycle','op1','op2','op3'] + [f"sensor{i}" for i in range(1,22)]
SENSOR_COLS = [f"sensor{i}" for i in range(1,22)]

@dataclass
class DTConfig:
    # core
    units: int = 200
    min_cycles: int = 150
    max_cycles: int = 380
    seed: int = 7
    train_split: float = 0.8
    set_id: Optional[str] = None
    # failure
    burn_in: int = 40
    fail_threshold: float = 0.06
    hazard_beta: float = 2.2
    hazard_eta: float = 420.0
    # physical mapping knobs
    a_pr: float = 1.0
    gamma_pr: float = 1.2
    beta_pr: float = 0.8
    b1_texh: float = 260.0
    eta_t_min: float = 0.30
    k_texh: float = 0.75
    k_pr:   float = 0.90
    k_flow: float = 0.85
    noise_base: float = 0.02
    noise_alt_gain: float = 0.01
    noise_thr_gain: float = 0.01
    corr_base:  float = 0.40
    corr_ht_gain: float = 0.05
    # statistical calibration
    calib_mode: str = DEFAULT_CALIB_MODE  # ["none","moments","cov"]
    cov_blend_lam: float = DEFAULT_COV_LAM
    # ref files (overridable)
    fd004_train_path: Path = DEFAULT_FD004_TRAIN
    fd004_rul_path: Path   = DEFAULT_FD004_RUL
    # output roots
    train_root: Path = DEFAULT_TRAIN_ROOT
    test_root: Path  = DEFAULT_TEST_ROOT
    rul_root: Path   = DEFAULT_RUL_ROOT
    # features
    save_vib: bool = True

# ----------------------- 버저닝 -----------------------
def ensure_common_version_dirs(cfg: DTConfig) -> tuple[Path, Path, Path]:
    for base in (cfg.train_root, cfg.test_root, cfg.rul_root):
        (base / cfg.set_id).mkdir(parents=True, exist_ok=True)
    def max_v(base: Path) -> int:
        root = base / cfg.set_id
        vers = [p for p in root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
        return max([int(p.name[1:]) for p in vers], default=0)
    v = max(max_v(cfg.train_root), max_v(cfg.test_root), max_v(cfg.rul_root)) + 1
    train_dir = cfg.train_root / cfg.set_id / f"v{v}"
    test_dir  = cfg.test_root  / cfg.set_id / f"v{v}"
    rul_dir   = cfg.rul_root   / cfg.set_id / f"v{v}"
    for d in (train_dir, test_dir, rul_dir): d.mkdir(parents=True, exist_ok=True)
    return train_dir, test_dir, rul_dir

def lock(path: Path):
    L = path / ".lock"
    L.write_text(str(time.time()), encoding="utf-8")
    return L

def unlock(L: Path):
    try: L.unlink()
    except Exception: pass

_run_pat = re.compile(r'^(train|test|RUL)_(\d+)\.txt$', re.IGNORECASE)
def next_run_index(train_dir: Path, test_dir: Path, rul_dir: Path) -> int:
    idx = []
    for d in (train_dir, test_dir, rul_dir):
        for p in d.glob("*.txt"):
            m = _run_pat.match(p.name)
            if m: idx.append(int(m.group(2)))
    return (max(idx) + 1) if idx else 1

# ----------------------- FD004 통계 -----------------------
def _read_cmapss_like(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df.dropna(axis=1, how="all", inplace=True)
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS
    return df

def _read_rul(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=int)
    if arr.ndim == 0: arr = np.array([int(arr)], dtype=int)
    return arr

def load_ref_stats_and_rul(cfg: DTConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not cfg.fd004_train_path.exists():
        raise FileNotFoundError(f"FD004 train not found: {cfg.fd004_train_path}")
    if not cfg.fd004_rul_path.exists():
        raise FileNotFoundError(f"FD004 RUL not found: {cfg.fd004_rul_path}")
    df = _read_cmapss_like(cfg.fd004_train_path)
    F = df[SENSOR_COLS].values.astype(float)
    mu_ref = F.mean(axis=0)
    cov_ref = np.cov(F, rowvar=False) + 1e-6*np.eye(21)
    rul_ref = _read_rul(cfg.fd004_rul_path)
    return mu_ref, cov_ref, rul_ref

def fit_cov_warp(X_sample: np.ndarray, mu_ref: np.ndarray, cov_ref: np.ndarray, lam: float):
    lam = float(np.clip(lam, 0.0, 1.0))
    if lam <= 0.0: return None
    Xc = X_sample - X_sample.mean(axis=0)
    cov_a = np.cov(Xc, rowvar=False) + 1e-6*np.eye(Xc.shape[1])
    Wa = np.linalg.cholesky(cov_a); Wb = np.linalg.cholesky(cov_ref)
    A_full = np.linalg.inv(Wa).T @ Wb.T
    mu_a = X_sample.mean(axis=0)
    def apply_warp_block(Xblock: np.ndarray):
        Xc = Xblock - mu_a
        Xw = Xc @ A_full
        Xb = (1.0 - lam)*Xc + lam*Xw
        mu_blend = (1.0 - lam)*mu_a + lam*mu_ref
        return Xb + mu_blend
    return apply_warp_block

def apply_moments_inplace(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                          mu_ref: np.ndarray, cov_ref: np.ndarray) -> None:
    Xtr = df_tr[SENSOR_COLS].to_numpy()
    mu_tr = np.nanmean(Xtr, axis=0)
    sd_tr = np.nanstd(Xtr, axis=0) + 1e-9
    sd_ref = np.sqrt(np.clip(np.diag(cov_ref), 1e-12, None))
    df_tr[SENSOR_COLS] = ((Xtr - mu_tr)/sd_tr)*sd_ref + mu_ref
    Xte = df_te[SENSOR_COLS].to_numpy()
    df_te[SENSOR_COLS] = ((Xte - mu_tr)/sd_tr)*sd_ref + mu_ref

# ----------------------- Physics helpers -----------------------
def isa_atmosphere_from_hnorm(h_norm: float) -> Tuple[float, float]:
    h = 11000.0 * float(np.clip(h_norm, 0.0, 1.0))
    T0=288.15; p0=101.325; L=0.0065; g=9.80665; R=287.058
    T = T0 - L*h
    p = p0 * (1.0 - (L*h)/T0) ** (g/(R*L))
    return T, p

def map_ops_to_physical(op_row: np.ndarray) -> Dict[str, float]:
    N1=float(np.clip(op_row[0],0.0,1.0)); h_norm=float(np.clip(op_row[1],0.0,1.0)); thr=float(np.clip(op_row[2],0.0,1.0))
    T_amb, p_amb = isa_atmosphere_from_hnorm(h_norm)
    return {"N1":N1, "h_norm":h_norm, "thr":thr, "T_amb":T_amb, "p_amb":p_amb}

def brayton_light(N1: float, thr: float, Tamb: float, pamb: float, hc: float, ht: float, cfg: DTConfig) -> Tuple[float, float, float]:
    PR_c = 1.03 + cfg.a_pr * (max(N1,0.0)**cfg.gamma_pr) * (max(hc,0.0)**cfg.beta_pr)
    PR_c = max(PR_c, 1.03)
    eta_t = 0.70*ht + cfg.eta_t_min
    T_exh = Tamb + cfg.b1_texh * (thr / max(eta_t, 1e-3))
    mdot_like = (N1+1e-6) * (pamb / max(Tamb,1.0))
    return PR_c, T_exh, mdot_like

# ----------------------- Operation sampler -----------------------
def sample_ops(T: int, rng: np.random.Generator) -> np.ndarray:
    if T < 8:
        base = rng.uniform(0.45, 0.7, size=3)
        op = base + rng.normal(0, np.array([0.02,0.015,0.02]), size=(T,3))
        return np.clip(op, 0.25, 0.98).astype(np.float32)

    t = np.arange(T); base = rng.uniform(0.45, 0.7, size=3); op = np.zeros((T,3), dtype=np.float32)
    def wav(c, a, per): return c + a*np.sin(2*np.pi*t/per + rng.uniform(0,2*np.pi))
    pat = rng.choice(["steady","step","ramp","jitter"], p=[0.25,0.30,0.25,0.20])
    if pat=="steady":
        op[:,0]=wav(base[0],0.03,90); op[:,1]=wav(base[1],0.02,110); op[:,2]=wav(base[2],0.03,80)
    elif pat=="step":
        k=rng.integers(T//4,3*T//4)
        def wavN(c, a, per, n): tt=np.arange(n); return c + a*np.sin(2*np.pi*tt/per + rng.uniform(0,2*np.pi))
        op[:k,0]=wavN(base[0]-0.05,0.02,100,k); op[k:,0]=wavN(base[0]+0.07,0.02,70,T-k)
        op[:k,1]=wavN(base[1],0.015,100,k);    op[k:,1]=wavN(base[1]+0.04,0.015,90,T-k)
        op[:k,2]=wavN(base[2]-0.03,0.02,95,k); op[k:,2]=wavN(base[2]+0.05,0.02,60,T-k)
    elif pat=="ramp":
        r=np.linspace(-0.06,0.08,T)
        op[:,0]=base[0]+r+0.02*rng.standard_normal(T)
        op[:,1]=base[1]+0.6*r+0.015*rng.standard_normal(T)
        op[:,2]=base[2]+0.8*r+0.02*rng.standard_normal(T)
    else:
        op=base + rng.normal(0, np.array([0.035,0.02,0.03]), size=(T,3))
    drift = rng.normal(0, np.array([0.0008,0.0005,0.0007]), size=(T,3)).cumsum(axis=0)
    phi=0.85; eps=rng.normal(0,[0.01,0.008,0.01], size=(T,3)); ar=np.zeros_like(op); ar[0]=eps[0]
    for tt in range(1,T): ar[tt]=phi*ar[tt-1]+eps[tt]
    op = np.clip(op + drift + ar, 0.25, 0.98).astype(np.float32)
    return op

# ----------------------- Health & failure -----------------------
def degrade_health(op: np.ndarray, alpha: Dict[str,float], cfg: DTConfig, rng: np.random.Generator) -> Dict[str,np.ndarray]:
    T=len(op); op1,op2,op3=op[:,0],op[:,1],op[:,2]
    h={k: np.ones(T, dtype=np.float32) for k in ["fan","comp","turb","brg"]}
    surge=np.zeros(T,bool); lube=np.zeros(T,bool); crack=np.zeros(T,bool)
    def apply_accel_decay(arr: np.ndarray, t0: int, mag: float, tau: int):
        T_rem=len(arr)-t0; L=min(tau,T_rem);
        if L<=0: return
        k=np.arange(L); decay=mag*(1.0-np.exp(-(k+1)/float(tau))); arr[t0:t0+L]-=decay
    for t in range(1,T):
        load=0.45*op1[t]+0.15*op2[t]+0.55*op3[t]; temp=op2[t]
        for comp in ("fan","comp","turb","brg"):
            base=alpha[comp]*(1+1.7*load**2+0.6*temp**2)
            dh=-base*(0.55*load+0.45*load**2)+rng.normal(0, alpha[comp]*0.05)
            h[comp][t]=max(0.0, min(h[comp][t-1]+dh, h[comp][t-1]))
        if (t>cfg.burn_in) and (load>0.75) and (h["comp"][t]<0.6) and rng.random()<0.003:
            surge[t]=True; apply_accel_decay(h["comp"],t,0.030,12); apply_accel_decay(h["turb"],t,0.018,10)
        if (t>cfg.burn_in) and (h["brg"][t]<0.7) and (op3[t]>0.7) and rng.random()<0.003:
            lube[t]=True; apply_accel_decay(h["brg"],t,0.035,14)
        if (t>cfg.burn_in) and (h["turb"][t]<0.5) and rng.random()<0.002:
            crack[t]=True; apply_accel_decay(h["turb"],t,0.028,16)
        for comp in h: h[comp][t]=float(np.clip(h[comp][t],0.0,1.0))
    h["surge"]=surge; h["lube"]=lube; h["crack"]=crack
    return h

def hazard_fail_time(op: np.ndarray, h: Dict[str,np.ndarray], cfg: DTConfig, rng: np.random.Generator) -> int | None:
    T=len(op)
    for t in range(T):
        if t<cfg.burn_in: continue
        if min(h["fan"][t],h["comp"][t],h["turb"][t],h["brg"][t])<=cfg.fail_threshold: return t+1
    for t in range(cfg.burn_in, T):
        load=0.5*op[t,2]+0.35*op[t,0]+0.15*op[t,1]
        beta,eta=cfg.hazard_beta,cfg.hazard_eta
        lam=(beta/eta)*((t+1)/eta)**(beta-1)*(1+1.2*load)
        h_min=min(h["fan"][t],h["comp"][t],h["turb"][t],h["brg"][t]); dmg=max(0.0,1.0-h_min)
        lam*=math.exp(3.0*dmg)
        if rng.random() < 1-math.exp(-lam): return t+1
    return None

# ----------------------- Sensor generator -----------------------
def gen_sensors(op: np.ndarray, h: Dict[str,np.ndarray], rng: np.random.Generator, cfg: DTConfig) -> np.ndarray:
    T=len(op); hf,hc,ht,hb=h["fan"],h["comp"],h["turb"],h["brg"]
    X=np.zeros((T,21), dtype=np.float32)
    N1_arr=np.zeros(T); thr_arr=np.zeros(T); Tamb_arr=np.zeros(T); pamb_arr=np.zeros(T)
    for t in range(T):
        m=map_ops_to_physical(op[t,:]); N1_arr[t]=m["N1"]; thr_arr[t]=m["thr"]; Tamb_arr[t]=m["T_amb"]; pamb_arr[t]=m["p_amb"]
    PR_like=np.zeros(T); T_exh_like=np.zeros(T); mdot_raw=np.zeros(T)
    for t in range(T):
        PR_c,T_exh,mdot_like=brayton_light(N1_arr[t],thr_arr[t],Tamb_arr[t],pamb_arr[t],hc[t],ht[t],cfg)
        PR_like[t]=max(PR_c,1.03); T_exh_like[t]=max(T_exh/288.15, Tamb_arr[t]/288.15+0.02); mdot_raw[t]=mdot_like
    mdot_max=max(float(mdot_raw.max()),1e-6); flow_like=0.10+0.90*(mdot_raw/mdot_max)
    Tamb_norm=Tamb_arr/288.15
    # mapping
    X[:,0]=0.10+cfg.k_texh*T_exh_like+0.05*thr_arr+0.15*(1.0-ht)
    X[:,1]=0.05+cfg.k_pr*PR_like-0.10*(1.0-hc)
    X[:,2]=0.05+cfg.k_flow*flow_like-0.10*(1.0-hf)
    X[:,3]=0.12+0.25*thr_arr+0.55*(1.0-hb)+0.10*(T_exh_like-Tamb_norm)
    X[:,4]=0.10+0.50*T_exh_like+0.15*thr_arr+0.25*(1.0-ht)
    X[:,5]=0.08+0.65*PR_like+0.10*(1.0-hc)
    X[:,6]=0.10+0.75*flow_like+0.10*(1.0-hf)
    X[:,7]=0.10+0.40*thr_arr+0.45*(1.0-hb)
    X[:,8]=0.10+0.50*flow_like+0.10*(1.0-hc)+0.10*(1.0-ht)
    X[:,9]=0.10+0.55*thr_arr+0.30*(1.0-hb)
    X[:,10]=0.08+0.60*PR_like+0.20*Tamb_norm+0.10*(1.0-hc)
    X[:,11]=0.10+0.35*thr_arr+0.55*(1.0-ht)
    X[:,12]=0.10+0.55*thr_arr+0.40*(1.0-hf)
    X[:,13]=0.10+0.58*flow_like+0.10*(1.0-hb)+0.10*(1.0-ht)
    X[:,14]=0.10+0.55*PR_like+0.20*(1.0-hc)
    X[:,15]=0.10+0.45*T_exh_like+0.45*(1.0-ht)
    X[:,16]=0.10+0.58*thr_arr+0.25*(1.0-hc)+0.20*(1.0-hb)
    X[:,17]=0.10+0.60*flow_like+0.20*(1.0-hf)
    X[:,18]=0.10+0.40*thr_arr+0.52*(1.0-hb)
    X[:,19]=0.10+0.60*T_exh_like+0.35*(1.0-ht)
    X[:,20]=0.10+0.55*PR_like+0.45*(1.0-hc)

    # within-group correlated noise  **M3: 공분산 1회 샘플링으로 수정**
    h_mean=float(h["turb"].mean())
    noise = float(np.clip(cfg.noise_base + cfg.noise_alt_gain*np.mean(op[:,1]) + cfg.noise_thr_gain*np.mean(op[:,2]),
                          0.005, 0.06))
    sigma2=noise**2
    groups={"temp":[0,4,15,19],"press":[1,5,10,14,20],"flow":[2,6,8,13,17],"thr":[3,7,9,11,12,16,18]}
    cov = sigma2*np.eye(21, dtype=np.float32)
    in_corr=float(np.clip(cfg.corr_base + cfg.corr_ht_gain*(1.0-h_mean), 0.0, 0.95))
    for g in groups.values():
        for i in g:
            for j in g:
                if i!=j: cov[i,j]=in_corr*sigma2
    # (수정점) 그룹 루프 밖에서 한 번만 샘플링
    eps = rng.multivariate_normal(np.zeros(21), cov, size=T).astype(np.float32)

    X = np.clip(X + eps, 0.0, 1.8).astype(np.float32)
    return X

# ----------------------- Vibration features (physics-informed v2.1, P0) -----------------------
def _hann(N: int, a0: float = 0.5) -> np.ndarray:
    n = np.arange(N)
    return a0 - (1.0 - a0) * np.cos(2 * np.pi * n / (N - 1))

def _rotor_proxy(op3: np.ndarray, s8: np.ndarray) -> np.ndarray:
    fr = 30.0 + 160.0 * op3 + 90.0 * s8
    return np.clip(fr, 10.0, 500.0)

def _safe_series(df: pd.DataFrame, col: str, default: float) -> np.ndarray:
    if col in df.columns:
        v = df[col].to_numpy(float)
        v[~np.isfinite(v)] = default
        return v
    else:
        return np.full(len(df), default, dtype=float)

def synthesize_vibration_features(df_with_rul: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    fs = 4096.0
    N = 2048
    t = np.arange(N) / fs
    w = _hann(N, 0.5).astype(np.float32)

    # drivers
    op3 = _safe_series(df_with_rul, "op3", 0.5)
    s8  = _safe_series(df_with_rul, "sensor8", 0.5)

    # progress & health
    if "cycle" in df_with_rul.columns and "unit" in df_with_rul.columns:
        max_c_by_u = df_with_rul.groupby("unit")["cycle"].transform("max").to_numpy(float)
        progress = (df_with_rul["cycle"].to_numpy(float) / (max_c_by_u + 1e-9)).clip(0.0, 1.0)
    else:
        progress = np.linspace(0.0, 1.0, len(df_with_rul))

    h_brg = _safe_series(df_with_rul, "h_brg", np.nan)
    h_tur = _safe_series(df_with_rul, "h_turb", np.nan)
    h_brg = np.where(np.isfinite(h_brg), h_brg, 1.0 - 0.20 * progress)
    h_tur = np.where(np.isfinite(h_tur), h_tur, 1.0 - 0.15 * progress)
    health = np.clip(0.5 * (1.0 - h_brg) + 0.3 * (1.0 - h_tur), 0.0, 1.0)

    rotor = _rotor_proxy(op3, s8)
    rotor_norm = ((rotor - 10.0) / (500.0 - 10.0)).clip(0.0, 1.0)

    # amplitude model  **V2: late-stage reboost로 RMS std 확장**
    A0, a1, a2, a3 = 0.12, 0.10, 0.08, 0.18
    stage_gain = np.where(
        progress < 0.4, 0.9,
        np.where(progress < 0.8, 1.05, 1.22 + 0.08 * rng.uniform(-1, 1, size=len(progress)))
    )
    drift = 0.020 * progress
    A = (A0 + a1 * op3 + a2 * rotor_norm + a3 * health + drift) * stage_gain
    A = np.clip(A, 0.05, 0.70)

    # heteroskedastic noise
    base_noise = 0.12
    noise_std = base_noise + 0.10 * op3 + 0.10 * health
    noise_std = np.clip(noise_std, 0.05, 0.35)

    # sparse impulses
    p_imp = np.clip(0.002 + 0.02 * health, 0.0, 0.07)

    # output frame (+ fe_RMS_norm 추가)
    cols = ["unit", "cycle"]
    if "RUL" in df_with_rul.columns: cols.append("RUL")
    cols += ["fe_RMS","fe_RMS_norm","fe_Kurtosis","fe_Crest","fe_Entropy"]
    out = pd.DataFrame(index=df_with_rul.index, columns=cols, dtype=float)
    out[["unit","cycle"]] = df_with_rul[["unit","cycle"]].values
    if "RUL" in df_with_rul.columns:
        out["RUL"] = df_with_rul["RUL"].values

    # per-row synthesis
    for i, idx in enumerate(df_with_rul.index):
        fr = rotor[i]
        amp = A[i]

        # base tones
        s  = 1.00 * amp * np.sin(2*np.pi*fr*t + rng.uniform(0, 2*np.pi))
        s += 0.45 * amp * np.sin(2*np.pi*(2*fr)*t + rng.uniform(0, 2*np.pi))
        s += 0.20 * amp * np.sin(2*np.pi*(3*fr)*t + rng.uniform(0, 2*np.pi))

        # weak AM
        am_depth = 0.06 * (health[i] + 0.1)
        s *= (1.0 + am_depth * np.sin(2*np.pi*0.5*t + rng.uniform(0, 2*np.pi)))

        # light BPF injection (bearing-like; scaled with health & late stage)
        f_bpf = 1.5 * fr
        bpf_gain = 0.05 * (0.5*health[i] + 0.5*(stage_gain[i]-0.9)/0.4)
        s += bpf_gain * amp * np.sin(2*np.pi*f_bpf*t + rng.uniform(0, 2*np.pi))
        s += 0.5*bpf_gain * amp * np.sin(2*np.pi*(2*f_bpf)*t + rng.uniform(0, 2*np.pi))

        # noise + impulses
        s += noise_std[i] * rng.standard_normal(N)
        if rng.random() < p_imp[i]:
            k_spike = rng.integers(3, 8)
            idxs = rng.integers(0, N, size=k_spike)
            s[idxs] += rng.uniform(1.5, 3.0) * amp

        # window & features
        sw = s * w
        rms = float(np.sqrt(np.mean(sw**2)))
        mu = float(sw.mean())
        std = float(sw.std() + 1e-12)
        kurt = float(np.mean(((sw - mu) / std)**4))
        crest = float(np.max(np.abs(sw)) / (rms + 1e-12))

        X = np.fft.rfft(sw)
        P = (np.abs(X)**2) / N
        P = np.maximum(P, 1e-16)
        p = P / P.sum()
        H = float(-(p * np.log(p)).sum() / np.log(len(P)))

        # load-normalized RMS (부하 영향 분리)
        denom = 0.25 + 0.75 * float(op3[i])  # [0.25, 1.0]
        rms_norm = float(rms / denom)

        out.at[idx, "fe_RMS"] = rms
        out.at[idx, "fe_RMS_norm"] = rms_norm
        out.at[idx, "fe_Kurtosis"] = kurt
        out.at[idx, "fe_Crest"] = crest
        out.at[idx, "fe_Entropy"] = H

    return out.reset_index(drop=True)

# ----------------------- 저장 -----------------------
def _dump_matrix(df: pd.DataFrame, path: Path):
    arr = df[COLS].values; np.savetxt(path, arr, fmt="%.6f", delimiter=" ")

def save_cmapss_like(df_tr:pd.DataFrame, df_te:pd.DataFrame, rul_vec:List[int],
                     out_train:Path, out_test:Path, out_rul:Path, run_index:int):
    train_name=f"train_{run_index}.txt"; test_name=f"test_{run_index}.txt"; rul_name=f"RUL_{run_index}.txt"
    out_train.mkdir(parents=True, exist_ok=True); out_test.mkdir(parents=True, exist_ok=True); out_rul.mkdir(parents=True, exist_ok=True)
    _dump_matrix(df_tr, out_train/train_name); _dump_matrix(df_te, out_test/test_name)
    np.savetxt(out_rul/rul_name, np.array(rul_vec, dtype=int), fmt="%d", delimiter="\n")

def sha1(path: Path) -> str:
    h=hashlib.sha1()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()

# ----------------------- 시뮬레이션 -----------------------
def simulate_unit(uid:int, cfg:DTConfig, rng:np.random.Generator) -> Tuple[pd.DataFrame, Dict]:
    T = rng.integers(cfg.min_cycles, cfg.max_cycles+1)
    op = sample_ops(T, rng)
    alpha={"fan":float(rng.uniform(0.0007,0.0011)), "comp":float(rng.uniform(0.0010,0.0015)),
           "turb":float(rng.uniform(0.0012,0.0018)), "brg":float(rng.uniform(0.0009,0.0013))}
    for k in rng.choice(list(alpha.keys()), size=2, replace=False): alpha[k]*=rng.uniform(1.2,1.6)
    h=degrade_health(op, alpha, cfg, rng)
    fail_t=hazard_fail_time(op, h, cfg, rng)
    censored=False

    if fail_t is None:
        extra=0
        while fail_t is None and extra<1500:
            op = np.vstack([op, sample_ops(1, rng)])
            h = degrade_health(op, alpha, cfg, rng)
            fail_t = hazard_fail_time(op, h, cfg, rng)
            extra += 1
        if fail_t is None:
            censored=True; fail_t = op.shape[0] + 1
    T=min(op.shape[0], fail_t); op=op[:T]
    for k in ("fan","comp","turb","brg","surge","lube","crack"): h[k]=h[k][:T]

    X=gen_sensors(op, h, rng, cfg)
    df=pd.DataFrame({"unit":uid, "cycle":np.arange(1,T+1), "op1":op[:,0], "op2":op[:,1], "op3":op[:,2]})
    for i in range(21): df[f"sensor{i+1}"]=X[:,i]
    df["h_fan"]=h["fan"]; df["h_comp"]=h["comp"]; df["h_turb"]=h["turb"]; df["h_brg"]=h["brg"]

    meta={"censored":bool(censored), "fail_cycle": None if censored else int(T)}
    return df, meta

# ----------------------- Advanced vibration (v3, P1 option) -----------------------
def synthesize_vibration_features_adv(df_with_rul: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    fs = 4096.0
    N = 2048
    t = np.arange(N) / fs
    w = _hann(N, 0.5).astype(np.float32)

    op3 = _safe_series(df_with_rul, "op3", 0.5)
    s8  = _safe_series(df_with_rul, "sensor8", 0.5)
    rotor = _rotor_proxy(op3, s8)

    if "cycle" in df_with_rul.columns and "unit" in df_with_rul.columns:
        max_c_by_u = df_with_rul.groupby("unit")["cycle"].transform("max").to_numpy(float)
        progress = (df_with_rul["cycle"].to_numpy(float) / (max_c_by_u + 1e-9)).clip(0.0, 1.0)
    else:
        progress = np.linspace(0.0, 1.0, len(df_with_rul))

    damage = np.clip(0.15*progress + 0.85*progress**2, 0.0, 1.0)

    A0, a1, a2 = 0.12, 0.10, 0.08
    rotor_norm = ((rotor - 10.0) / (500.0 - 10.0)).clip(0.0, 1.0)
    A = (A0 + a1*op3 + a2*rotor_norm + 0.25*damage) * (0.9 + 0.5*damage)
    A = np.clip(A, 0.05, 0.70)
    noise_std = np.clip(0.10 + 0.10*op3 + 0.10*damage, 0.05, 0.35)

    cols = ["unit","cycle"]
    if "RUL" in df_with_rul.columns: cols.append("RUL")
    cols += ["fe_RMS","fe_RMS_norm","fe_Kurtosis","fe_Crest","fe_Entropy","fe_env_BPF","fe_env_2BPF"]
    out = pd.DataFrame(index=df_with_rul.index, columns=cols, dtype=float)
    out[["unit","cycle"]] = df_with_rul[["unit","cycle"]].values
    if "RUL" in df_with_rul.columns:
        out["RUL"] = df_with_rul["RUL"].values

    for i, idx in enumerate(df_with_rul.index):
        fr = rotor[i]
        amp = A[i]

        s  = 1.00 * amp * np.sin(2*np.pi*fr*t + rng.uniform(0, 2*np.pi))
        s += 0.45 * amp * np.sin(2*np.pi*(2*fr)*t + rng.uniform(0, 2*np.pi))
        s += 0.20 * amp * np.sin(2*np.pi*(3*fr)*t + rng.uniform(0, 2*np.pi))
        s += noise_std[i] * rng.standard_normal(N)

        f_bpf = 1.5 * fr
        s += 0.05*amp*(1+damage[i]) * np.sin(2*np.pi*f_bpf*t + rng.uniform(0, 2*np.pi))
        s += 0.03*amp*(1+damage[i]) * np.sin(2*np.pi*(2*f_bpf)*t + rng.uniform(0, 2*np.pi))

        sw = s * w
        rms = float(np.sqrt(np.mean(sw**2)))
        mu = float(sw.mean()); std = float(sw.std() + 1e-12)
        kurt = float(np.mean(((sw - mu)/std)**4))
        crest = float(np.max(np.abs(sw)) / (rms + 1e-12))

        X = np.fft.rfft(sw); freqs = np.fft.rfftfreq(N, d=1.0/fs)
        P = (np.abs(X)**2) / N; P = np.maximum(P, 1e-16)

        def band_power(center_hz: float) -> float:
            lo = center_hz * 0.97; hi = center_hz * 1.03
            mask = (freqs >= lo) & (freqs <= hi)
            return float(P[mask].sum())

        env_bpf  = band_power(f_bpf)
        env_2bpf = band_power(2.0 * f_bpf)

        denom = 0.25 + 0.75 * float(op3[i])
        rms_norm = float(rms / denom)

        out.at[idx,"fe_RMS"] = rms
        out.at[idx,"fe_RMS_norm"] = rms_norm
        out.at[idx,"fe_Kurtosis"] = kurt
        out.at[idx,"fe_Crest"] = crest
        out.at[idx,"fe_Entropy"] = float(-(P/P.sum()).dot(np.log(P/P.sum())) / np.log(len(P)))
        out.at[idx,"fe_env_BPF"] = env_bpf
        out.at[idx,"fe_env_2BPF"] = env_2bpf

    return out.reset_index(drop=True)

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=200)
    ap.add_argument("--min-cycles", type=int, default=150)
    ap.add_argument("--max-cycles", type=int, default=380)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--set-id", type=str, default=None)

    ap.add_argument("--calib-mode", type=str, default=DEFAULT_CALIB_MODE, choices=["none","moments","cov"])
    ap.add_argument("--cov-blend-lam", type=float, default=DEFAULT_COV_LAM)

    # path overrides
    ap.add_argument("--fd004-train", type=str, default=str(DEFAULT_FD004_TRAIN))
    ap.add_argument("--fd004-rul", type=str, default=str(DEFAULT_FD004_RUL))
    ap.add_argument("--train-root", type=str, default=str(DEFAULT_TRAIN_ROOT))
    ap.add_argument("--test-root",  type=str, default=str(DEFAULT_TEST_ROOT))
    ap.add_argument("--rul-root",   type=str, default=str(DEFAULT_RUL_ROOT))

    ap.add_argument("--no-vib", action="store_true", help="진동 피처 저장 끔")

    args = ap.parse_args()

    cfg = DTConfig(
        units=args.units,
        min_cycles=args.min_cycles,
        max_cycles=args.max_cycles,
        seed=args.seed,
        set_id=args.set_id or f"data{args.units}",
        calib_mode=args.calib_mode,
        cov_blend_lam=args.cov_blend_lam,
        fd004_train_path=Path(args.fd004_train),
        fd004_rul_path=Path(args.fd004_rul),
        train_root=Path(args.train_root),
        test_root=Path(args.test_root),
        rul_root=Path(args.rul_root),
        save_vib=not args.no_vib,
    )

    rng = np.random.default_rng(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)

    # FD004 참조 통계 & RUL
    mu_ref, cov_ref, rul_ref = load_ref_stats_and_rul(cfg)
    print(f"[CAL] FD004 stats loaded: mu={mu_ref.shape}, cov={cov_ref.shape}, RUL_len={len(rul_ref)} | mode='{cfg.calib_mode}', λ={cfg.cov_blend_lam}")

    # 시뮬레이션
    train_dir, test_dir, rul_dir = ensure_common_version_dirs(cfg)
    L1=lock(train_dir); L2=lock(test_dir); L3=lock(rul_dir)
    try:
        rows=[]; metas={}
        for uid in range(1, cfg.units+1):
            df_u, meta = simulate_unit(uid, cfg, rng)
            rows.append(df_u); metas[uid]=meta
        df_all=pd.concat(rows, ignore_index=True)

        # split
        units = np.array(sorted(metas.keys())); rng.shuffle(units)
        n_tr = int(len(units) * cfg.train_split)
        tr_units = set(units[:n_tr]); te_units_all = list(units[n_tr:])
        df_tr = df_all[df_all.unit.isin(tr_units)].copy()
        df_te = df_all[df_all.unit.isin(te_units_all)].copy()

        # train censored 제거
        censored_units = {u for u, m in metas.items() if m["fail_cycle"] is None}
        df_tr = df_tr[~df_tr["unit"].isin(censored_units)].copy()

        # ====== 보정 적용 (moments / cov) ======
        if cfg.calib_mode == "moments":
            apply_moments_inplace(df_tr, df_te, mu_ref, cov_ref)
        elif cfg.calib_mode == "cov":
            warp = fit_cov_warp(df_tr[SENSOR_COLS].to_numpy(), mu_ref, cov_ref, cfg.cov_blend_lam)
            if warp is not None:
                df_tr[SENSOR_COLS] = warp(df_tr[SENSOR_COLS].to_numpy())
                df_te[SENSOR_COLS] = warp(df_te[SENSOR_COLS].to_numpy())
        # --------- M2: 문제 센서 μ/σ 보정(단일 패스) ----------
        problem_sensors = ["sensor1","sensor3","sensor4","sensor8","sensor9","sensor13","sensor18"]
        sd_ref = np.sqrt(np.clip(np.diag(cov_ref), 1e-12, None))
        mu_tr_now = df_tr[SENSOR_COLS].mean().to_numpy()
        sd_tr_now = df_tr[SENSOR_COLS].std(ddof=0).to_numpy() + 1e-12
        for s in problem_sensors:
            k = int(s.replace("sensor","")) - 1
            a = (sd_ref[k] / sd_tr_now[k])
            b = (mu_ref[k] - a * mu_tr_now[k])
            for df_ in (df_tr, df_te):
                df_[s] = a * df_[s] + b
        # =====================================================

        # TEST 트렁케이션 & RUL (FD004 RUL 분포에서 샘플링)
        df_te_trunc_rows=[]; rul_vec=[]; te_units=[]
        rng_rul = np.random.default_rng(cfg.seed + 12345)
        fd_rul_vals = rul_ref.astype(int)

        # --------- M1: RUL 샘플링 편향 완화(클램프 + 가중 + 재시도 강화) ----------
        for u in sorted(te_units_all):
            fc = metas[u]["fail_cycle"]
            if fc is None:
                continue
            g = df_te[df_te["unit"] == u].sort_values("cycle").copy()

            chosen = None
            max_cut_rul = max(1, fc - (cfg.burn_in + 1))
            weights = None
            try:
                ranks = np.argsort(np.argsort(fd_rul_vals))
                weights = np.sqrt((ranks + 1) / len(fd_rul_vals))
                weights = weights / weights.sum()
            except Exception:
                pass

            for _ in range(80):
                sample_rul = int(rng_rul.choice(fd_rul_vals, p=weights) if weights is not None
                                 else rng_rul.choice(fd_rul_vals))
                sample_rul = int(min(sample_rul, max_cut_rul))
                cut = fc - sample_rul
                if cut > cfg.burn_in and cut < fc:
                    chosen = sample_rul
                    break

            if chosen is None:
                if fc - 1 <= cfg.burn_in:
                    continue
                chosen = int(rng_rul.integers(1, max(2, fc - cfg.burn_in)))
                cut = fc - chosen

            g = g[g["cycle"] <= cut]
            if len(g) < cfg.burn_in:
                continue
            rul = int(fc - int(g["cycle"].max()))
            if rul <= 0: rul = 1
            df_te_trunc_rows.append(g); rul_vec.append(rul); te_units.append(u)
        # ------------------------------------------------------------------------

        if not df_te_trunc_rows:
            raise RuntimeError("No valid truncated test units. Increase max_cycles or relax settings.")
        df_te = pd.concat(df_te_trunc_rows, ignore_index=True)

        # SAVE
        run_idx=next_run_index(train_dir,test_dir,rul_dir)
        save_cmapss_like(df_tr[["unit","cycle"]+COLS[2:]],
                         df_te[["unit","cycle"]+COLS[2:]],
                         rul_vec, train_dir, test_dir, rul_dir, run_index=run_idx)

        # vib (간단)
        if cfg.save_vib:
            df_tr_rul=df_tr.copy()
            df_tr_rul["max_cycle"]=df_tr_rul.groupby("unit")["cycle"].transform("max")
            df_tr_rul["RUL"]=(df_tr_rul["max_cycle"]-df_tr_rul["cycle"]).astype(int)
            df_tr_rul.drop(columns=["max_cycle"], inplace=True)
            unit2rem={u:r for u,r in zip(te_units, rul_vec)}
            df_te_rul=df_te.copy()
            last_cycle=df_te_rul.groupby("unit")["cycle"].transform("max")
            fail_cycle=last_cycle + df_te_rul["unit"].map(unit2rem).astype(int)
            df_te_rul["RUL"]=(fail_cycle - df_te_rul["cycle"]).astype(int)

            rng2 = np.random.default_rng(cfg.seed+999)
            vib_tr=synthesize_vibration_features(df_tr_rul, rng2)
            vib_te=synthesize_vibration_features(df_te_rul, rng2)
            (train_dir / f"train_{run_idx}_vib.csv").write_text(vib_tr.to_csv(index=False), encoding="utf-8")
            (test_dir  / f"test_{run_idx}_vib.csv").write_text(vib_te.to_csv(index=False), encoding="utf-8")

        # manifest
        pairs=pd.DataFrame(list({u:r for u,r in zip(te_units, rul_vec)}.items()), columns=["unit","RUL"]).sort_values("unit")
        pairs_csv=train_dir / f"test_rul_pairs_{run_idx}.csv"; pairs.to_csv(pairs_csv, index=False, encoding="utf-8")
        train_p=train_dir/f"train_{run_idx}.txt"; test_p=test_dir/f"test_{run_idx}.txt"; rul_p=rul_dir/f"RUL_{run_idx}.txt"
        manifest={
            "set_id": cfg.set_id, "version_dir": train_dir.name, "run_index": run_idx,
            "units_total": int(cfg.units), "train_units": int(df_tr['unit'].nunique()), "test_units": int(len(set(te_units))),
            "seed": int(cfg.seed), "min_cycles": int(cfg.min_cycles), "max_cycles": int(cfg.max_cycles),
            "calib_mode": cfg.calib_mode, "cov_blend_lam": cfg.cov_blend_lam,
            "fd004_train_path": str(cfg.fd004_train_path), "fd004_rul_path": str(cfg.fd004_rul_path),
            "files": {
                "train": {"path": str(train_p), "sha1": sha1(train_p)},
                "test":  {"path": str(test_p),  "sha1": sha1(test_p)},
                "RUL":   {"path": str(rul_p),   "sha1": sha1(rul_p)},
                "pairs_csv": {"path": str(pairs_csv), "sha1": sha1(pairs_csv)},
                "train_vib":   {"path": str(train_dir / f"train_{run_idx}_vib.csv")},
                "test_vib":    {"path": str(test_dir  / f"test_{run_idx}_vib.csv")}
            },
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        (train_dir / f"manifest_{run_idx}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[OK] Generated set_id={cfg.set_id}")
        print(f"  version -> {train_dir}")
        print(f"  run     -> {run_idx}")
        print(f"  train   -> {train_p}")
        print(f"  test    -> {test_p}")
        print(f"  RUL     -> {rul_p}")
        if cfg.save_vib:
            print(f"  vib     -> {train_dir / f'train_{run_idx}_vib.csv'} / {test_dir / f'test_{run_idx}_vib.csv'}")
    finally:
        unlock(L1); unlock(L2); unlock(L3)

if __name__ == "__main__":
    main()
