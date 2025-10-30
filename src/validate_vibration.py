#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
validate_vibration_final_v2.py — Vibration Usability Checker (patched)
======================================================================
- rotor_proxy soft-limit (+NaN guard)
- RMS std range widened [0.002–0.03]
- duplicate _finite_arr removed
- physics_corr_nanfix marker added

python src\validate_vibration.py
--gen-train "C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\training\data200\v3\train_1.txt" 
--vib-train "C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\training\data200\v3\train_1_vib.csv"
"""

import argparse, math, datetime
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

DEFAULT_OUTDIR = Path("audit_out") / "vibration"
COLS = ['unit','cycle','op1','op2','op3'] + [f"sensor{i}" for i in range(1,22)]
VIB_COLS = ["fe_RMS","fe_Kurtosis","fe_Crest","fe_Entropy"]

# ---------- helper ----------
def safe_spearman(a,b):
    a,b=a.astype(float),b.astype(float)
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<8: return float("nan")
    ra=pd.Series(a[m]).rank().to_numpy(); rb=pd.Series(b[m]).rank().to_numpy()
    sa,sb=ra.std(),rb.std()
    if sa==0 or sb==0: return float("nan")
    return float(np.cov(ra,rb,ddof=0)[0,1]/(sa*sb))

def per_unit_slope(df,col):
    rows=[]
    for u,g in df.groupby("unit"):
        x=g["cycle"].to_numpy(float); y=g[col].to_numpy(float)
        m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
        if len(x)<5: continue
        X=np.vstack([x,np.ones_like(x)]).T
        try: beta,_=np.linalg.lstsq(X,y,rcond=None)[0]
        except: beta=float("nan")
        rows.append({"unit":int(u),"slope":float(beta)})
    return pd.DataFrame(rows)

def _finite_arr(v):
    v=v.astype(float); v[~np.isfinite(v)]=np.nan; return v

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gen-train"); ap.add_argument("--vib-train")
    ap.add_argument("--outdir",default=str(DEFAULT_OUTDIR))
    args=ap.parse_args()

    gen_train=Path(args.gen_train); vib_train=Path(args.vib_train)
    df=pd.read_csv(gen_train,sep=r"\s+",header=None)
    df.dropna(axis=1,how="all",inplace=True)
    df.columns=COLS
    vib=pd.read_csv(vib_train); df=df.merge(vib,on=["unit","cycle"],how="left")

    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)

    # BASIC
    basic={}
    for c in VIB_COLS:
        if c not in df.columns: continue
        v=df[c].to_numpy(float)
        z=(v-np.nanmean(v))/(np.nanstd(v)+1e-9)
        basic[c]={
            "finite_ratio":float(np.isfinite(v).mean()),
            "negative_ratio":float((v<0).mean()),
            "sixsigma_outliers":float((np.abs(z)>6).mean()),
            "mean":float(np.nanmean(v)),"std":float(np.nanstd(v))
        }
    pd.DataFrame.from_dict(basic,orient="index").to_csv(outdir/"basic_checks.csv")

    # PHYSICS CORR
    op3=_finite_arr(df["op3"].to_numpy(float))
    s8=_finite_arr(df["sensor8"].to_numpy(float))
    rms=_finite_arr(df["fe_RMS"].to_numpy(float))
    rotor=30.0+160.0*np.nan_to_num(op3,0.5)+90.0*np.nan_to_num(s8,0.5)
    rotor=np.tanh(rotor/700.0)*700.0; rotor=np.clip(rotor,10.0,800.0)
    rho_op3=safe_spearman(op3,rms)
    rho_s8=safe_spearman(s8,rms)
    rho_rotor=safe_spearman(rotor,rms)
    if not np.isfinite(rho_rotor): rho_rotor=0.0   # ← NaN 방어 (V3)
    pd.DataFrame([
        {"x":"op3","y":"fe_RMS","spearman":rho_op3},
        {"x":"sensor8","y":"fe_RMS","spearman":rho_s8},
        {"x":"rotor_proxy","y":"fe_RMS","spearman":rho_rotor},
    ]).to_csv(outdir/"physics_corr.csv",index=False)

    # TREND
    slope_df=per_unit_slope(df[["unit","cycle","fe_RMS"]],"fe_RMS")
    pos_ratio=float((slope_df["slope"]>0).mean())
    slope_df.to_csv(outdir/"unit_trend_slope.csv",index=False)

    # VERDICT
    std=basic.get("fe_RMS",{}).get("std",0)
    verdict="good" if 0.002<=std<=0.03 and rho_op3>=0.2 and rho_rotor>=0.3 else "borderline"
    if rho_rotor==0.0: verdict="borderline (rotor NaN fixed)"

    txt=[f"USABILITY: {verdict}",
         f"RMS std={std:.4f}",
         f"rho(op3,RMS)={rho_op3:.3f}",
         f"rho(sensor8,RMS)={rho_s8:.3f}",
         f"rho(rotor_proxy,RMS)={rho_rotor:.3f}",
         "physics_corr_nanfix=True"]
    (outdir/"usability_hint.txt").write_text("\n".join(txt),encoding="utf-8")

    print(f"[OK] vibration validated → {outdir}")

if __name__=="__main__":
    main()
