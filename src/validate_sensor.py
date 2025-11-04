#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Audit — GEN vs FD004 (train/test/RUL)
=============================================
개선 사항
- outdir 기본값을 다음으로 고정: C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\audit_out
- 경로 지정 간편화: (--set-id, --version, --run)로 자동 추적 가능
  * 기본 트리: data\training\<set_id>\vN\train_<run>.txt 등
  * --run 미지정 또는 'auto'면 해당 vN에서 가장 최신 run 자동 선택
- 기존 지표 + KS p-value 추가
- 요약 텍스트/JSON/CSV 자동 저장

빠른 실행 예 (최신 run 자동):
  python validate_sensor.py --set-id data200 --version v3

직접 파일 지정:
  python validate_sensor.py --gen-train "...train_3.txt" --gen-test "...test_3.txt" --gen-rul "...RUL_3.txt"

FD004 경로는 기본값이 들어있지만, --fd-train/--fd-test/--fd-rul 로 변경 가능.
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp

# ---------------- 기본 상수 ----------------
DEFAULT_OUTDIR = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\audit_out")

BASE_TRAIN = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\training")
BASE_TEST  = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\test")
BASE_RUL   = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\rul")

FD_TRAIN_DEF = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\cmapss_raw\train_FD004.txt")
FD_TEST_DEF  = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\cmapss_raw\test_FD004.txt")
FD_RUL_DEF   = Path(r"C:\Users\sjmoo\OneDrive\바탕 화면\turbo jet engine\data\cmapss_raw\RUL_FD004.txt")

COLS = ['unit','cycle','op1','op2','op3'] + [f"sensor{i}" for i in range(1,22)]
SENSOR_COLS = [f"sensor{i}" for i in range(1,22)]
_run_pat = re.compile(r'^(train|test|RUL)_(\d+)\.txt$', re.IGNORECASE)

# ---------------- Utils ----------------
def read_cmapss_like(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df.dropna(axis=1, how="all", inplace=True)
    if df.shape[1] != len(COLS):
        raise ValueError(f"[{path.name}] expected {len(COLS)} columns, got {df.shape[1]}")
    df.columns = COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df

def read_rul(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    arr = np.loadtxt(path, dtype=int)
    if arr.ndim == 0:
        arr = np.array([int(arr)], dtype=int)
    return arr

def upper_tri(A: np.ndarray) -> np.ndarray:
    m = A.shape[0]
    idx = np.triu_indices(m, k=1)
    return A[idx]

def symmetric_kl_from_hist(a: np.ndarray, b: np.ndarray, bins: int=50, eps: float=1e-12) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a)==0 or len(b)==0:
        return float("nan")
    mn = float(min(a.min(initial=0), b.min(initial=0)))
    mx = float(max(a.max(initial=1), b.max(initial=1)))
    if mx <= mn:
        return 0.0
    edges = np.linspace(mn, mx, bins+1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    pa = np.clip(pa, eps, None); pb = np.clip(pb, eps, None)
    pa = pa/pa.sum(); pb = pb/pb.sum()
    return float(np.sum(pa*np.log(pa/pb) + pb*np.log(pb/pa)))

def continuity_warnings(df: pd.DataFrame) -> List[dict]:
    warn = []
    for u, g in df.groupby("unit"):
        cyc = g["cycle"].values
        if not np.all(np.diff(cyc) >= 1):
            warn.append({"unit": int(u), "issue": "non-monotonic or duplicate cycles"})
        if len(cyc) >= 2 and not np.all(np.diff(cyc) == 1):
            warn.append({"unit": int(u), "issue": "cycle gaps exist"})
    return warn

def corr_similarity(A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    Cg = np.corrcoef(A, rowvar=False)
    Cf = np.corrcoef(B, rowvar=False)
    ut_g = upper_tri(Cg); ut_f = upper_tri(Cf)
    mse = float(np.nanmean((ut_g - ut_f)**2))
    fro = float(np.linalg.norm(Cg - Cf, ord="fro"))
    return mse, fro

def sensor_similarity(gen: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SENSOR_COLS:
        g = gen[s].values.astype(float)
        f = ref[s].values.astype(float)
        wd  = wasserstein_distance(g[~np.isnan(g)], f[~np.isnan(f)])
        skl = symmetric_kl_from_hist(g, f, bins=50)
        mean_diff = float(np.nanmean(g) - np.nanmean(f))
        std_ratio = float(np.nanstd(g) / (np.nanstd(f) + 1e-12))
        # KS test (정규화/스케일 불변 아님, 참고용)
        g_c = g[~np.isnan(g)]; f_c = f[~np.isnan(f)]
        ks_p = float(ks_2samp(g_c, f_c).pvalue) if (len(g_c)>5 and len(f_c)>5) else float('nan')
        rows.append({"sensor": s, "wasserstein": wd, "sym_kl": skl,
                     "mean_diff": mean_diff, "std_ratio": std_ratio, "ks_p": ks_p})
    return pd.DataFrame(rows).sort_values("sensor")

def rul_similarity(gen_rul: np.ndarray, ref_rul: np.ndarray) -> dict:
    g = gen_rul.astype(float); f = ref_rul.astype(float)
    wd  = wasserstein_distance(g, f)
    skl = symmetric_kl_from_hist(g, f, bins=50)
    # KS
    ks = ks_2samp(g, f)
    return {
        "gen_len": int(len(g)), "ref_len": int(len(f)),
        "gen_pos_all": bool(np.all(g > 0)), "ref_pos_all": bool(np.all(f > 0)),
        "wd": float(wd), "sym_kl": float(skl),
        "ks_p": float(ks.pvalue),
        "gen_mean": float(np.mean(g)), "gen_std": float(np.std(g)),
        "gen_min": float(np.min(g)),  "gen_max": float(np.max(g)),
        "ref_mean": float(np.mean(f)), "ref_std": float(np.std(f)),
        "ref_min": float(np.min(f)),  "ref_max": float(np.max(f)),
    }

def to_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def resolve_gen_paths(set_id: str, version: str, run: Optional[str]) -> Tuple[Path, Path, Path, int]:
    tdir = BASE_TRAIN / set_id / version
    vdir_test = BASE_TEST / set_id / version
    vdir_rul  = BASE_RUL  / set_id / version
    if not tdir.exists():
        raise FileNotFoundError(f"Train version dir not found: {tdir}")
    # run 자동 탐색
    cand = []
    for p in tdir.glob("train_*.txt"):
        m = _run_pat.match(p.name)
        if m: cand.append(int(m.group(2)))
    if not cand:
        raise FileNotFoundError(f"No train_*.txt in {tdir}")
    if run is None or str(run).lower() == "auto" or str(run).strip() == "":
        run_idx = max(cand)
    else:
        run_idx = int(run)
    tr = tdir / f"train_{run_idx}.txt"
    te = vdir_test / f"test_{run_idx}.txt"
    ru = vdir_rul  / f"RUL_{run_idx}.txt"
    for p in (tr, te, ru):
        if not p.exists():
            raise FileNotFoundError(str(p))
    return tr, te, ru, run_idx

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # 모드1: set-id/version/run로 자동 경로
    ap.add_argument("--set-id", type=str, default=None)
    ap.add_argument("--version", type=str, default=None, help="예: v3")
    ap.add_argument("--run", type=str, default="auto", help="'auto'면 최신 run 자동 선택")

    # 모드2: 직접 지정
    ap.add_argument("--gen-train", type=str, default=None)
    ap.add_argument("--gen-test",  type=str, default=None)
    ap.add_argument("--gen-rul",   type=str, default=None)

    # FD004 경로
    ap.add_argument("--fd-train", type=str, default=str(FD_TRAIN_DEF))
    ap.add_argument("--fd-test",  type=str, default=str(FD_TEST_DEF))
    ap.add_argument("--fd-rul",   type=str, default=str(FD_RUL_DEF))

    ap.add_argument("--outdir",   type=str, default=str(DEFAULT_OUTDIR))

    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # GEN 경로 해석
    if args.gen_train and args.gen_test and args.gen_rul:
        gen_tr = Path(args.gen_train); gen_te = Path(args.gen_test); gen_ru = Path(args.gen_rul)
        set_id = args.set_id or "manual"
        version = args.version or "manual"
        run_idx = -1
    else:
        if not (args.set_id and args.version):
            raise ValueError("Either provide --gen-train/--gen-test/--gen-rul OR provide --set-id and --version")
        gen_tr, gen_te, gen_ru, run_idx = resolve_gen_paths(args.set_id, args.version, args.run)
        set_id = args.set_id; version = args.version

    # FD004 로드
    fd_tr  = read_cmapss_like(Path(args.fd_train))
    fd_te  = read_cmapss_like(Path(args.fd_test))
    fd_ru  = read_rul(Path(args.fd_rul))

    # GEN 로드
    g_tr = read_cmapss_like(gen_tr)
    g_te = read_cmapss_like(gen_te)
    g_ru = read_rul(gen_ru)

    # --- Validity for GEN (독립) ---
    validity = {}
    for tag, df in [("train", g_tr), ("test", g_te)]:
        validity[tag] = {
            "columns_ok": list(df.columns) == COLS,
            "units": int(df["unit"].nunique()),
            "rows": int(len(df)),
        }
        warn = continuity_warnings(df.sort_values(["unit","cycle"]))
        validity[tag]["continuity_warnings"] = warn
        validity[tag]["continuity_warning_count"] = len(warn)

    rul_ok = {"len_gt_zero": int(len(g_ru))>0, "all_positive": bool(np.all(g_ru>0))}
    validity["rul"] = rul_ok

    # --- Similarity ---
    train_sensor_df = sensor_similarity(g_tr, fd_tr)
    test_sensor_df  = sensor_similarity(g_te, fd_te)
    train_corr_mse, train_corr_fro = corr_similarity(g_tr[SENSOR_COLS].values,
                                                     fd_tr[SENSOR_COLS].values)
    test_corr_mse, test_corr_fro   = corr_similarity(g_te[SENSOR_COLS].values,
                                                     fd_te[SENSOR_COLS].values)
    rul_sim = rul_similarity(g_ru, fd_ru)

    # Save outputs (세부 폴더)
    tag = f"{set_id}_{version}_run{run_idx}" if run_idx != -1 else f"{set_id}_{version}"
    sub = outdir / tag
    sub.mkdir(parents=True, exist_ok=True)

    train_sensor_df.to_csv(sub / "train_similarity_sensor.csv", index=False, encoding="utf-8")
    test_sensor_df.to_csv(sub / "test_similarity_sensor.csv", index=False, encoding="utf-8")

    summary = {
        "paths": {
            "gen_train": str(gen_tr),
            "gen_test":  str(gen_te),
            "gen_rul":   str(gen_ru),
            "fd_train":  str(Path(args.fd_train)),
            "fd_test":   str(Path(args.fd_test)),
            "fd_rul":    str(Path(args.fd_rul)),
        },
        "validity_gen": validity,
        "train_global": {"corr_mse_upper": train_corr_mse, "corr_frobenius": train_corr_fro},
        "test_global":  {"corr_mse_upper": test_corr_mse,  "corr_frobenius": test_corr_fro},
        "rul_similarity": rul_sim
    }
    rul_mu_diff = float(rul_sim["gen_mean"] - rul_sim["ref_mean"])
    rul_sd_ratio = float(rul_sim["gen_std"] / (rul_sim["ref_std"] + 1e-12))
    summary["rul_deltas"] = {"mean_diff": rul_mu_diff, "std_ratio": rul_sd_ratio}

    (sub / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable text
    lines = []
    lines.append("=== GEN vs FD004 Audit — PLUS ===")
    lines.append(f"GEN train : {gen_tr}")
    lines.append(f"GEN test  : {gen_te}")
    lines.append(f"GEN RUL   : {gen_ru}")
    lines.append(f"FD004 train: {args.fd_train}")
    lines.append(f"FD004 test : {args.fd_test}")
    lines.append(f"FD004 RUL  : {args.fd_rul}")
    lines.append("")
    lines.append("[Validity — GEN only]")
    lines.append(f"train units={validity['train']['units']}, rows={validity['train']['rows']}, cols_ok={validity['train']['columns_ok']}, continuity_warns={validity['train']['continuity_warning_count']}")
    lines.append(f"test  units={validity['test']['units']}, rows={validity['test']['rows']}, cols_ok={validity['test']['columns_ok']}, continuity_warns={validity['test']['continuity_warning_count']}")
    lines.append(f"rul   len>0={rul_ok['len_gt_zero']}, all_positive={rul_ok['all_positive']}")
    lines.append("")
    lines.append("[Similarity — TRAIN]")
    lines.append(f"corr_mse_upper={train_corr_mse}, corr_frobenius={train_corr_fro}")
    lines.append("per-sensor metrics -> train_similarity_sensor.csv (wasserstein / sym_kl / mean_diff / std_ratio / ks_p)")
    lines.append("")
    lines.append("[Similarity — TEST]")
    lines.append(f"corr_mse_upper={test_corr_mse}, corr_frobenius={test_corr_fro}")
    lines.append("per-sensor metrics -> test_similarity_sensor.csv (wasserstein / sym_kl / mean_diff / std_ratio / ks_p)")
    lines.append("")
    lines.append("[Similarity — RUL]")
    for k, v in summary["rul_similarity"].items():
        lines.append(f"{k}: {v}")

    (sub / "readable_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Audit complete")
    print(f" - Folder : {sub}")
    print(f" - train per-sensor: {sub / 'train_similarity_sensor.csv'}")
    print(f" - test  per-sensor: {sub / 'test_similarity_sensor.csv'}")
    print(f" - summary JSON    : {sub / 'audit_summary.json'}")
    print(f" - text report     : {sub / 'readable_report.txt'}")

if __name__ == "__main__":
    main()