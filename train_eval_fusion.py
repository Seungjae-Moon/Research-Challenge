#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_rulnet.py — MAE‑Optimized (TCN+BiGRU+Attention, Quantile/Huber)
=========================================================================
Purpose
  • Fresh, MAE‑first deep model for CMAPSS‑style RUL with optional vib features.
  • Drop‑in compatible with your folder layout (training/test/rul under data/...).
  • Strong short‑sequence handling (last‑row padding), world‑alignment guard,
    quantile(τ=0.5) loss (direct MAE surrogate), EMA, early stopping, AMP.

CLI (examples)
  # Synthetic world (recommended)
  python train_eval_rulnet.py --set-id data200 --version v4 --vib 1 --seq-len 20 \
      --test syn --epochs 40 --model tcn_bigru --loss quantile --quantile 0.5

  # Compare vib on/off quickly (same seed)
  python train_eval_rulnet.py --set-id data200 --version v4 --vib 0 --seq-len 20 --test syn
  python train_eval_rulnet.py --set-id data200 --version v4 --vib 1 --seq-len 20 --test syn

  # FD004 only
  python train_eval_rulnet.py --train-src fd --test fd --vib 0 --seq-len 50 --epochs 30

Outputs
  runs/<tag>/model_best.pt, eval_summary.json, preds_*.csv

Requirements: numpy, pandas, scikit-learn, torch>=1.12
"""
from __future__ import annotations
import argparse, json, os, random, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader

# ------------------------------- Paths & Columns -------------------------------
COLS = ['unit','cycle','op1','op2','op3'] + [f"sensor{i}" for i in range(1,22)]
OPS  = ['op1','op2','op3']
SENS = [f"sensor{i}" for i in range(1,22)]
VIB_COLS = ["fe_RMS","fe_Kurtosis","fe_Crest","fe_Entropy"]

BASE_TRAIN = Path("data")/"training"
BASE_TEST  = Path("data")/"test"
BASE_RUL   = Path("data")/"rul"

FD_TRAIN = Path("data")/"cmapss_raw"/"train_FD004.txt"
FD_TEST  = Path("data")/"cmapss_raw"/"test_FD004.txt"
FD_RUL   = Path("data")/"cmapss_raw"/"RUL_FD004.txt"

RUNS_DIR = Path("runs")
_pat = re.compile(r"^(train|test|RUL)_(\d+)\.txt$")

# --------------------------------- IO Helpers ---------------------------------

def read_cmapss_like(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    df.columns = COLS
    return df.sort_values(["unit","cycle"]).reset_index(drop=True)

def read_rul(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float)
    return arr.reshape(-1)

def compute_rul_for_train(df: pd.DataFrame) -> pd.Series:
    mx = df.groupby('unit')['cycle'].transform('max')
    return (mx - df['cycle']).astype(float)

def resolve_paths(set_id: str, version: str, run: Optional[str]):
    tdir = BASE_TRAIN/set_id/version
    vdir_t = BASE_TEST/set_id/version
    vdir_r = BASE_RUL/set_id/version
    if not tdir.exists():
        raise FileNotFoundError(f"Train version dir not found: {tdir}")
    runs = []
    for p in tdir.glob("train_*.txt"):
        m = _pat.match(p.name)
        if m: runs.append(int(m.group(2)))
    if not runs:
        raise FileNotFoundError(f"No train_*.txt in {tdir}")
    run_idx = max(runs) if (run is None or run.lower()=="auto" or run.strip()=="") else int(run)
    tr = tdir/f"train_{run_idx}.txt"; te = vdir_t/f"test_{run_idx}.txt"; ru = vdir_r/f"RUL_{run_idx}.txt"
    for p in (tr, te, ru):
        if not p.exists():
            raise FileNotFoundError(str(p))
    return tr, te, ru, run_idx, tdir, vdir_t, vdir_r

def try_merge_vib(df: pd.DataFrame, base_dir: Path, run_idx: int, split: str, vib_on: bool) -> pd.DataFrame:
    if not vib_on: return df
    vib_path = base_dir/f"{split}_{run_idx}_vib.csv"
    if not vib_path.exists():
        print(f"[INFO] {vib_path.name} not found → vib ignored for {split}.")
        return df
    vib = pd.read_csv(vib_path)
    expect = set(["unit","cycle"]+VIB_COLS)
    if not expect.issubset(vib.columns):
        print(f"[WARN] {vib_path.name} missing vib cols → ignored.")
        return df
    df = df.merge(vib[['unit','cycle']+VIB_COLS], on=['unit','cycle'], how='left')
    for c in VIB_COLS:
        if c in df.columns:
            df[c] = df[c].astype(float).fillna(df[c].median())
    print(f"[OK] merged vib → {vib_path}")
    return df

# --------------------------------- Datasets -----------------------------------
class SlidingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat_cols: List[str], seq_len: int,
                 scaler: Optional[StandardScaler], fit_scaler: bool=False,
                 rul_clip: Optional[float]=125.0):
        self.seq_len = seq_len
        X = df[feat_cols].to_numpy(float)
        if fit_scaler and scaler is not None:
            scaler.fit(X)
        X = scaler.transform(X) if scaler is not None else X
        dfX = pd.DataFrame(X, columns=feat_cols, index=df.index)
        if 'RUL' not in df.columns:
            df = df.copy(); df['RUL'] = compute_rul_for_train(df)
        y = df['RUL'].astype(float)
        if rul_clip is not None:
            y = np.clip(y, 0.0, float(rul_clip))
            df = df.copy(); df['RUL'] = y
        rowsX, rowsY = [], []
        for _, g in df.groupby('unit'):
            g = g.sort_values('cycle')
            M = len(g)
            if M < seq_len: continue
            Xg = dfX.loc[g.index, feat_cols].to_numpy(float)
            Yg = g['RUL'].to_numpy(float)
            for t in range(seq_len-1, M):
                rowsX.append(Xg[t-seq_len+1:t+1])
                rowsY.append(Yg[t])
        self.X = np.asarray(rowsX, np.float32)
        self.Y = np.asarray(rowsY, np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.Y[i])

class LastWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat_cols: List[str], seq_len: int,
                 scaler: Optional[StandardScaler], labels: Optional[np.ndarray]=None,
                 rul_clip: Optional[float]=125.0):
        self.seq_len = seq_len
        X = df[feat_cols].to_numpy(float)
        X = scaler.transform(X) if scaler is not None else X
        dfX = pd.DataFrame(X, columns=feat_cols, index=df.index)
        ulist = sorted(df['unit'].unique())
        rows, ys, units = [], [], []
        for i, u in enumerate(ulist):
            g = df[df['unit']==u].sort_values('cycle')
            Xg = dfX.loc[g.index, feat_cols].to_numpy(float)
            if len(Xg) < seq_len:
                pad = np.repeat(Xg[-1:], seq_len-len(Xg), axis=0)
                win = np.concatenate([pad, Xg], axis=0)
            else:
                win = Xg[-seq_len:]
            rows.append(win)
            y = -1.0 if labels is None else float(labels[i])
            if rul_clip is not None and y >= 0:
                y = float(np.clip(y, 0.0, rul_clip))
            ys.append(y); units.append(int(u))
        self.X = np.asarray(rows, np.float32)
        self.Y = np.asarray(ys,   np.float32)
        self.units = np.asarray(units, np.int32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.Y[i]), int(self.units[i])

# ---------------------------------- Model -------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, hid: int=128, ks: int=5, dilations=(1,2,4), dropout=0.1):
        super().__init__()
        layers = []
        cin = in_ch
        for d in dilations:
            layers += [
                nn.Conv1d(cin, hid, kernel_size=ks, padding=(ks-1)//2*d, dilation=d),
                nn.ReLU(),
                nn.Conv1d(hid, hid, kernel_size=ks, padding=(ks-1)//2*d, dilation=d),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            cin = hid
        self.net = nn.Sequential(*layers)
        self.res = nn.Conv1d(in_ch, hid, kernel_size=1) if in_ch!=hid else nn.Identity()
        self.act = nn.ReLU()
    def forward(self, x):  # x: (B,F,T)
        out = self.net(x)
        out = self.act(out + self.res(x))
        return out

class AttnPool(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):  # (B,T,H)
        a, _ = self.mha(x, x, x, need_weights=False)
        return self.ln(a + x).mean(dim=1)

class RULNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int=160, layers: int=2, dropout: float=0.2,
                 use_tcn: bool=True, nonneg: bool=True):
        super().__init__()
        self.use_tcn = use_tcn
        if use_tcn:
            self.tcn = TCNBlock(in_ch=input_dim, hid=128, ks=5, dilations=(1,2,4), dropout=0.1)
            tcn_out = 128
        else:
            self.tcn = None
            tcn_out = input_dim
        self.bigru = nn.GRU(input_size=tcn_out, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=(dropout if layers>1 else 0.0), bidirectional=True)
        self.attn = AttnPool(dim=hidden*2, heads=4)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.nonneg = nonneg
        self.softplus = nn.Softplus()
    def forward(self, x):  # x: (B,T,F)
        if self.use_tcn:
            x = x.transpose(1,2)             # (B,F,T)
            x = self.tcn(x)
            x = x.transpose(1,2)             # (B,T,C)
        h, _ = self.bigru(x)
        feat = self.attn(h)
        y = self.head(feat).squeeze(-1)
        if self.nonneg:
            y = self.softplus(y)             # enforce non‑negative RUL
        return y

# ---------------------------------- Losses ------------------------------------
class QuantileLoss(nn.Module):
    def __init__(self, q: float=0.7):
        super().__init__(); self.q = q
    def forward(self, pred, target):
        e = target - pred
        return torch.mean(torch.maximum(self.q*e, (self.q-1)*e))

# --------------------------------- Utilities ----------------------------------
def fix_seed(seed: int=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        # batch could be (x,y) or (x,y,u)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0].to(device)
            y = batch[1].to(device)
        else:
            raise RuntimeError("Unexpected batch format")
        yhat, _ = model(x) if isinstance(model(x), tuple) else (model(x), None)
        preds.append(yhat.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    yhat = np.concatenate(preds, axis=0)
    ytrue = np.concatenate(trues, axis=0)
    err = yhat - ytrue
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    std_abs = float(np.std(np.abs(err)))
    return {"MAE": mae, "RMSE": rmse, "STD_abs": std_abs}

# === [AFFINE CALIBRATION – helpers] ============================================
def _fit_affine_on_loader(model, loader, device):
    """학습 끝난 뒤, 검증 로더에서 y' = a*ŷ + b 를 최소제곱으로 적합."""
    model.eval()
    yh, yt = [], []
    with torch.no_grad():
        for batch in loader:
            # (x, y) 또는 (x, y, u) 모두 지원
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x = batch[0].to(device); y = batch[1]
            else:
                raise RuntimeError("Unexpected batch format for calibration")
            pred = model(x)
            if isinstance(pred, tuple): pred = pred[0]
            yh.append(pred.detach().cpu().numpy()); yt.append(y.numpy())
    yhat = np.concatenate(yh, axis=0).reshape(-1)
    ytrue = np.concatenate(yt, axis=0).reshape(-1)

    A = np.column_stack([yhat, np.ones_like(yhat)])
    sol, *_ = np.linalg.lstsq(A, ytrue, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    return a, b

def _apply_affine(yhat_np, a, b):
    yhat_np = np.asarray(yhat_np).reshape(-1)
    return a * yhat_np + b
# ============================================================================



# Simple EMA wrapper
class EMA:
    def __init__(self, model: nn.Module, decay: float=0.995):
        self.decay = decay
        self.shadow = {k: p.clone().detach() for k,p in model.state_dict().items() if p.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0-self.decay)
    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                p.copy_(self.shadow[k])

# ------------------------------- Main Pipeline --------------------------------

def main():
    from torch.utils.data import DataLoader
    ap = argparse.ArgumentParser()
    ap.add_argument('--set-id', type=str, default='data200')
    ap.add_argument('--version', type=str, default='v4')
    ap.add_argument('--run', type=str, default='auto')

    ap.add_argument('--train-src', type=str, default='gen', choices=['gen','fd'])
    ap.add_argument('--test', type=str, default='syn', choices=['fd','syn','both','none'])
    ap.add_argument('--vib', type=int, default=1, choices=[0,1])

    ap.add_argument('--model', type=str, default='tcn_bigru', choices=['bigru','tcn_bigru'])
    ap.add_argument('--seq-len', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--hidden', type=int, default=160)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--rul-clip', type=float, default=125.0)

    ap.add_argument('--loss', type=str, default='quantile', choices=['mae','huber','quantile'])
    ap.add_argument('--huber-delta', type=float, default=1.0)
    ap.add_argument('--quantile', type=float, default=0.5)

    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--amp', type=int, default=1, choices=[0,1])
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--ema', type=int, default=1, choices=[0,1])
    ap.add_argument('--patience', type=int, default=6)

    ap.add_argument('--eval-only', type=int, default=0, choices=[0,1])
    ap.add_argument('--ckpt', type=str, default=None)

    args = ap.parse_args()
    fix_seed(args.seed)

    tr_syn, te_syn, ru_syn, run_idx, tdir, tedir, rdir = resolve_paths(args.set_id, args.version, args.run)
    tag = f"{args.set_id}_{args.version}_run{run_idx}"
    outdir = RUNS_DIR/tag; outdir.mkdir(parents=True, exist_ok=True)

    # World alignment guard
    if args.test in ('syn','both') and args.train_src=='fd' and not args.eval_only:
        print('\n[WARN] Training on FD004 but evaluating SYN. Expect domain shift and larger MAE.\n')

    # --------------------------- Load TRAIN ---------------------------
    if args.train_src=='gen':
        df_tr = read_cmapss_like(tr_syn); df_tr['RUL'] = compute_rul_for_train(df_tr)
        df_tr = try_merge_vib(df_tr, tdir, run_idx, 'train', vib_on=bool(args.vib==1))
    else:
        df_tr = read_cmapss_like(FD_TRAIN); df_tr['RUL'] = compute_rul_for_train(df_tr)

    feat_cols = OPS + SENS
    if args.vib==1 and set(VIB_COLS).issubset(df_tr.columns):
        feat_cols += VIB_COLS; print(f"[INFO] Using vib: {VIB_COLS}")
    else:
        feat_cols = [c for c in feat_cols if c not in VIB_COLS]
        if args.vib==1: print("[INFO] Vib requested but not present in train → skipping.")

    # Split by units
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    (idx_tr, idx_va), = gss.split(df_tr, groups=df_tr['unit'])
    u_tr = set(df_tr.iloc[idx_tr]['unit'].unique()); u_va=set(df_tr.iloc[idx_va]['unit'].unique())
    df_trn = df_tr[df_tr['unit'].isin(u_tr)].copy(); df_val = df_tr[df_tr['unit'].isin(u_va)].copy()

    scaler = StandardScaler()
    _ = SlidingDataset(df_trn, feat_cols, args.seq_len, scaler, fit_scaler=True, rul_clip=args.rul_clip)

    ds_tr = SlidingDataset(df_trn, feat_cols, args.seq_len, scaler, fit_scaler=False, rul_clip=args.rul_clip)
    ds_va = SlidingDataset(df_val, feat_cols, args.seq_len, scaler, fit_scaler=False, rul_clip=args.rul_clip)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

    # ---------------------------- Build Model ----------------------------
    use_tcn = (args.model=='tcn_bigru')
    model = RULNet(input_dim=len(feat_cols), hidden=args.hidden, layers=args.layers,
                   dropout=args.dropout, use_tcn=use_tcn, nonneg=True).to(args.device)

    if args.loss=='mae':
        crit = nn.L1Loss()
    elif args.loss=='huber':
        crit = nn.SmoothL1Loss(beta=args.huber_delta)
    else:
        crit = QuantileLoss(q=args.quantile)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=bool(args.amp==1))
    ema = EMA(model, decay=0.995) if args.ema==1 else None

    state = {}
    if args.eval_only:
        assert args.ckpt and Path(args.ckpt).exists(), "--eval-only requires --ckpt"
        state = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(state['model'])
        # === [AFFINE CALIBRATION – learn & save from validation] ======================
        try:
            # ds_va / dl_va 가 앞에서 이미 만들어져 있음 (검증 분할)
            from torch.utils.data import DataLoader
            dl_va_cal = DataLoader(ds_va, batch_size=256, shuffle=False)
            a_cal, b_cal = _fit_affine_on_loader(model, dl_va_cal, args.device)

            (outdir / "affine_cal.json").write_text(
                json.dumps({"a": a_cal, "b": b_cal}, indent=2), encoding="utf-8"
            )
            print(f"[CAL] affine learned on validation: a={a_cal:.4f}, b={b_cal:.4f}")
        except Exception as e:
            print(f"[CAL][WARN] calibration learning skipped: {e}")
        # ============================================================================

        scaler.mean_ = np.array(state['scaler_mean']); scaler.scale_ = np.array(state['scaler_scale'])
        feat_ck = state.get('feat_cols', feat_cols)
        if feat_ck!=feat_cols: print('[WARN] Feature list in ckpt differs from current!')
        print('[OK] Checkpoint loaded. Eval‑only mode.')
    else:
        # ----------------------------- TRAIN -----------------------------
        best = float('inf'); wait=0
        for ep in range(1, args.epochs+1):
            model.train(); tr_loss=0.0
            for x,y in dl_tr:
                x=x.to(args.device); y=y.to(args.device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool(args.amp==1)):
                    p = model(x)
                    loss = crit(p, y)
                scaler_amp.scale(loss).backward()
                if args.grad_clip>0:
                    scaler_amp.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler_amp.step(opt); scaler_amp.update()
                if ema: ema.update(model)
                tr_loss += float(loss.detach().cpu().item())
            tr_loss /= max(1,len(dl_tr))
            model.eval();
            with torch.no_grad():
                va_stats = evaluate(model, DataLoader(ds_va, batch_size=256), args.device)
            sched.step(va_stats['MAE'])
            print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_MAE={va_stats['MAE']:.3f} | val_RMSE={va_stats['RMSE']:.3f}")
            metric = va_stats['MAE']
            if metric < best:
                best = metric; wait=0
                ckpt = {
                    'model': model.state_dict(),
                    'scaler_mean': scaler.mean_.tolist(),
                    'scaler_scale': scaler.scale_.tolist(),
                    'feat_cols': feat_cols,
                    'seq_len': args.seq_len,
                    'args': vars(args)
                }
                torch.save(ckpt, outdir/"model_best.pt")
                (outdir/"best_val.json").write_text(json.dumps({'best_val_MAE': best}, indent=2))
            else:
                wait += 1
                if wait >= args.patience:
                    print('[EarlyStopping] patience reached.'); break
        # Optionally apply EMA weights for eval
        if ema:
            ema.apply_to(model)
        state = torch.load(outdir/"model_best.pt", map_location=args.device)
        model.load_state_dict(state['model'])
        
        # === [AFFINE CALIBRATION – learn on validation set] =======================
        try:
            dl_va_cal = DataLoader(ds_va, batch_size=256, shuffle=False)
            a_cal, b_cal = _fit_affine_on_loader(model, dl_va_cal, args.device)
            (outdir / "affine_cal.json").write_text(
                json.dumps({"a": a_cal, "b": b_cal}, indent=2), encoding="utf-8"
            )
            print(f"[CAL] Affine calibration learned: a={a_cal:.4f}, b={b_cal:.4f}")
        except Exception as e:
            print(f"[CAL][WARN] Calibration learning failed: {e}")
        # ============================================================================

    # Ensure scaler from state for deterministic eval
    if 'scaler_mean' in state:
        scaler.mean_ = np.array(state['scaler_mean']); scaler.scale_ = np.array(state['scaler_scale'])

    # ---------------------------- Evaluations ----------------------------
    def dist_debug(df_train: pd.DataFrame, df_test: pd.DataFrame, cols: List[str], tag: str):
        mu_tr = df_train[cols].mean(); sd_tr = df_train[cols].std(ddof=0)+1e-12
        mu_te = df_test[cols].mean();  sd_te = df_test[cols].std(ddof=0)+1e-12
        dmu = (mu_te - mu_tr).abs().sort_values(ascending=False)
        ratio = (sd_te/sd_tr).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        print(f"\n[Dist-debug:{tag}] top-5 |Δμ| features:")
        for c in list(dmu.index[:5]):
            print(f"  {c:12s}  Δμ={float(dmu[c]):8.4f}  σ_ratio={float(ratio[c]):6.3f}")

    eval_syn = args.test in ('syn','both')
    eval_fd  = args.test in ('fd','both')

    # Synthetic eval
    syn_stats=None
    if eval_syn:
        df_te = read_cmapss_like(te_syn)
        df_te = try_merge_vib(df_te, tedir, run_idx, 'test', vib_on=bool(args.vib==1))
        rul = read_rul(ru_syn)
        nU = df_te['unit'].nunique();
        if nU != len(rul):
            raise RuntimeError(f"[ERR] SYN units {nU} != RUL len {len(rul)} (run mismatch)")
        min_len = df_te.groupby('unit')['cycle'].count().min()
        if min_len < args.seq_len:
            print(f"[WARN] SYN: min unit len={min_len} < seq_len={args.seq_len} → last-row padding in effect")
        # Feature compatibility
        feat_cols_te = [c for c in feat_cols if c in df_te.columns]
        dist_debug(df_tr, df_te, feat_cols_te, tag='SYN')
        ds = LastWindowDataset(df_te, feat_cols_te, args.seq_len, scaler, labels=rul, rul_clip=args.rul_clip)
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        syn_stats = evaluate(model, dl, args.device)
        # dump preds
        preds=[]; units=[]; trues=[]
        with torch.no_grad():
            for x,y,u in dl:
                x=x.to(args.device); p=model(x)
                preds.append(p.cpu().numpy()); trues.append(y.numpy()); units.append(u.numpy())
        yhat=np.concatenate(preds); ytrue=np.concatenate(trues); u=np.concatenate(units)
        # === [AFFINE CALIBRATION – apply to test predictions] =========================
        try:
            cal = json.loads((outdir / "affine_cal.json").read_text(encoding="utf-8"))
            yhat = _apply_affine(yhat, cal["a"], cal["b"])
            print("[CAL] applied affine calibration to SYN predictions.")
        except Exception as e:
            print(f"[CAL] no calibration applied for SYN: {e}")
        # ============================================================================

        pd.DataFrame({'unit':u,'rul_true':ytrue,'rul_pred':yhat,'abs_err':np.abs(yhat-ytrue)}).to_csv(outdir/"preds_synthetic.csv", index=False)

    # FD004 eval
    fd_stats=None
    if eval_fd:
        df_fd = read_cmapss_like(FD_TEST)
        feat_fd = [c for c in feat_cols if c in df_fd.columns]
        dist_debug(df_tr, df_fd, feat_fd, tag='FD004')
        ds = LastWindowDataset(df_fd, feat_fd, args.seq_len, scaler, labels=read_rul(FD_RUL), rul_clip=args.rul_clip)
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        fd_stats = evaluate(model, dl, args.device)
        preds=[]; units=[]; trues=[]
        with torch.no_grad():
            for x,y,u in dl:
                x=x.to(args.device); p=model(x)
                preds.append(p.cpu().numpy()); trues.append(y.numpy()); units.append(u.numpy())
        yhat=np.concatenate(preds); ytrue=np.concatenate(trues); u=np.concatenate(units)
        # === [AFFINE CALIBRATION – apply to test predictions] =========================
        try:
            cal = json.loads((outdir / "affine_cal.json").read_text(encoding="utf-8"))
            yhat = _apply_affine(yhat, cal["a"], cal["b"])
            print("[CAL] applied affine calibration to FD predictions.")
        except Exception as e:
            print(f"[CAL] no calibration applied for FD: {e}")
        # ============================================================================

        pd.DataFrame({'unit':u,'rul_true':ytrue,'rul_pred':yhat,'abs_err':np.abs(yhat-ytrue)}).to_csv(outdir/"preds_fd004.csv", index=False)

    summary = {
        'tag': tag,
        'feat_cols': feat_cols,
        'seq_len': args.seq_len,
        'use_vib': bool(args.vib==1 and set(VIB_COLS).issubset(df_tr.columns)),
        'synthetic_metrics': syn_stats,
        'fd004_metrics': fd_stats,
        'paths': {
            'ckpt': str(outdir/"model_best.pt"),
        }
    }
    (outdir/"eval_summary.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print("\n=== Evaluation Summary ===")
    print(f"Tag: {tag}")
    print(f"Features ({len(feat_cols)}): {feat_cols}")
    if syn_stats: print(f"Synthetic test -> MAE={syn_stats['MAE']:.3f}, RMSE={syn_stats['RMSE']:.3f}")
    if fd_stats:  print(f"FD004 test     -> MAE={fd_stats['MAE']:.3f}, RMSE={fd_stats['RMSE']:.3f}")
    print(f"Artifacts saved in: {outdir}")

if __name__ == '__main__':
    main()
