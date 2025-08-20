from __future__ import annotations
import numpy as np
import pandas as pd

def adstock_geometric(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        s[t] = x[t] + (alpha * s[t-1] if t > 0 else x[t])
    return s

def saturation_exponential(x: np.ndarray, k: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 - np.exp(-k * np.maximum(x, 0.0))

def fourier_seasonality(n: int, period: int = 7, K: int = 2) -> np.ndarray:
    t = np.arange(n, dtype=float)
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.vstack(cols).T  # (n, 2K)

def build_media_features(df: pd.DataFrame, channels: list[str], adstock: dict, sat: dict) -> pd.DataFrame:
    feats = {}
    for ch in channels:
        x = df[ch].to_numpy(dtype=float)
        alpha = float(adstock.get(ch, 0.0))
        k = float(sat.get(ch, 0.01))
        xs = adstock_geometric(x, alpha) if alpha > 0 else x
        zs = saturation_exponential(xs, k)
        feats[f"{ch}_sat"] = zs
    return pd.DataFrame(feats, index=df.index)
