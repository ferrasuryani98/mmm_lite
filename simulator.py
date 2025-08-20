from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from .transforms import adstock_geometric, saturation_exponential

def simulate_timeseries(
    n: int = 200,
    channels: List[str] = ("search","social","video"),
    seed: int = 42,
    betas: Dict[str, float] | None = None,
    adstock: Dict[str, float] | None = None,
    sat: Dict[str, float] | None = None,
    base: float = 1000.0,
    trend_slope: float = 0.0,
    seas_period: int = 7,
    noise_sigma: float = 50.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if betas is None:
        betas = {f"{ch}_sat": b for ch, b in zip(channels, [500.0, 350.0, 250.0])}
    if adstock is None:
        adstock = {ch: a for ch, a in zip(channels, [0.5, 0.3, 0.2])}
    if sat is None:
        sat = {ch: k for ch, k in zip(channels, [0.005, 0.007, 0.004])}

    spends = {ch: np.clip(rng.normal(2000, 400, size=n).cumsum() - rng.normal(0, 50, size=n).cumsum(), 200, 8000) for ch in channels}
    df = pd.DataFrame(spends)
    df["date"] = pd.date_range("2023-01-01", periods=n, freq="D")
    df = df[["date"] + list(channels)]

    contrib = np.zeros(n, dtype=float)
    for ch in channels:
        xs = adstock_geometric(df[ch].to_numpy(float), adstock[ch])
        zs = saturation_exponential(xs, sat[ch])
        contrib += betas[f"{ch}_sat"] * zs

    trend = trend_slope * np.arange(n)
    season = 100.0 * np.sin(2 * np.pi * np.arange(n) / seas_period)
    noise = rng.normal(0, noise_sigma, size=n)
    sales = base + trend + season + contrib + noise

    df["sales"] = sales
    return df
