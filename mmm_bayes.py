from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional
import json
import numpy as np
import pandas as pd
from .transforms import build_media_features, fourier_seasonality
from .bayes import gibbs_linear_regression

@dataclass
class BayesConfig:
    channels: List[str]
    adstock: Dict[str, float]
    saturation: Dict[str, float]
    add_trend: bool = True
    add_seasonality: bool = True
    seasonality_period: int = 7
    seasonality_K: int = 2
    # Gibbs options
    draws: int = 1000
    burn: int = 500
    thin: int = 1
    tau2: float = 1.0
    a0: float = 2.0
    b0: float = 1.0
    seed: int = 42
    save_draws: int = 400  # how many draws to persist (for CI/predictive). If <=0, saves none.

class BayesianMMM:
    def __init__(self, config: BayesConfig):
        self.config = config
        self.feature_names_: List[str] = []
        self.scalers_: Dict[str, Tuple[float, float]] = {}
        self.y_mu_: float = 0.0
        self.y_std_: float = 1.0
        self.beta_draws_: Optional[np.ndarray] = None  # (S, p)
        self.sigma2_draws_: Optional[np.ndarray] = None
        self.beta_mean_: Optional[np.ndarray] = None

    # ----- design matrix helpers -----
    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        media = build_media_features(df, self.config.channels, self.config.adstock, self.config.saturation)
        X_list = [media]
        if self.config.add_trend:
            X_list.append(pd.DataFrame({"trend": np.arange(len(df), dtype=float)}, index=df.index))
        if self.config.add_seasonality:
            seas = fourier_seasonality(len(df), self.config.seasonality_period, self.config.seasonality_K)
            S = pd.DataFrame(seas, index=df.index, columns=[f"s{i}" for i in range(seas.shape[1])])
            X_list.append(S)
        X = pd.concat(X_list, axis=1)
        self.feature_names_ = list(X.columns)
        # standardize
        Xs = []
        self.scalers_.clear()
        for c in self.feature_names_:
            col = X[c].to_numpy(float)
            mu = float(np.nanmean(col))
            sd = float(np.nanstd(col) + 1e-8)
            self.scalers_[c] = (mu, sd)
            Xs.append((col - mu) / sd)
        return np.column_stack(Xs)

    def _design_matrix_with_scalers(self, df: pd.DataFrame) -> np.ndarray:
        media = build_media_features(df, self.config.channels, self.config.adstock, self.config.saturation)
        X_list = [media]
        if self.config.add_trend:
            X_list.append(pd.DataFrame({"trend": np.arange(len(df), dtype=float)}, index=df.index))
        if self.config.add_seasonality:
            seas = fourier_seasonality(len(df), self.config.seasonality_period, self.config.seasonality_K)
            S = pd.DataFrame(seas, index=df.index, columns=[f"s{i}" for i in range(seas.shape[1])])
            X_list.append(S)
        X = pd.concat(X_list, axis=1)[self.feature_names_]
        Xs = []
        for c in self.feature_names_:
            mu, sd = self.scalers_[c]
            col = X[c].to_numpy(float)
            Xs.append((col - mu) / sd)
        return np.column_stack(Xs)

    # ----- fit/predict -----
    def fit(self, df: pd.DataFrame, y_col: str = "sales") -> "BayesianMMM":
        y = df[y_col].to_numpy(float)
        self.y_mu_ = float(np.nanmean(y))
        self.y_std_ = float(np.nanstd(y) + 1e-8)
        y_std = (y - self.y_mu_) / self.y_std_

        X = self._design_matrix(df)

        out = gibbs_linear_regression(
            X, y_std,
            draws=self.config.draws,
            burn=self.config.burn,
            thin=self.config.thin,
            tau2=self.config.tau2,
            a0=self.config.a0,
            b0=self.config.b0,
            seed=self.config.seed,
        )
        beta_draws = out["beta"]
        sigma2_draws = out["sigma2"]

        # keep at most save_draws
        if self.config.save_draws and self.config.save_draws > 0:
            S = min(self.config.save_draws, beta_draws.shape[0])
            self.beta_draws_ = beta_draws[-S:]
            self.sigma2_draws_ = sigma2_draws[-S:]
        else:
            self.beta_draws_ = None
            self.sigma2_draws_ = None

        self.beta_mean_ = beta_draws.mean(axis=0)
        return self

    def predict(self, df: pd.DataFrame, return_ci: bool = True, level: float = 0.95) -> pd.DataFrame:
        Xnew = self._design_matrix_with_scalers(df)
        mean_mu = Xnew @ self.beta_mean_
        y_mean = self.y_mu_ + mean_mu * self.y_std_
        out = {"y_pred": y_mean}

        if return_ci and self.beta_draws_ is not None:
            mu_draws = (self.beta_draws_ @ Xnew.T)  # (S, n)
            y_draws = self.y_mu_ + mu_draws * self.y_std_
            lo = np.percentile(y_draws, (1 - level) / 2 * 100, axis=0)
            hi = np.percentile(y_draws, (1 + level) / 2 * 100, axis=0)
            out["y_lo"] = lo
            out["y_hi"] = hi

        return pd.DataFrame(out, index=df.index)

    def channel_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._design_matrix_with_scalers(df)
        # map first len(channels) features to channels (they are *_sat first)
        k = len(self.config.channels)
        if self.beta_draws_ is not None:
            contrib = (self.beta_draws_[:, :k] @ X[:, :k].T)  # (S, n)
            contrib_mean = contrib.mean(axis=0)
        else:
            contrib_mean = X[:, :k] @ self.beta_mean_[:k]
        contrib_mean = self.y_mu_ + contrib_mean * self.y_std_ - self.y_mu_  # scale to original units (mean effect)
        out = pd.DataFrame({"contribution": contrib_mean}, index=df.index)
        return out

    # ----- persistence -----
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "config": asdict(self.config),
            "feature_names": self.feature_names_,
            "scalers": {k: [float(v[0]), float(v[1])] for k, v in self.scalers_.items()},
            "y_mu": self.y_mu_,
            "y_std": self.y_std_,
            "beta_mean": self.beta_mean_.tolist() if self.beta_mean_ is not None else None,
        }
        if self.beta_draws_ is not None:
            d["beta_draws"] = self.beta_draws_.tolist()
            d["sigma2_draws"] = self.sigma2_draws_.tolist()
        else:
            d["beta_draws"] = None
            d["sigma2_draws"] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BayesianMMM":
        cfg = BayesConfig(**d["config"])
        m = cls(cfg)
        m.feature_names_ = list(d["feature_names"])
        m.scalers_ = {k: (float(v[0]), float(v[1])) for k, v in d["scalers"].items()}
        m.y_mu_ = float(d["y_mu"])
        m.y_std_ = float(d["y_std"])
        m.beta_mean_ = np.asarray(d["beta_mean"]) if d["beta_mean"] is not None else None
        if d.get("beta_draws") is not None:
            m.beta_draws_ = np.asarray(d["beta_draws"])
            m.sigma2_draws_ = np.asarray(d["sigma2_draws"])
        else:
            m.beta_draws_ = None
            m.sigma2_draws_ = None
        return m

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BayesianMMM":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
