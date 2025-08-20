from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from .transforms import saturation_exponential

@dataclass
class BudgetConstraints:
    total_budget: float
    min_spend: Dict[str, float]
    max_spend: Dict[str, float]
    step: float = 100.0

class BudgetOptimizer:
    def __init__(self, channels: List[str], betas: Dict[str, float], k: Dict[str, float], alpha: Dict[str, float]):
        self.channels = channels
        self.betas = betas   # posterior mean betas for *_sat features
        self.k = k
        self.alpha = alpha

    def _steady_multiplier(self, ch: str) -> float:
        a = float(self.alpha.get(ch, 0.0))
        return 1.0 / max(1e-8, (1.0 - a))

    def _marginal_gain(self, ch: str, spend: float) -> float:
        m = self._steady_multiplier(ch)
        k = float(self.k.get(ch, 0.01))
        beta = float(self.betas.get(f"{ch}_sat", 0.0))
        return beta * k * m * np.exp(-k * m * spend)

    def optimize(self, cons: BudgetConstraints) -> pd.DataFrame:
        spends = {ch: float(cons.min_spend.get(ch, 0.0)) for ch in self.channels}
        budget_left = float(cons.total_budget - sum(spends.values()))
        step = float(cons.step)
        while budget_left >= step - 1e-9:
            gains = {ch: self._marginal_gain(ch, spends[ch]) for ch in self.channels
                     if spends[ch] + step <= float(cons.max_spend.get(ch, np.inf))}
            if not gains:
                break
            ch_star = max(gains, key=gains.get)
            spends[ch_star] += step
            budget_left -= step
        contrib = {ch: float(self.betas.get(f"{ch}_sat", 0.0)) *
                        saturation_exponential(self._steady_multiplier(ch) * spends[ch], float(self.k.get(ch, 0.01)))
                   for ch in self.channels}
        out = pd.DataFrame({"channel": list(spends.keys()), "spend": list(spends.values()), "contribution_units": list(contrib.values())})
        return out.sort_values("contribution_units", ascending=False).reset_index(drop=True)
