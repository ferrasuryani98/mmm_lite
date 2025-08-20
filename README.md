# mmm_lite
Bayesian Marketing Mix Modeling in Python — adstock, saturation, CIs, and budget optimization

# Bayesian MMM Lite (MVP)

A minimal **Bayesian Marketing Mix Modeling** toolkit with no external probabilistic libraries.
Uses a fast **Gibbs sampler** for Bayesian linear regression over adstocked/saturated media features,
plus optional trend and Fourier seasonality. Includes a simple **budget optimizer** and **synthetic data simulator**.

## Why this repo?
- Clean, readable reference implementation suited for learning, audits, and O-1A portfolio evidence.
- Few dependencies (`numpy`, `pandas`).
- End-to-end: simulate → fit (Bayesian) → predict (with credible intervals) → budget optimize.

## Model
We model sales as:
\[ y_t \sim \mathcal{N}(\mu_t, \sigma^2),\quad \mu_t = \sum_i \beta_i\, \underbrace{\text{sat}(\text{adstock}(x_{i,t}))}_{\text{media}} + \text{trend}_t + \text{season}_t. \]
Priors (conjugate):
- \(\beta \sim \mathcal{N}(0, \tau^2 I)\)
- \(\sigma^2 \sim \text{InvGamma}(a_0, b_0)\)

We standardize features and the target; the intercept maps back to the original mean of \(y\).

## Install
```bash
pip install numpy pandas
```

## Quickstart (CLI)
```bash
# 1) Simulate data
python -m mmm_bayes_lite.cli simulate --channels search,social,video --n 220 --out data.csv

# 2) Fit Bayesian MMM (with Gibbs sampler)
python -m mmm_bayes_lite.cli fit --data data.csv   --channels search,social,video   --adstock '{"search":0.5,"social":0.3,"video":0.2}'   --saturation '{"search":0.005,"social":0.007,"video":0.004}'   --draws 800 --burn 400 --tau2 1.0 --a0 2.0 --b0 1.0   --model-out mmm_bayes_model.json

# 3) Predict with 95% credible intervals
python -m mmm_bayes_lite.cli predict --model mmm_bayes_model.json --data data.csv --out preds.csv

# 4) Optimize next-period budget using posterior mean betas
python -m mmm_bayes_lite.cli optimize --model mmm_bayes_model.json --budget 60000 --min 0 --max 40000 --step 500 --out alloc.csv
```

## Python API
```python
import pandas as pd
from mmm_bayes_lite import BayesianMMM, BayesConfig, simulate_timeseries
from mmm_bayes_lite import BudgetOptimizer, BudgetConstraints

df = simulate_timeseries(n=180, channels=["search","social","video"], seed=7)

cfg = BayesConfig(
    channels=["search","social","video"],
    adstock={"search":0.5,"social":0.3,"video":0.2},
    saturation={"search":0.005,"social":0.007,"video":0.004},
    draws=600, burn=300, thin=1, tau2=1.0, a0=2.0, b0=1.0, seed=7, save_draws=300
)

m = BayesianMMM(cfg).fit(df, y_col="sales")
preds = m.predict(df, return_ci=True, level=0.95)

# Posterior-mean betas for media features (first len(channels))
betas = {f"{ch}_sat": float(m.beta_mean_[i]) for i, ch in enumerate(cfg.channels)}

from mmm_bayes_lite import BudgetOptimizer, BudgetConstraints
opt = BudgetOptimizer(cfg.channels, betas, cfg.saturation, cfg.adstock)
cons = BudgetConstraints(60000, {c:0 for c in cfg.channels}, {c:40000 for c in cfg.channels}, step=500)
alloc = opt.optimize(cons)
print(alloc.head())
```

## Notes & Caveats
- This is an MVP for educational/open-source reference. For production-grade MMM, consider richer priors (e.g., hierarchical, horseshoe) and more elaborate seasonality/holidays.
- The budget optimizer uses a steady-state approximation for adstock (\(1/(1-\alpha)\)).
- Credible intervals are based on parameter draws (posterior of the mean). To include observation noise, add draws from \(\mathcal{N}(\mu_t, \sigma^2)\).
- Ensure media spends are in comparable units (e.g., \$) before fitting.

## Project layout
```
mmm_bayes_lite/
  __init__.py
  bayes.py
  cli.py
  mmm_bayes.py
  optimizer.py
  simulator.py
  transforms.py
  README.md
```
