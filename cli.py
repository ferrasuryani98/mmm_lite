from __future__ import annotations
import argparse, json, sys
import pandas as pd
from .simulator import simulate_timeseries
from .mmm_bayes import BayesianMMM, BayesConfig
from .optimizer import BudgetOptimizer, BudgetConstraints

def cmd_simulate(args):
    channels = args.channels.split(",")
    df = simulate_timeseries(n=args.n, channels=channels, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote simulated dataset to {args.out}")

def cmd_fit(args):
    df = pd.read_csv(args.data, parse_dates=[args.date_col])
    cfg = BayesConfig(
        channels=args.channels.split(","),
        adstock=json.loads(args.adstock),
        saturation=json.loads(args.saturation),
        add_trend=not args.no_trend,
        add_seasonality=not args.no_season,
        seasonality_period=args.season_period,
        seasonality_K=args.season_K,
        draws=args.draws, burn=args.burn, thin=args.thin,
        tau2=args.tau2, a0=args.a0, b0=args.b0, seed=args.seed,
        save_draws=args.save_draws
    )
    m = BayesianMMM(cfg).fit(df, y_col=args.y_col)
    m.save(args.model_out)
    print(f"Bayesian MMM trained. Saved to {args.model_out}")
    print("Posterior mean (first 10):")
    print(pd.Series(m.beta_mean_, index=m.feature_names_).head(10))

def cmd_predict(args):
    m = BayesianMMM.load(args.model)
    df = pd.read_csv(args.data, parse_dates=[args.date_col])
    preds = m.predict(df, return_ci=not args.no_ci, level=args.level)
    out = df[[args.date_col]].join(preds)
    out.to_csv(args.out, index=False)
    print(f"Wrote predictions to {args.out}")

def cmd_optimize(args):
    m = BayesianMMM.load(args.model)
    # Map channel betas from posterior mean
    ch = m.config.channels
    betas = {f"{c}_sat": float(m.beta_mean_[i]) for i, c in enumerate(ch)}
    opt = BudgetOptimizer(ch, betas, m.config.saturation, m.config.adstock)
    cons = BudgetConstraints(
        total_budget=args.budget,
        min_spend={c: args.min for c in ch},
        max_spend={c: args.max for c in ch},
        step=args.step
    )
    alloc = opt.optimize(cons)
    alloc.to_csv(args.out, index=False)
    print("Recommended allocation:")
    print(alloc.to_string(index=False))
    print(f"Saved to {args.out}")

def build_parser():
    p = argparse.ArgumentParser(prog="mmm-bayes-lite", description="Bayesian MMM (Gibbs sampler) MVP.")
    sub = p.add_subparsers()

    psim = sub.add_parser("simulate", help="Generate a synthetic dataset")
    psim.add_argument("--channels", default="search,social,video")
    psim.add_argument("--n", type=int, default=200)
    psim.add_argument("--seed", type=int, default=42)
    psim.add_argument("--out", default="sim_bayes.csv")
    psim.set_defaults(func=cmd_simulate)

    pfit = sub.add_parser("fit", help="Fit Bayesian MMM to CSV")
    pfit.add_argument("--data", required=True)
    pfit.add_argument("--date-col", default="date")
    pfit.add_argument("--y-col", default="sales")
    pfit.add_argument("--channels", default="search,social,video")
    pfit.add_argument("--adstock", default='{"search":0.5,"social":0.3,"video":0.2}')
    pfit.add_argument("--saturation", default='{"search":0.005,"social":0.007,"video":0.004}')
    pfit.add_argument("--no-trend", action="store_true")
    pfit.add_argument("--no-season", action="store_true")
    pfit.add_argument("--season-period", type=int, default=7)
    pfit.add_argument("--season-K", type=int, default=2)
    # Gibbs options
    pfit.add_argument("--draws", type=int, default=800)
    pfit.add_argument("--burn", type=int, default=400)
    pfit.add_argument("--thin", type=int, default=1)
    pfit.add_argument("--tau2", type=float, default=1.0)
    pfit.add_argument("--a0", type=float, default=2.0)
    pfit.add_argument("--b0", type=float, default=1.0)
    pfit.add_argument("--seed", type=int, default=42)
    pfit.add_argument("--save-draws", type=int, default=300)
    pfit.add_argument("--model-out", default="mmm_bayes_model.json")
    pfit.set_defaults(func=cmd_fit)

    ppred = sub.add_parser("predict", help="Predict with credible intervals")
    ppred.add_argument("--model", required=True)
    ppred.add_argument("--data", required=True)
    ppred.add_argument("--date-col", default="date")
    ppred.add_argument("--level", type=float, default=0.95)
    ppred.add_argument("--no-ci", action="store_true")
    ppred.add_argument("--out", default="pred_bayes.csv")
    ppred.set_defaults(func=cmd_predict)

    popt = sub.add_parser("optimize", help="Budget recommendation using posterior mean betas")
    popt.add_argument("--model", required=True)
    popt.add_argument("--budget", type=float, required=True)
    popt.add_argument("--min", type=float, default=0.0)
    popt.add_argument("--max", type=float, default=1e9)
    popt.add_argument("--step", type=float, default=100.0)
    popt.add_argument("--out", default="alloc_bayes.csv")
    popt.set_defaults(func=cmd_optimize)

    return p

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = build_parser()
    if len(argv) == 0:
        p.print_help()
        return 0
    args = p.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
