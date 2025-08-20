from mmm_bayes_lite import simulate_timeseries, BayesianMMM, BayesConfig
df = simulate_timeseries(n=80, channels=["search","social","video"], seed=3)
cfg = BayesConfig(channels=["search","social","video"],
                  adstock={"search":0.5,"social":0.3,"video":0.2},
                  saturation={"search":0.005,"social":0.007,"video":0.004},
                  draws=200, burn=100, seed=3, save_draws=100)
m = BayesianMMM(cfg).fit(df, y_col="sales")
pred = m.predict(df)
print(pred.head().to_string(index=False))
