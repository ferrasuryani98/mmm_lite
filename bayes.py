from __future__ import annotations
import numpy as np

def invgamma_sample(shape: float, scale: float, rng: np.random.Generator) -> float:
    # Inverse-Gamma(shape=a, scale=b): p(sigma2) ∝ sigma2^{-(a+1)} exp(-b/sigma2)
    # If X ~ Gamma(a, 1/scale) then 1/X ~ InvGamma(a, scale).
    g = rng.gamma(shape, 1.0/scale)
    return 1.0 / max(g, 1e-12)

def gibbs_linear_regression(
    X: np.ndarray, y: np.ndarray, draws: int = 1000, burn: int = 500, thin: int = 1,
    tau2: float = 1.0, a0: float = 2.0, b0: float = 1.0, seed: int = 42
):
    """Bayesian linear regression via Gibbs sampler.
    Model: y ~ N(X beta, sigma2 I), beta ~ N(0, tau2 I), sigma2 ~ InvGamma(a0, b0).
    Returns dict with beta draws (n_draws x p) and sigma2 draws.
    y should be centered/scaled (intercept handled outside).
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y

    beta = np.zeros(p, dtype=float)
    sigma2 = 1.0

    kept_beta = []
    kept_sigma2 = []

    # Precompute identity
    I = np.eye(p, dtype=float)

    total_iter = burn + draws * thin
    for it in range(total_iter):
        # beta | sigma2, y ~ N(μ, Σ), Σ = (XtX/sigma2 + I/tau2)^(-1), μ = Σ (X^T y)/sigma2
        A = XtX / sigma2 + I / max(tau2, 1e-12)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        mu = A_inv @ (Xty / sigma2)
        # sample beta
        L = np.linalg.cholesky(A_inv + 1e-12*np.eye(p))
        z = rng.normal(size=p)
        beta = mu + L @ z

        # sigma2 | beta, y ~ InvGamma(a0 + n/2, b0 + 0.5 * ||y - X beta||^2)
        resid = y - X @ beta
        shape = a0 + 0.5 * n
        scale = b0 + 0.5 * float(resid @ resid)
        sigma2 = invgamma_sample(shape, scale, rng)

        if it >= burn and ((it - burn) % thin == 0):
            kept_beta.append(beta.copy())
            kept_sigma2.append(sigma2)

    betas = np.asarray(kept_beta)
    sigmas = np.asarray(kept_sigma2)
    return { "beta": betas, "sigma2": sigmas }
