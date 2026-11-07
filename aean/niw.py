"""Hierarchical normal-inverse-Wishart (NIW) background models with Student-t posterior predictive scores.

Every spatial cell g of a fixed camera has its own Gaussian (mu_g, Sigma_g). The scene-wide statistics are the
prior centre, so that a cell that has seen little normal data falls back to the scene:

    Sigma_g ~ IW(Psi0, nu0),   mu_g | Sigma_g ~ N(mu0, Sigma_g / kappa0),   Psi0 = (nu0 - C - 1) Sigma0.

With n_g observations, mean xbar_g and scatter S_g, the posterior is NIW(mu_n, kappa_n, Psi_n, nu_n),

    kappa_n = kappa0 + n,  nu_n = nu0 + n,  mu_n = (kappa0 mu0 + n xbar) / kappa_n,
    Psi_n = Psi0 + S + kappa0 n / kappa_n (xbar - mu0)(xbar - mu0)^T,

and the posterior predictive of a new observation is a multivariate Student-t with nu_n - C + 1 degrees of
freedom, location mu_n and scale Psi_n (kappa_n + 1) / (kappa_n (nu_n - C + 1)). The amount of shrinkage is
therefore set by the data count of each cell, not by a fixed weight. kappa0 and nu0 are chosen by type-II
maximum likelihood: the held-out predictive log-likelihood of normal training data (no test data, no labels).

Observations inside a video are strongly correlated, so `n` counts effective observations: the caller passes
a weight per observation (e.g. 1 / number of pixels of a cell for dense residual maps, so that one frame counts
once per cell; 1 for detected objects).
"""
import math

import torch


class NIWCells:
    def __init__(self, C: int, G: int, device: str = "cpu"):
        self.C, self.G, self.device = C, G, device
        self.n = torch.zeros(G, dtype=torch.float64, device=device)
        self.s1 = torch.zeros(G, C, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(G, C, C, dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor, cell: torch.Tensor, w: torch.Tensor | float = 1.0):
        """x: (N, C) observations, cell: (N,) cell index, w: weight per observation (effective count)."""
        x = x.to(self.device, torch.float64)
        cell = cell.to(self.device)
        w = torch.as_tensor(w, dtype=torch.float64, device=self.device).expand(len(x))
        step = max(1024, int(2e7 // (x.shape[1] ** 2)))                    # bounds the (N, C, C) outer products
        for i in range(0, len(x), step):
            xi, ci, wi = x[i:i + step], cell[i:i + step], w[i:i + step]
            self.n.index_add_(0, ci, wi)
            self.s1.index_add_(0, ci, wi[:, None] * xi)
            self.s2.index_add_(0, ci, wi[:, None, None] * xi[:, :, None] * xi[:, None, :])
        return self

    def add(self, other: "NIWCells"):
        self.n += other.n; self.s1 += other.s1; self.s2 += other.s2
        return self

    def finalize(self, kappa0: float, nu0: float, ridge: float = 1e-3):
        """Posterior predictive Student-t parameters of every cell."""
        C = self.C
        N = self.n.sum()
        mu0 = self.s1.sum(0) / N
        Sigma0 = self.s2.sum(0) / N - torch.outer(mu0, mu0)
        Sigma0 = Sigma0 + (ridge * Sigma0.diagonal().mean() + 1e-12) * torch.eye(C, dtype=Sigma0.dtype, device=Sigma0.device)
        nu0 = max(float(nu0), C + 1.0 + 1e-3)
        Psi0 = (nu0 - C - 1) * Sigma0
        n = self.n[:, None]
        xbar = self.s1 / n.clamp(min=1e-12)
        S = self.s2 - n[..., None] * xbar[:, :, None] * xbar[:, None, :]
        kn = kappa0 + self.n
        nun = nu0 + self.n
        mun = (kappa0 * mu0 + self.s1) / kn[:, None]
        d = xbar - mu0
        Psin = Psi0 + S + (kappa0 * self.n / kn)[:, None, None] * d[:, :, None] * d[:, None, :]
        dof = nun - C + 1
        scale = Psin * ((kn + 1) / (kn * dof))[:, None, None]
        L = torch.linalg.cholesky(scale)
        self.mu, self.L, self.dof = mun, L, dof
        self.logdet = 2 * torch.log(L.diagonal(dim1=-2, dim2=-1)).sum(-1)
        self.lognorm = (torch.lgamma((dof + C) / 2) - torch.lgamma(dof / 2)
                        - 0.5 * C * torch.log(dof * math.pi) - 0.5 * self.logdet)
        self.kappa0, self.nu0 = kappa0, nu0
        return self

    def maha(self, x: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Squared Mahalanobis distance under each cell's predictive scale, (N,)."""
        z = (x.to(self.device, torch.float64) - self.mu[cell])[..., None]
        y = torch.linalg.solve_triangular(self.L[cell], z, upper=False)
        return (y[..., 0] ** 2).sum(-1)

    def logpdf(self, x: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        dof = self.dof[cell]
        return self.lognorm[cell] - 0.5 * (dof + self.C) * torch.log1p(self.maha(x, cell) / dof)

    def logpdf_all(self, x: torch.Tensor) -> torch.Tensor:
        """Log predictive density of every observation under every cell, (N, G)."""
        x = x.to(self.device, torch.float64)
        z = x.T[None] - self.mu[:, :, None]                                   # (G, C, N)
        y = torch.linalg.solve_triangular(self.L, z, upper=False)
        m = (y ** 2).sum(1).T                                                 # (N, G)
        return self.lognorm[None] - 0.5 * (self.dof + self.C)[None] * torch.log1p(m / self.dof[None])

    def score(self, x: torch.Tensor, cell: torch.Tensor, kind: str = "nll") -> torch.Tensor:
        """Anomaly score, higher = more anomalous.
        nll : negative log posterior predictive density (includes the per-cell normaliser);
        t   : the data term 0.5 (dof + C) log(1 + delta / dof) only, comparable across cells of different spread."""
        if kind == "nll":
            return -self.logpdf(x, cell)
        dof = self.dof[cell]
        return 0.5 * (dof + self.C) * torch.log1p(self.maha(x, cell) / dof)


def select_hyper(parts: list, held: list, C: int, G: int, kappas=(0.1, 1, 3, 10, 30, 100, 300, 1000, 1e4),
                 nus_extra=(1, 3, 10, 30, 100, 1000), device: str = "cpu"):
    """Type-II ML choice of (kappa0, nu0) by 2-fold cross-validated predictive log-likelihood on NORMAL data.

    parts: list of NIWCells with the sufficient statistics of each training video (same C, G);
    held:  list of (x, cell) held-out observations of the same videos (a subsample).
    nu0 = C + 1 + extra. Returns (kappa0, nu0), table."""
    folds = (list(range(0, len(parts), 2)), list(range(1, len(parts), 2)))
    table = {}
    for k0 in kappas:
        for e in nus_extra:
            nu0 = C + 1 + e
            ll = 0.0
            for fit_i, ev_i in (folds, folds[::-1]):
                if not fit_i or not ev_i:
                    continue
                m = NIWCells(C, G, device)
                for i in fit_i:
                    m.add(parts[i])
                m.finalize(k0, nu0)
                for i in ev_i:
                    x, c = held[i]
                    if len(x):
                        ll += float(m.logpdf(x, c).sum())
            table[(k0, nu0)] = ll
    best = max(table, key=table.get)
    return best, table


def _kmeans(X: torch.Tensor, K: int, iters: int = 30, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        a = torch.cat([torch.cdist(X[i:i + 8192], C).argmin(1) for i in range(0, len(X), 8192)])
        n = torch.zeros(K, dtype=X.dtype, device=X.device).index_add_(0, a, torch.ones_like(a, dtype=X.dtype))
        s = torch.zeros_like(C).index_add_(0, a, X)
        C = torch.where(n[:, None] > 0, s / n.clamp(min=1)[:, None], C)
    return C


class NIWMixture:
    """Finite mixture of Gaussians with conjugate priors: every component has a NIW prior centred on the data-wide
    statistics (NIWCells with one 'cell' per component) and the weights a symmetric Dirichlet(alpha0 / K) prior.
    Components are fixed by k-means on normal data (hard assignment); the score is the negative log posterior
    predictive density, a mixture of multivariate Student-t distributions:
        p(x) = sum_k (n_k + alpha0 / K) / (N + alpha0) * t_k(x).
    K = 1 is the single NIW model; growing K approaches a non-parametric density of the normal data."""

    def __init__(self, K: int, kappa0: float, nu0: float, alpha0: float = 1.0, device: str = "cpu"):
        self.K, self.kappa0, self.nu0, self.alpha0, self.device = K, kappa0, nu0, alpha0, device

    def fit(self, X: torch.Tensor, seed: int = 0, centres: torch.Tensor | None = None):
        X = X.to(self.device, torch.float64)
        if centres is not None:
            self.centres = centres
        else:
            self.centres = _kmeans(X, self.K, seed=seed) if self.K > 1 else X.mean(0, keepdim=True)
        a = self.assign(X)
        self.cells = NIWCells(X.shape[1], self.K, self.device).update(X, a).finalize(self.kappa0, self.nu0)
        n = self.cells.n
        self.logw = torch.log((n + self.alpha0 / self.K) / (n.sum() + self.alpha0))
        return self

    def assign(self, X):
        return torch.cat([torch.cdist(X[i:i + 8192], self.centres).argmin(1) for i in range(0, len(X), 8192)])

    def logpdf(self, X: torch.Tensor, chunk: int = 0) -> torch.Tensor:
        X = X.to(self.device, torch.float64)
        chunk = chunk or max(64, int(4e6 // (self.K * X.shape[1])))      # bounds the (K, C, chunk) work tensor
        out = []
        for i in range(0, len(X), chunk):
            out.append(torch.logsumexp(self.cells.logpdf_all(X[i:i + chunk]) + self.logw, 1))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64, device=self.device)


def select_mixture(fit_sets: list, eval_sets: list, Ks=(1, 4, 16, 64, 128, 256), kappas=(1, 10, 100), nu_extra=(3, 30, 300),
                   device: str = "cpu"):
    """Choose (K, kappa0, nu0) by held-out predictive log-likelihood of NORMAL data: fit_sets[i] / eval_sets[i]
    are the two halves of fold i (split by video). Returns the best triple and the table."""
    table = {}
    fit_sets = [X.to(device, torch.float64) for X in fit_sets]
    for K in Ks:
        cents = [(_kmeans(Xf, K) if K > 1 else Xf.mean(0, keepdim=True)) if len(Xf) >= 2 * K else None for Xf in fit_sets]
        for k0 in kappas:
            for e in nu_extra:
                ll = 0.0
                for Xf, Xe, cen in zip(fit_sets, eval_sets, cents):
                    if cen is None:
                        ll = -math.inf
                        break
                    m = NIWMixture(K, k0, Xf.shape[1] + 1 + e, device=device).fit(Xf, centres=cen)
                    ll += float(m.logpdf(Xe).sum())
                table[(K, k0, e)] = ll
    best = max(table, key=table.get)
    return best, table


class PluginMixture(NIWMixture):
    """Ablation without priors: the same k-means components, each scored with its maximum-likelihood Gaussian
    (sample mean and covariance + the ridge of NIWCells.finalize), weights n_k / N. Not Bayesian."""

    def fit(self, X: torch.Tensor, seed: int = 0, centres: torch.Tensor | None = None):
        X = X.to(self.device, torch.float64)
        self.centres = centres if centres is not None else (_kmeans(X, self.K, seed=seed) if self.K > 1 else X.mean(0, keepdim=True))
        a = self.assign(X)
        c = NIWCells(X.shape[1], self.K, self.device).update(X, a)
        C = X.shape[1]
        n = c.n.clamp(min=1)[:, None]
        mu = c.s1 / n
        cov = c.s2 / n[..., None] - mu[:, :, None] * mu[:, None, :]
        eye = torch.eye(C, dtype=cov.dtype, device=cov.device)
        cov = cov + (1e-3 * cov.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-6)[:, None, None] * eye
        self.mu, self.L = mu, torch.linalg.cholesky(cov)
        self.lognorm = -0.5 * C * math.log(2 * math.pi) - torch.log(self.L.diagonal(dim1=-2, dim2=-1)).sum(-1)
        self.logw = torch.log(c.n.clamp(min=1e-12) / c.n.sum())
        return self

    def logpdf(self, X: torch.Tensor, chunk: int = 0) -> torch.Tensor:
        X = X.to(self.device, torch.float64)
        chunk = chunk or max(64, int(4e6 // (self.K * X.shape[1])))
        out = []
        for i in range(0, len(X), chunk):
            z = X[i:i + chunk].T[None] - self.mu[:, :, None]
            y = torch.linalg.solve_triangular(self.L, z, upper=False)
            lp = self.lognorm[None] - 0.5 * (y ** 2).sum(1).T
            out.append(torch.logsumexp(lp + self.logw, 1))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64, device=self.device)
