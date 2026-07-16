"""
ci_test.py
==========
Doubly-robust conditional independence (CI) test with generative neural networks,
following Zhang, Huang, Yang & Shao (2026, JRSSB), refactored from the original
``skew_*.ipynb`` experiments.

This version fixes the Type-I-error inflation observed on the skewed / heteroscedastic
data-generating process. The diagnosis (see the accompanying report) was that the test
statistic and the wild bootstrap are correct -- with the *true* conditional generators
the size is exactly nominal -- but the learned generators were badly under-trained:
they collapsed the conditional standard deviation to about half of the truth, which
breaks the double-robustness the size guarantee relies on.

Fixes applied here, relative to the original notebooks
------------------------------------------------------
1. Proper generator training: learning rate 5e-4 (was 5e-5), no aggressive gradient
   clipping (was clip_grad_norm=0.5, which throttled almost every step), more epochs.
2. Standardization is actually applied (the original passed ``preprocess='scale'`` but
   never used it). Each variable is z-scored using the *training* fold's statistics.
3. Selectable alignment penalty (``align_mode``): 'mean' (the tutorial's E[X|Z] MSE
   idea), 'median' (the original notebook's median pinball), 'quantile' (a spread-aware
   multi-quantile pinball, the recommended default), or 'none'. The median-only penalty
   only pins the center, so it cannot fix the conditional *spread* that was collapsing;
   the multi-quantile version matches center AND spread.
4. Honest cross-fitting: a FRESH generator is trained on each fold's training data
   (the original created the generators once and warm-started them across folds, so
   later test folds had already been seen during training).
5. Configurable generator depth / width (the seven original notebooks differed only
   in these two numbers).
6. Vectorized conditional-MMD losses (faster and easier to read).
7. Oracle mode is kept as a built-in size sanity-check.
8. Pluggable data-generating models via a small DGP registry, so a student can switch
   between DGPs (e.g. this project's skewed one and the paper's Section 4.1 Gaussian
   model) or register their own with ``register_dgp`` -- see :class:`DGP`.

The public entry points are :func:`run_ci_test` (one dataset -> one p-value) and
:func:`run_experiment` (Monte-Carlo rejection rate over many datasets); both take a
``dgp`` argument (a name in ``DGPS`` or a :class:`DGP` object).
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ----------------------------------------------------------------------------------
# reproducibility / device
# ----------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------------
# data-generating process (skewed, heteroscedastic, nonlinear-mean)
# ----------------------------------------------------------------------------------
def _standardized_lognormal(num, dim, sigma=1.0, device=None):
    """Right-skewed noise with mean 0 and variance 1 (shifted, scaled lognormal)."""
    n = torch.randn(num, dim, device=device)
    denom = np.sqrt((np.exp(sigma ** 2) - 1.0) * np.exp(sigma ** 2))
    return (torch.exp(sigma * n) - 1.0) / denom


def _mean_std_functions(z1, linear_mean=False):
    """Conditional mean / std of X|Z and Y|Z as functions of the first coordinate of Z.
    ``linear_mean=True`` uses the simplified linear means m_X=0.8z, m_Y=-0.6z from
    Simulation 8 (nonlinear otherwise)."""
    sig = torch.sigmoid
    if linear_mean:
        m_x = 0.8 * z1
        m_y = -0.6 * z1
    else:
        m_x = 0.8 * z1 + 0.5 * torch.sin(1.5 * z1)
        m_y = -0.6 * z1 + 0.4 * (z1 ** 2)
    s_x = 0.5 + 0.25 * torch.abs(z1) + 0.15 * sig(z1)
    s_y = 0.5 + 0.20 * torch.abs(z1) + 0.15 * sig(-z1)
    return m_x, m_y, s_x, s_y


def sample_data(n, hypothesis="H0", dx=1, dy=1, dz=1, nstd=1.0, alpha_x=0.20,
                dist_z="gaussian", lognormal_sigma=1.0, linear_mean=False, device=None):
    """Draw ``n`` i.i.d. triples (X, Y, Z) from the skewed DGP.

    Under ``H0`` the noise terms of X and Y are independent given Z, so X _|_ Y | Z.
    Under ``H1`` a Bernoulli(alpha_x) fraction of samples share Y's noise in X, which
    injects conditional dependence whose strength grows with ``alpha_x``.
    """
    if dist_z == "gaussian":
        Z = torch.randn(n, dz, device=device)
    elif dist_z == "laplace":
        Z = torch.distributions.Laplace(0.0, 1.0).sample((n, dz)).to(device)
    else:
        raise ValueError("dist_z must be 'gaussian' or 'laplace'.")

    eps_x = _standardized_lognormal(n, dx, lognormal_sigma, device)
    eps_y = _standardized_lognormal(n, dy, lognormal_sigma, device)

    z1 = Z[:, [0]]
    m_x, m_y, s_x, s_y = _mean_std_functions(z1, linear_mean)
    if dx > 1:
        m_x, s_x = m_x.repeat(1, dx), s_x.repeat(1, dx)
    if dy > 1:
        m_y, s_y = m_y.repeat(1, dy), s_y.repeat(1, dy)

    Y = m_y + nstd * s_y * eps_y
    if hypothesis == "H0":
        X = m_x + nstd * s_x * eps_x
    elif hypothesis == "H1":
        delta = torch.bernoulli(torch.full((n, 1), float(alpha_x), device=device))
        delta_x = delta.repeat(1, dx) if dx > 1 else delta
        shared = eps_y if dx == dy else (eps_y.repeat(1, dx) if dy == 1 else eps_y[:, :dx])
        X = m_x + nstd * s_x * ((1.0 - delta_x) * eps_x + delta_x * shared)
    else:
        raise ValueError("hypothesis must be 'H0' or 'H1'.")
    return X, Y, Z


def oracle_conditional_sample(Z, m, dx=1, dy=1, nstd=1.0, lognormal_sigma=1.0,
                              linear_mean=False, device=None, **_):
    """Sample ``m`` draws of (X, Y) from the TRUE conditionals P(X|z), P(Y|z) for each
    row ``z`` of ``Z``. Used for the oracle size sanity-check. Returns tensors of shape
    (len(Z), m, dx) and (len(Z), m, dy). (The marginal conditionals are the same under
    H0 and H1 for this DGP, so no hypothesis argument is needed; extra kwargs like
    ``dist_z`` / ``alpha_x`` are accepted and ignored.)"""
    n = Z.shape[0]
    z1 = Z[:, [0]]
    m_x, m_y, s_x, s_y = _mean_std_functions(z1, linear_mean)       # (n, 1)
    eps_x = _standardized_lognormal(n * m, dx, lognormal_sigma, device).reshape(n, m, dx)
    eps_y = _standardized_lognormal(n * m, dy, lognormal_sigma, device).reshape(n, m, dy)
    X = m_x.unsqueeze(1) + nstd * s_x.unsqueeze(1) * eps_x
    Y = m_y.unsqueeze(1) + nstd * s_y.unsqueeze(1) * eps_y
    return X, Y


# ----------------------------------------------------------------------------------
# Second DGP: the Gaussian post-linear model from Zhang et al. Section 4.1
#   Z = e3,  Y = Z + e1,  X = Z + delta*e1 + (1-delta)*e2,  delta ~ Bernoulli(alpha_x)
# with e1, e2, e3 ~ N(0,1) independent. Under H0 (alpha_x=0) X = Z + e2 _|_ Y | Z; the
# conditional dependence strength grows with alpha_x. This is an "easy" DGP (Gaussian,
# homoscedastic, linear mean) -- a useful contrast to the hard skewed one.
# ----------------------------------------------------------------------------------
def sample_gaussian(n, hypothesis="H0", dz=1, alpha_x=0.20, device=None, **_):
    Z = torch.randn(n, dz, device=device)
    z1 = Z[:, [0]]
    e1 = torch.randn(n, 1, device=device)
    e2 = torch.randn(n, 1, device=device)
    Y = z1 + e1
    if hypothesis == "H0":
        X = z1 + e2
    elif hypothesis == "H1":
        delta = torch.bernoulli(torch.full((n, 1), float(alpha_x), device=device))
        X = z1 + delta * e1 + (1.0 - delta) * e2
    else:
        raise ValueError("hypothesis must be 'H0' or 'H1'.")
    return X, Y, Z


def oracle_gaussian(Z, m, device=None, **_):
    """True conditionals: X|Z ~ N(z1, 1) and Y|Z ~ N(z1, 1) (same under H0 and H1)."""
    n = Z.shape[0]
    z1 = Z[:, [0]].unsqueeze(1)                       # (n, 1, 1)
    X = z1 + torch.randn(n, m, 1, device=device)
    Y = z1 + torch.randn(n, m, 1, device=device)
    return X, Y


# ----------------------------------------------------------------------------------
# Heteroskedastic model (Simulation 5, Section 3.1): X has a Z-dependent variance and
# ZERO conditional mean, so mean-alignment is useless by design.
#   Y = Z + ey,  sigma(Z) = 0.3 + 1.2|Z|,  H0: X = sigma(Z) ex
#   H1: X = sigma(Z)[(1-d) ex + d ey],  d ~ Bernoulli(alpha_x)
# ----------------------------------------------------------------------------------
def sample_heteroskedastic(n, hypothesis="H0", dz=1, alpha_x=0.20, device=None, **_):
    Z = torch.randn(n, dz, device=device)
    z1 = Z[:, [0]]
    ex = torch.randn(n, 1, device=device)
    ey = torch.randn(n, 1, device=device)
    sig = 0.3 + 1.2 * z1.abs()
    Y = z1 + ey
    if hypothesis == "H0":
        X = sig * ex
    elif hypothesis == "H1":
        d = torch.bernoulli(torch.full((n, 1), float(alpha_x), device=device))
        X = sig * ((1.0 - d) * ex + d * ey)
    else:
        raise ValueError("hypothesis must be 'H0' or 'H1'.")
    return X, Y, Z


def oracle_heteroskedastic(Z, m, device=None, **_):
    n = Z.shape[0]
    z1 = Z[:, [0]].unsqueeze(1)
    sig = 0.3 + 1.2 * z1.abs()
    X = sig * torch.randn(n, m, 1, device=device)         # X|Z ~ N(0, sigma(Z)^2)
    Y = z1 + torch.randn(n, m, 1, device=device)          # Y|Z ~ N(z1, 1)
    return X, Y


# ----------------------------------------------------------------------------------
# Heavy-tailed Student-t model (Simulation 5, Section 4): heteroskedastic with heavy
# tails, so the conditional median is a more stable target than the mean.
#   mX=mY=Z,  s(Z) = 0.5 + |Z|,  eX, eY ~ standardized Student-t (df, unit variance)
#   H0: X = Z + s(Z)eX, Y = Z + s(Z)eY
#   H1: X = Z + s(Z)eX, Y = Z + s(Z)[d eX + (1-d) eY],  d ~ Bernoulli(alpha_x)
# ----------------------------------------------------------------------------------
def _standardized_t(shape, df, device):
    """Student-t scaled to unit variance (requires df > 2)."""
    t = torch.distributions.StudentT(float(df)).sample(shape).to(device)
    return t * float(np.sqrt((df - 2.0) / df))


def sample_student_t(n, hypothesis="H0", dz=1, alpha_x=0.20, df=3.0, device=None, **_):
    Z = torch.randn(n, dz, device=device)
    z1 = Z[:, [0]]
    s = 0.5 + z1.abs()
    eX = _standardized_t((n, 1), df, device)
    eY = _standardized_t((n, 1), df, device)
    X = z1 + s * eX
    if hypothesis == "H0":
        Y = z1 + s * eY
    elif hypothesis == "H1":
        d = torch.bernoulli(torch.full((n, 1), float(alpha_x), device=device))
        Y = z1 + s * (d * eX + (1.0 - d) * eY)
    else:
        raise ValueError("hypothesis must be 'H0' or 'H1'.")
    return X, Y, Z


def oracle_student_t(Z, m, df=3.0, device=None, **_):
    n = Z.shape[0]
    z1 = Z[:, [0]].unsqueeze(1)
    s = 0.5 + z1.abs()
    X = z1 + s * _standardized_t((n, m, 1), df, device)
    Y = z1 + s * _standardized_t((n, m, 1), df, device)
    return X, Y


# ----------------------------------------------------------------------------------
# DGP registry -- lets the notebook / a student swap or add data-generating models.
# A DGP bundles two functions: how to draw (X, Y, Z), and how to draw from the TRUE
# conditionals P(X|Z), P(Y|Z) (needed only for the oracle size sanity-check).
# ----------------------------------------------------------------------------------
class DGP:
    def __init__(self, name, sample, oracle, description=""):
        self.name = name
        self._sample = sample          # (n, hypothesis, device, **kw) -> (X, Y, Z)
        self._oracle = oracle          # (Z, m, device, **kw) -> (X_fake, Y_fake)
        self.description = description

    def sample(self, n, hypothesis="H0", device=None, **kw):
        return self._sample(n, hypothesis=hypothesis, device=device, **kw)

    def oracle(self, Z, m, device=None, **kw):
        return self._oracle(Z, m, device=device, **kw)


DGPS = {}


def register_dgp(dgp):
    """Add a DGP to the registry so it can be selected by name. Returns the DGP."""
    DGPS[dgp.name] = dgp
    return dgp


def get_dgp(dgp):
    """Resolve a DGP given either a name or a DGP object."""
    if isinstance(dgp, DGP):
        return dgp
    if dgp in DGPS:
        return DGPS[dgp]
    raise KeyError(f"unknown dgp {dgp!r}; registered: {list(DGPS)}")


register_dgp(DGP(
    "skew", sample=sample_data, oracle=oracle_conditional_sample,
    description="This project's hard case: skewed lognormal noise, heteroscedastic "
                "spread, nonlinear means. Knobs: dx, dy, dz, nstd, dist_z, alpha_x."))
register_dgp(DGP(
    "gaussian", sample=sample_gaussian, oracle=oracle_gaussian,
    description="Zhang et al. Section 4.1 (Sim 4/5): Z=e3, Y=Z+e1, X=Z+d*e1+(1-d)*e2, "
                "d~Bernoulli(alpha_x). Gaussian / homoscedastic / linear. Knobs: dz, alpha_x."))
register_dgp(DGP(
    "skew_linear",
    sample=lambda n, hypothesis, device, **kw: sample_data(
        n, hypothesis, device=device, linear_mean=True, **kw),
    oracle=lambda Z, m, device, **kw: oracle_conditional_sample(
        Z, m, device=device, linear_mean=True, **kw),
    description="Simulation 8 (Sec 1.3): the skewed DGP but with LINEAR means "
                "m_X=0.8z, m_Y=-0.6z. Knobs: dx, dy, dz, nstd, dist_z, alpha_x."))
register_dgp(DGP(
    "heteroskedastic", sample=sample_heteroskedastic, oracle=oracle_heteroskedastic,
    description="Simulation 5 (Sec 3.1): Y=Z+ey, X=sigma(Z)*ex, sigma(Z)=0.3+1.2|Z|. "
                "Gaussian but heteroscedastic with ZERO conditional mean. Knobs: dz, alpha_x."))
register_dgp(DGP(
    "student_t", sample=sample_student_t, oracle=oracle_student_t,
    description="Simulation 5 (Sec 4): heavy-tailed, mX=mY=Z, s(Z)=0.5+|Z|, standardized "
                "Student-t noise. Knobs: dz, alpha_x, df (default 3)."))


# ----------------------------------------------------------------------------------
# per-variable standardization (fit on the training fold, applied everywhere)
# ----------------------------------------------------------------------------------
class Scaler:
    """z-score each column; ``fit`` on training data only to avoid test leakage."""
    def __init__(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, keepdim=True).clamp_min(1e-6)

    def transform(self, x):
        return (x - self.mean) / self.std


# ----------------------------------------------------------------------------------
# conditional generator  G(z, noise) -> target
# ----------------------------------------------------------------------------------
class ConditionalGenerator(nn.Module):
    """MLP with ``depth`` hidden layers of width ``width``; maps concat(Z, noise) to the
    target variable. ``depth`` reproduces the 1-/2-/3-layer notebooks."""
    def __init__(self, z_dim, out_dim, noise_dim, width=1024, depth=3,
                 relu_slope=0.1, dropout=0.0):
        super().__init__()
        acts = []
        dims = [z_dim + noise_dim] + [width] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LeakyReLU(relu_slope)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, noise):
        return self.net(torch.cat([z, noise], dim=1))


def sample_noise(n, noise_dim, device, var=1.0, kind="normal"):
    if kind == "normal":
        return torch.randn(n, noise_dim, device=device) * np.sqrt(var)
    if kind == "uniform":
        return torch.rand(n, noise_dim, device=device)
    raise ValueError("noise kind must be 'normal' or 'uniform'.")


# ----------------------------------------------------------------------------------
# kernels and conditional-MMD training loss (vectorized)
# ----------------------------------------------------------------------------------
def _median_bandwidth(a, p=1):
    d = torch.cdist(a, a, p=p)
    d = d[d > 0]
    return d.median().clamp_min(1e-8).item() if d.numel() else 1.0


def _pairwise(a, b, p):
    # ||a_i - b_j||_p over the last dim; a:(...,d) b:(...,d)
    if p == 1:
        return torch.sum(torch.abs(a - b), dim=-1)
    return torch.sum((a - b) ** 2, dim=-1)


def conditional_mmd_loss(Y, Y_fake, Z, sigma_z, sigma_y, w_laplacian=1.0, w_gaussian=1.0):
    """Sample conditional-MMD training loss between P(Y|Z) and the generated P(Yhat|Z).

    ``Y``: (n, d) real targets. ``Y_fake``: (n, M, d) generated targets (M draws per z).
    ``w_laplacian`` / ``w_gaussian`` weight the two kernels (original ``lambda_1`` /
    ``lambda_2``); set one to 0 to use a single kernel.
    """
    if Y.dim() == 1:
        Y = Y.reshape(-1, 1)
    if Y_fake.dim() == 2:
        Y_fake = Y_fake.unsqueeze(1)
    n = Z.shape[0]
    eye = torch.eye(n, device=Y.device)

    def _loss(p, sy):
        Kz = torch.exp(-torch.cdist(Z, Z, p=1) / sigma_z) if p == 1 \
            else torch.exp(-(torch.cdist(Z, Z, p=2) ** 2) / sigma_z)
        Kyy = torch.exp(-_pairwise(Y[:, None, :], Y[None, :, :], p) / sy)
        d_yf = _pairwise(Y[:, None, None, :], Y_fake[None, :, :, :], p)     # (n,n,M)
        Kyf = torch.exp(-d_yf / sy).mean(2)
        Kfy = Kyf.T
        d_ff = _pairwise(Y_fake[:, None, :, None, :], Y_fake[None, :, None, :, :], p)
        Kff = torch.exp(-d_ff / sy).mean(dim=(2, 3))                        # (n,n)
        FF = (Kyy - Kyf - Kfy + Kff) * Kz * (1 - eye)
        return FF.sum() / (n * (n - 1))

    out = 0.0
    if w_laplacian:
        out = out + w_laplacian * _loss(1, sigma_y)
    if w_gaussian:
        out = out + w_gaussian * _loss(2, sigma_y)
    return out


def alignment_penalty(Y, Y_group, mode="quantile", taus=(0.1, 0.5, 0.9)):
    """Optional penalty that pulls the generated conditional summary toward the data.

    ``Y``: (n, d) observed.  ``Y_group``: (n, S, d) generated draws per observation.
    ``mode``:
        'none'     -- no penalty (pure conditional-MMD, as in the original paper).
        'mean'     -- MSE between the generated conditional mean E[Yhat|Z] and Y
                      (the mean-alignment idea from the tutorial).
        'median'   -- pinball at tau=0.5 (the original notebook's ``lambda_median`` term);
                      pins the center only, so it does NOT fix the spread.
        'quantile' -- multi-quantile pinball over ``taus`` (default 0.1/0.5/0.9);
                      matches center AND spread -- the recommended, size-restoring choice.
    """
    if Y.dim() == 1:
        Y = Y.reshape(-1, 1)
    if mode == "none":
        return Y.new_zeros(())
    if mode == "mean":
        return F.mse_loss(Y_group.mean(dim=1), Y)
    qs = (0.5,) if mode == "median" else tuple(taus)
    q = torch.tensor(qs, device=Y.device, dtype=Y.dtype)
    pred = torch.quantile(Y_group, q, dim=1)           # (T, n, d)
    diff = Y.unsqueeze(0) - pred
    qv = q.view(-1, 1, 1)
    return torch.maximum(qv * diff, (qv - 1.0) * diff).mean()


# ----------------------------------------------------------------------------------
# train ONE conditional generator  target ~ P(target | Z)
# ----------------------------------------------------------------------------------
def train_conditional_generator(target, Z, cfg, device, seed=None):
    """Train a generator for P(target | Z) by minimizing conditional-MMD plus the
    multi-quantile alignment penalty. Returns the trained generator (in eval mode).

    ``epochs`` is a *cap*: if ``early_stop`` is on, training halts once a smoothed
    (EMA) training loss stops improving for ``patience`` epochs. This makes the run
    self-tune its length -- important because deeper generators need more epochs to
    converge, and an under-converged generator is exactly what inflates the size.
    """
    if seed is not None:
        torch.manual_seed(seed)
    n, out_dim = target.shape
    G = ConditionalGenerator(Z.shape[1], out_dim, cfg["noise_dim"],
                             width=cfg["width"], depth=cfg["depth"],
                             dropout=cfg["dropout"]).to(device)
    opt = optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.5, 0.999),
                     weight_decay=cfg["weight_decay"])

    sigma_z = _median_bandwidth(Z, p=1)
    sigma_t = _median_bandwidth(target, p=1)
    idx = torch.arange(n, device=device)
    bs = min(cfg["batch_size"], n)
    ema, best, wait = None, float("inf"), 0

    for epoch in range(cfg["epochs"]):
        perm = idx[torch.randperm(n, device=device)]
        epoch_loss, nb = 0.0, 0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            if b.numel() < 3:
                continue
            Zb, Tb = Z[b], target[b]
            m = Zb.shape[0]

            # generated draws for the conditional-MMD term
            Zr = Zb.repeat_interleave(cfg["M_train"], 0)
            fake = G(Zr, sample_noise(Zr.shape[0], cfg["noise_dim"], device,
                                      cfg["noise_var"], cfg["noise_kind"])
                     ).reshape(m, cfg["M_train"], out_dim)
            loss = conditional_mmd_loss(Tb, fake, Zb, sigma_z, sigma_t,
                                        cfg["mmd_w_laplacian"], cfg["mmd_w_gaussian"])

            # optional alignment penalty (mean / median / quantile / none)
            if cfg["align_mode"] != "none" and cfg["lambda_align"] > 0:
                Zq = Zb.repeat_interleave(cfg["align_samples"], 0)
                grp = G(Zq, sample_noise(Zq.shape[0], cfg["noise_dim"], device,
                                         cfg["noise_var"], cfg["noise_kind"])
                        ).reshape(m, cfg["align_samples"], out_dim)
                loss = loss + cfg["lambda_align"] * alignment_penalty(
                    Tb, grp, cfg["align_mode"], cfg["taus"])

            opt.zero_grad()
            loss.backward()
            if cfg["grad_clip"] is not None:
                nn.utils.clip_grad_norm_(G.parameters(), cfg["grad_clip"])
            opt.step()
            epoch_loss += float(loss.detach())
            nb += 1

        # early stopping on a smoothed training loss (after a warm-up)
        if cfg.get("early_stop", True) and nb > 0:
            epoch_loss /= nb
            ema = epoch_loss if ema is None else 0.85 * ema + 0.15 * epoch_loss
            if epoch >= cfg.get("min_epochs", 60):
                if ema < best - cfg.get("min_delta", 1e-4):
                    best, wait = ema, 0
                else:
                    wait += 1
                    if wait >= cfg.get("patience", 40):
                        break
    return G.eval()


@torch.no_grad()
def generate_conditional(G, Z, m, cfg, device):
    """Draw ``m`` samples of the target for each row of ``Z``. Returns (len(Z), m, d)."""
    Zr = Z.repeat_interleave(m, 0)
    out = G(Zr, sample_noise(Zr.shape[0], cfg["noise_dim"], device,
                             cfg["noise_var"], cfg["noise_kind"]))
    return out.reshape(Z.shape[0], m, -1)


# ----------------------------------------------------------------------------------
# test statistic (degenerate U-statistic) + wild bootstrap
# ----------------------------------------------------------------------------------
def _residual_gram(real, fake, sigma):
    """Û[i,j] = K(r_i,r_j) - mean_m K(r_i, f_j^m) - mean_m K(r_j, f_i^m)
                + mean_{m,m'} K(f_i^m, f_j^m'),  Laplacian kernel.
    ``real``:(n,d)  ``fake``:(n,M,d). Loops over M for the fake-fake term to stay
    memory-safe for the larger test-fold sizes."""
    n, M, _ = fake.shape
    Krr = torch.exp(-torch.cdist(real, real, p=1) / sigma)
    d_rf = _pairwise(real[:, None, None, :], fake[None, :, :, :], 1)  # (n,n,M)
    Krf = torch.exp(-d_rf / sigma).mean(2)
    Kfr = Krf.T
    Kff = torch.zeros(n, n, device=real.device)
    for mm in range(M):
        d = _pairwise(fake[:, mm, :][:, None, None, :], fake[None, :, :, :], 1)  # (n,n,M)
        Kff += torch.exp(-d / sigma).mean(2)
    Kff /= M
    return Krr - Krf - Kfr + Kff


def ci_statistic_and_bootstrap(X, Y, Z, X_fake, Y_fake, sigma_x, sigma_y, sigma_z,
                               n_boot=1000, rv="gaussian", device=None):
    """Return (statistic, bootstrap_null_samples) for one fold. The statistic is the
    degenerate U-statistic U_X * U_Y * K_Z averaged over off-diagonal pairs; the wild
    bootstrap injects independent multipliers e_i e_j on each pair."""
    n = X.shape[0]
    U = _residual_gram(X, X_fake, sigma_x)
    V = _residual_gram(Y, Y_fake, sigma_y)
    Kz = torch.exp(-torch.cdist(Z, Z, p=1) / sigma_z)
    FF = U * V * Kz * (1 - torch.eye(n, device=device))
    norm = 1.0 / (n * (n - 1))
    stat = norm * FF.sum()

    if rv == "rademacher":
        e = torch.sign(torch.randn(n, n_boot, device=device))
    else:
        e = torch.randn(n, n_boot, device=device)
    # boot[b] = norm * sum_{i,j} FF[i,j] e_i^b e_j^b  =  norm * e_b^T FF e_b
    boot = norm * torch.einsum("ib,ij,jb->b", e, FF, e)
    return stat.item(), boot.cpu().numpy()


# ----------------------------------------------------------------------------------
# full CI test on one dataset (cross-fitting, fresh generator per fold)
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# All tunables in one place. The trailing comment gives the matching name in the
# original notebooks' ``param`` dict, so nothing here should feel unfamiliar.
# ----------------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    # --- generator architecture ---
    depth=2,              # number of hidden layers: 1/2/3 == old skew_1 / skew_2 / skew_3
    width=1024,           # hidden width                                  (hidden_layer_size)
    noise_dim=5,          # latent noise dimension                        (noise_dimension)
    noise_var=1.0,        # latent noise variance                         (noise_dimension_var)
    noise_kind="normal",  # 'normal' or 'uniform'                         (noise_dimension_type)
    dropout=0.0,          # dropout prob in the generator                 (drop_out_p)

    # --- generator training  (this block is the main Type-I-error fix) ---
    lr=5e-4,              # Adam learning rate  (was 5e-5 -> too small)   (G_lr)
    epochs=600,           # CAP on epochs; early stopping usually ends earlier (epochs_num)
    batch_size=128,       #                                               (batch_size)
    grad_clip=None,       # max grad norm; None = off  (was 0.5 -> throttled learning)
    weight_decay=1e-5,    # Adam weight decay                             (wgt_decay)
    M_train=20,           # generated draws per z for the training MMD    (M_train)
    mmd_w_laplacian=1.0,  # weight on the Laplacian-kernel MMD term        (lambda_1)
    mmd_w_gaussian=1.0,   # weight on the Gaussian-kernel MMD term         (lambda_2)

    # --- early stopping (new; makes training length self-tune to the architecture) ---
    early_stop=True,      # halt when the smoothed training loss plateaus
    min_epochs=60,        # always train at least this many epochs first
    patience=40,          # stop after this many epochs with no improvement
    min_delta=1e-4,       # smallest loss drop that counts as improvement

    # --- alignment penalty  (extends the original median penalty) ---
    align_mode="quantile",   # 'none' | 'mean' | 'median' | 'quantile'
                             #   'median'   == the original penalty        (lambda_median term)
                             #   'mean'     == the tutorial's E[X|Z] MSE idea
                             #   'quantile' == spread-aware (recommended)
    lambda_align=0.5,        # penalty weight                             (lambda_median)
    taus=(0.1, 0.5, 0.9),    # quantiles used when align_mode='quantile'
    align_samples=64,        # generated draws per z for the penalty      (median_samples)

    # --- test statistic / wild bootstrap ---
    n_folds=2,            # cross-fitting folds J                         (k_value)
    M_test=100,           # generated draws per z for the statistic       (m_value)
    n_boot=1000,          # wild-bootstrap replicates B                   (j_value)
    boot_rv="gaussian",   # bootstrap multiplier: 'gaussian' or 'rademacher' (boor_rv_type)

    # --- data handling ---
    standardize=True,     # z-score X,Y,Z per fold  (original preprocess='scale' was ignored)
)
# NOTE. A few original knobs were intentionally dropped as negligible or superseded:
#   lambda_3 / lambda_4 (tiny 1e-5 L1 penalties) -> folded into weight_decay;
#   using_Gen '1'/'2' and is_sparse/sparse_ratio -> replaced by the single 'depth' knob;
#   normal_ini -> default PyTorch initialization.
# Data-generating knobs (sample_size, dx, dy, z_dim, eps_std, dist_z, alpha_x) are passed
# to run_experiment via n / data_kwargs; see that function.


def run_ci_test(X, Y, Z, config=None, oracle=False, device=None, seed=0,
                dgp="skew", dgp_kwargs=None):
    """Cross-fitted CI test on a single dataset. Returns the bootstrap p-value.

    A fresh generator is trained on each fold's training data (no warm-start leakage);
    if ``oracle=True`` the true conditionals of ``dgp`` are used instead of learned
    generators, which should sit at the nominal level and serves as a size sanity-check.
    ``dgp`` is only consulted in oracle mode (to draw from the matching true conditionals).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    dgp = get_dgp(dgp)
    dgp_kwargs = dgp_kwargs or {}
    device = device or get_device()
    X, Y, Z = X.to(device), Y.to(device), Z.to(device)
    n = X.shape[0]
    folds = np.array_split(np.random.permutation(n), cfg["n_folds"])

    stats, boots = [], []
    for j, test_idx in enumerate(folds):
        test_idx = torch.as_tensor(test_idx, device=device)
        train_mask = torch.ones(n, dtype=torch.bool, device=device)
        train_mask[test_idx] = False
        Xtr, Ytr, Ztr = X[train_mask], Y[train_mask], Z[train_mask]
        Xte, Yte, Zte = X[test_idx], Y[test_idx], Z[test_idx]

        # standardize using the TRAINING fold only
        if cfg["standardize"]:
            sx, sy, sz = Scaler(Xtr), Scaler(Ytr), Scaler(Ztr)
            Xtr, Ytr, Ztr = sx.transform(Xtr), sy.transform(Ytr), sz.transform(Ztr)
            Xte, Yte, Zte = sx.transform(Xte), sy.transform(Yte), sz.transform(Zte)

        nte = Xte.shape[0]
        if oracle:
            # true conditionals of this DGP, standardized with the same scaler
            Z_orig = (Zte * sz.std + sz.mean) if cfg["standardize"] else Zte
            Xf, Yf = dgp.oracle(Z_orig, cfg["M_test"], device=device, **dgp_kwargs)
            if cfg["standardize"]:
                Xf = (Xf - sx.mean.unsqueeze(1)) / sx.std.unsqueeze(1)
                Yf = (Yf - sy.mean.unsqueeze(1)) / sy.std.unsqueeze(1)
        else:
            Gx = train_conditional_generator(Xtr, Ztr, cfg, device, seed=seed * 100 + j)
            Gy = train_conditional_generator(Ytr, Ztr, cfg, device, seed=seed * 100 + 50 + j)
            Xf = generate_conditional(Gx, Zte, cfg["M_test"], cfg, device)
            Yf = generate_conditional(Gy, Zte, cfg["M_test"], cfg, device)

        # median-heuristic bandwidths on the test fold (in the space used for the stat)
        sig_x = _median_bandwidth(Xte, 1)
        sig_y = _median_bandwidth(Yte, 1)
        sig_z = _median_bandwidth(Zte, 1)

        stat, boot = ci_statistic_and_bootstrap(
            Xte, Yte, Zte, Xf, Yf, sig_x, sig_y, sig_z,
            n_boot=cfg["n_boot"], rv=cfg["boot_rv"], device=device)
        stats.append(stat)
        boots.append(boot)

    T = float(np.mean(stats))
    T_boot = np.mean(np.stack(boots, 0), axis=0)
    return float(np.mean(T_boot > T))


# ----------------------------------------------------------------------------------
# Monte-Carlo experiment (rejection rate)
# ----------------------------------------------------------------------------------
def _one_replicate(seed, n, hypothesis, config, oracle, data_kwargs, prefer_gpu, dgp):
    set_seed(seed)
    device = get_device(prefer_gpu)
    X, Y, Z = get_dgp(dgp).sample(n, hypothesis=hypothesis, device=device, **data_kwargs)
    return run_ci_test(X, Y, Z, config=config, oracle=oracle, device=device, seed=seed,
                       dgp=dgp, dgp_kwargs=data_kwargs)


def run_experiment(n=400, hypothesis="H0", n_rep=200, config=None, oracle=False,
                   levels=(0.10, 0.05), data_kwargs=None, dgp="skew", n_jobs=1,
                   prefer_gpu=True, verbose=True):
    """Estimate the rejection rate over ``n_rep`` datasets. Under H0 this is the
    empirical size; under H1 (with ``alpha_x`` in ``data_kwargs``) it is the power.

    Parameters mapped to the original notebooks' ``param`` dict:
        n           sample size per dataset                    (sample_size)
        hypothesis  'H0' (size) or 'H1' (power)                (test: 'type1error'/'power')
        n_rep       number of Monte-Carlo datasets             (n_test)
        levels      significance levels to report              (alpha, alpha1)
        oracle      use the true conditionals                  (using_orcale)
        config      the model/test settings (see DEFAULT_CONFIG)
        dgp         which data-generating model: a name in DGPS ('skew', 'gaussian')
                    or a DGP object you registered yourself
        data_kwargs extra knobs forwarded to the chosen DGP, e.g. for 'skew':
                        dx, dy      dims of X, Y                (dx, dy)
                        dz          dim of Z                    (z_dim)
                        nstd        noise scale                 (eps_std)
                        dist_z      'gaussian' or 'laplace'     (dist_z)
                        alpha_x     H1 dependence strength      (alpha_x)
                    and for 'gaussian': dz, alpha_x.

    Set ``n_jobs>1`` (or -1 for all cores) for CPU parallelism via joblib; on a GPU
    leave ``n_jobs=1`` since each replicate already uses the GPU. Returns a dict with
    the rejection rate at each level and the raw p-values.
    """
    data_kwargs = data_kwargs or {}
    if n_jobs == 1:
        pvals = [_one_replicate(s, n, hypothesis, config, oracle, data_kwargs, prefer_gpu, dgp)
                 for s in range(n_rep)]
    else:
        from joblib import Parallel, delayed
        pvals = Parallel(n_jobs=n_jobs)(
            delayed(_one_replicate)(s, n, hypothesis, config, oracle, data_kwargs, False, dgp)
            for s in range(n_rep))
    pvals = np.asarray(pvals)
    rej = {lvl: float(np.mean(pvals < lvl)) for lvl in levels}
    if verbose:
        gtag = get_dgp(dgp).name
        mtag = "ORACLE" if oracle else f"depth={ (config or {}).get('depth', DEFAULT_CONFIG['depth']) }"
        print(f"[{hypothesis} dgp={gtag} {mtag} n={n} reps={n_rep}] "
              + "  ".join(f"rej@{lvl:.2f}={rej[lvl]:.3f}" for lvl in levels))
    return dict(rejection=rej, pvalues=pvals)
