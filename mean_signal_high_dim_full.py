# -*- coding: utf-8 -*-
"""
High-dimensional mean-signal conditional independence simulation with mean alignment.

Main design:
1. Z has dimension 50 or higher, e.g. d_z = 100.
2. X|Z and Y|Z have strong nonlinear conditional means mX(Z), mY(Z).
3. Under H0, X and Y are conditionally independent given Z.
4. Under H1, X and Y share a residual component, so dependence is in the residual mean/coupling signal.
5. The marginal conditional laws X|Z and Y|Z are unchanged under H1, so the generator target remains valid.
6. Mean alignment directly helps learn E[X|Z] and E[Y|Z].
7. Cross-fitting is strict: generators are reinitialized and retrained for every fold.
8. The training CMMD loss is normalized by n * (n - 1), matching the U-statistic scaling.
"""

import os
import math
import functools
import multiprocessing
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as TD
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed


# =========================
# Parameters
# =========================

param = {
    "test": "type1error",           # ['type1error', 'power']
    "sample_size": 1000,             # use 2000 for formal runs
    "batch_size": 256,
    "z_dim": 100,                    # recommended grid: [50, 100, 200]
    "dx": 1,
    "dy": 1,
    "n_test": 100,                   # increase to 500/1000 for formal runs
    "epochs_num": 200,
    "eps_std": 0.4,                  # conditional noise std for mean-signal DGP
    "dist_z": "gaussian",            # Section 4.2 uses Gaussian Z
    "alpha_x": 0.08,                 # H1 residual correlation rho; try [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    "m_value": 100,                  # M synthetic samples for test statistic
    "k_value": 2,                    # Section 4.2 uses T_hat_2, i.e. J = 2 folds
    "j_value": 1000,                 # bootstrap replicates B
    "noise_dimension": 50,
    "hidden_layer_size": 512,
    "normal_ini": False,
    "preprocess": "None",            # do not normalize this DGP
    "G_lr": 2e-5,
    "alpha": 0.1,
    "alpha1": 0.05,
    "set_seeds": 42,
    "using_orcale": False,           # kept spelling for compatibility with your code
    "lambda_1": 1.0,                 # Laplace-kernel MMD loss weight
    "lambda_2": 0.0,                 # Gaussian-kernel MMD loss not used
    "using_Gen": "1",                # ['1', '2']
    "boor_rv_type": "gaussian",      # ['rademacher', 'gaussian']
    "wgt_decay": 1e-5,
    "lambda_3": 1e-5,                # L1 regularization for non-first layers
    "lambda_4": 2e-5,                # L1 regularization for first layer
    "drop_out_p": 0.2,
    "M_train": 100,
    "lambda_mean": 0.3,
    "mean_samples": 100,
    "workers_per_gpu": 2,            # average allocation: total workers = num_gpus * workers_per_gpu
    "enable_cuda": True,
}


# This global variable is set inside each parallel process by set_process_device().
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_available_gpus(enable_cuda=True):
    """Return a list of visible CUDA GPU ids."""
    if enable_cuda and torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def set_process_device(gpu_id=None, enable_cuda=True):
    """Set the global torch device inside a joblib worker process."""
    global device
    if enable_cuda and torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    return device


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Data generation: high-dimensional mean-signal DGP
# =========================


def _current_device():
    """Return the global torch device, with a safe fallback."""
    if "device" in globals():
        return device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_mean_signal_projection(dz, k=4, rng=None):
    """
    Generate k orthonormal projection directions in R^dz.

    The high-dimensional signal is low-rank but nonlinear:
        S_j = a_j^T Z, j = 1,...,4.

    Parameters
    ----------
    dz : int
        Dimension of Z. Use dz >= 50 for the high-dimensional experiment.
    k : int
        Number of projection directions. This DGP uses k = 4.
    rng : np.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    P : torch.FloatTensor, shape (dz, k)
        Orthonormal projection matrix.
    """
    if dz < k:
        raise ValueError(f"dz must be at least {k}, but got dz={dz}.")

    rng = np.random.default_rng() if rng is None else rng
    A = rng.normal(0.0, 1.0, size=(dz, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return torch.from_numpy(Q[:, :k].astype(np.float32)).float()


# Backward-compatible name used in the original code.
def make_l1_normalized_projection(dz, out_dim=1, rng=None):
    """
    Compatibility wrapper.

    The original Section 4.2 code used this name for one-dimensional random
    projections. The new mean-signal DGP needs four projection directions, so
    this wrapper ignores out_dim and returns P with shape (dz, 4).
    """
    return make_mean_signal_projection(dz=dz, k=4, rng=rng)


def mean_signal_functions(P, Z):
    """
    Compute nonlinear conditional means mX(Z) and mY(Z).

    DGP:
        S_j = a_j^T Z
        mY(Z) = 1.0 sin(S1) + 0.7 S2 + 0.4 tanh(S1 S2)
        mX(Z) = 1.0 cos(S1) + 0.7 S3 + 0.4 tanh(S1 + S4)

    Parameters
    ----------
    P : torch.Tensor, shape (dz, 4)
        Projection directions.
    Z : torch.Tensor, shape (n, dz)
        Conditioning variable.

    Returns
    -------
    mX, mY : torch.Tensor, shape (n, 1)
    """
    if not torch.is_tensor(P):
        P = torch.tensor(P, dtype=Z.dtype, device=Z.device)
    else:
        P = P.to(device=Z.device, dtype=Z.dtype)

    if P.dim() != 2 or P.shape[1] < 4:
        raise ValueError(
            "P must have shape (dz, at least 4). "
            "Use make_mean_signal_projection(dz, k=4, rng=rng)."
        )

    S = Z @ P[:, :4]
    s1 = S[:, [0]]
    s2 = S[:, [1]]
    s3 = S[:, [2]]
    s4 = S[:, [3]]

    mY = 1.0 * torch.sin(s1) + 0.7 * s2 + 0.4 * torch.tanh(s1 * s2)
    mX = 1.0 * torch.cos(s1) + 0.7 * s3 + 0.4 * torch.tanh(s1 + s4)

    return mX.float(), mY.float()


def generate_samples_mean_signal(
    P,
    size=1000,
    sType="CI",
    dx=1,
    dy=1,
    dz=100,
    nstd=0.4,
    alpha_x=0.05,
    preprocess="None",
    dist_z="gaussian",
):
    """
    High-dimensional DGP designed for mean alignment.

    Null H0:
        Z ~ N(0, I_dz)
        Y = mY(Z) + sigma * eps_y
        X = mX(Z) + sigma * eps_x
        eps_x independent of eps_y
        Therefore X independent of Y conditional on Z.

    Alternative H1:
        Y = mY(Z) + sigma * eps_y
        X = mX(Z) + sigma * (sqrt(1-rho^2) eps_x + rho eps_y)
        rho = alpha_x

    Under both H0 and H1:
        E[X | Z] = mX(Z), E[Y | Z] = mY(Z).

    The alternative keeps the two marginal conditional distributions X|Z and Y|Z
    fixed, but introduces conditional dependence through a shared residual. This
    is useful for checking whether better conditional mean learning improves CI
    testing power.
    """
    if dx != 1 or dy != 1:
        raise ValueError("This mean-signal DGP currently expects dx = dy = 1.")

    if preprocess not in ["None", None]:
        raise ValueError("For this mean-signal DGP, set preprocess='None'.")

    if dist_z == "gaussian":
        Z = torch.randn(size, dz).float()
    elif dist_z == "laplace":
        laplace = torch.distributions.Laplace(torch.tensor(0.0), torch.tensor(1.0))
        Z = laplace.sample((size, dz)).float()
    else:
        raise ValueError("dist_z must be 'gaussian' or 'laplace'.")

    P = P.float()
    mX, mY = mean_signal_functions(P, Z)

    eps_x = torch.randn(size, 1).float()
    eps_y = torch.randn(size, 1).float()

    Y = mY + float(nstd) * eps_y

    is_null = sType in ["CI", "type1error", "H0", "null"]
    if is_null:
        X = mX + float(nstd) * eps_x
    else:
        rho = float(alpha_x)
        rho = max(min(rho, 0.99), 0.0)
        shared_residual = np.sqrt(1.0 - rho ** 2) * eps_x + rho * eps_y
        X = mX + float(nstd) * shared_residual

    return X.float(), Y.float(), Z.float()


def generate_samples_from_fixed_Z_mean_signal(
    P,
    Z,
    size=100,
    sType="CI",
    dx=1,
    dy=1,
    dz=100,
    nstd=0.4,
    alpha_x=0.05,
):
    """
    Oracle generator for fixed Z under the mean-signal DGP.

    It must generate conditionally independent fake X and fake Y given Z, even
    under H1. For this DGP the marginal conditional laws are the same under H0
    and H1:
        X | Z ~ N(mX(Z), sigma^2)
        Y | Z ~ N(mY(Z), sigma^2)
    so alpha_x is intentionally ignored here.
    """
    if dx != 1 or dy != 1:
        raise ValueError("This mean-signal oracle currently expects dx = dy = 1.")

    dev = _current_device()
    Z = Z.to(dev).float()
    if torch.is_tensor(P):
        P = P.to(dev).float()
    else:
        P = torch.tensor(P, device=dev).float()

    n_mc = Z.shape[0]
    mX, mY = mean_signal_functions(P, Z)

    eps_x = torch.randn(n_mc, dx, device=dev).float()
    eps_y = torch.randn(n_mc, dy, device=dev).float()

    X_fake = mX + float(nstd) * eps_x
    Y_fake = mY + float(nstd) * eps_y

    return X_fake.float(), Y_fake.float()


# ============================================================
# Backward-compatible wrappers used by the existing mGAN code
# ============================================================


def generate_samples_random(
    Ax,
    Ay,
    size=1000,
    sType="CI",
    dx=1,
    dy=1,
    dz=20,
    nstd=0.4,
    alpha_x=0.05,
    preprocess="None",
    dist_z="gaussian",
):
    """
    Compatibility wrapper.

    The original mGAN() calls generate_samples_random(Ax, Ay, ...). For the new
    DGP, Ax is treated as the projection matrix P. Ay is kept only for interface
    compatibility and is not used.
    """
    P = Ax if Ax is not None else Ay
    return generate_samples_mean_signal(
        P=P,
        size=size,
        sType=sType,
        dx=dx,
        dy=dy,
        dz=dz,
        nstd=nstd,
        alpha_x=alpha_x,
        preprocess=preprocess,
        dist_z=dist_z,
    )


def generate_samples_from_fixed_Z_random(
    Ax,
    Ay,
    Z,
    size=100,
    sType="CI",
    dx=1,
    dy=1,
    dz=20,
    nstd=0.4,
    alpha_x=0.05,
    normalize=False,
    seed=None,
    dist_z="gaussian",
):
    """
    Compatibility oracle wrapper.

    The original mGAN() oracle path calls generate_samples_from_fixed_Z_random.
    For the new DGP, Ax is treated as P and Ay is unused.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    P = Ax if Ax is not None else Ay
    return generate_samples_from_fixed_Z_mean_signal(
        P=P,
        Z=Z,
        size=size,
        sType=sType,
        dx=dx,
        dy=dy,
        dz=dz,
        nstd=nstd,
        alpha_x=alpha_x,
    )


# =========================
# Dataset classes
# =========================


class DatasetSelect(Dataset):
    def __init__(self, X, Y, Z):
        self.X_real = X
        self.Y_real = Y
        self.Z_real = Z
        self.sample_size = X.shape[0]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        return self.X_real[index], self.Y_real[index], self.Z_real[index]


class DatasetSelect_GAN(Dataset):
    def __init__(self, X, Y, Z, batch_size):
        self.X_real = X
        self.Y_real = Y
        self.Z_real = Z
        self.batch_size = batch_size
        self.sample_size = X.shape[0]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        # Z_fake is kept only for compatibility with your old loop signature.
        return (
            self.X_real[index],
            self.Y_real[index],
            self.Z_real[index],
            self.Z_real[(self.batch_size + index) % self.sample_size],
        )


# =========================
# Utilities
# =========================


def sample_noise(sample_size, noise_dimension, noise_type="normal", input_var=1.0 / 3.0):
    if noise_type == "normal":
        noise_generator = TD.MultivariateNormal(
            torch.zeros(noise_dimension, device=device),
            input_var * torch.eye(noise_dimension, device=device),
        )
        return noise_generator.sample((sample_size,))
    if noise_type == "unif":
        return torch.rand(sample_size, noise_dimension, device=device)
    if noise_type == "Cauchy":
        return TD.Cauchy(
            torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
        ).sample((sample_size, noise_dimension)).squeeze(-1)
    raise ValueError("noise_type must be 'normal', 'unif', or 'Cauchy'.")


def safe_median_bandwidth(distance_matrix, eps=1e-6):
    sigma = torch.median(distance_matrix).detach()
    if not torch.isfinite(sigma) or sigma.item() <= eps:
        sigma = torch.tensor(eps, device=distance_matrix.device)
    return sigma.item()


def pairwise_l1(a, b):
    """Pairwise L1 distances. a: (n,d), b: (m,d). Returns (n,m)."""
    return torch.cdist(a, b, p=1)


def laplace_kernel_pairwise(a, b, sigma):
    sigma = max(float(sigma), 1e-6)
    return torch.exp(-pairwise_l1(a, b) / sigma)


# =========================
# Generator architectures
# =========================


class Generator(nn.Module):
    """
    Fully-connected generator.
    Fixed: no sigmoid at the output, because Section 4.2 X and Y lie in [-1, 1].
    """

    def __init__(self, input_dimension, output_dimension, noise_dimension, hidden_layer_size,
                 BN_type=False, ReLU_coef=0.1, drop_out_p=0.2, drop_input=False):
        super().__init__()
        self.BN_type = BN_type
        self.drop_input = drop_input
        self.fc1 = nn.Linear(input_dimension + noise_dimension, hidden_layer_size, bias=True)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size, bias=True)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size, bias=True)
        self.fc_last = nn.Linear(hidden_layer_size, output_dimension, bias=True)
        self.leakyReLU1 = nn.LeakyReLU(ReLU_coef)
        self.drop_out0 = nn.Dropout(p=drop_out_p)
        self.drop_out1 = nn.Dropout(p=drop_out_p)
        self.drop_out2 = nn.Dropout(p=drop_out_p)
        self.drop_out3 = nn.Dropout(p=drop_out_p)
        if BN_type:
            self.BN1 = nn.BatchNorm1d(hidden_layer_size, momentum=0.8, affine=False)
            self.BN2 = nn.BatchNorm1d(hidden_layer_size, momentum=0.8, affine=False)
            self.BN3 = nn.BatchNorm1d(hidden_layer_size, momentum=0.8, affine=False)

    def forward(self, x):
        if self.drop_input:
            x = self.drop_out0(x)
        if self.BN_type:
            x = self.drop_out1(self.leakyReLU1(self.BN1(self.fc1(x))))
            x = self.drop_out2(self.leakyReLU1(self.BN2(self.fc2(x))))
            # Keep the same effective depth as your current code.
            x = self.fc_last(x)
        else:
            x = self.drop_out1(self.leakyReLU1(self.fc1(x)))
            x = self.drop_out2(self.leakyReLU1(self.fc2(x)))
            # Fixed: no sigmoid output.
            x = self.fc_last(x)
        return x


class NonFullyConnected_1(nn.Module):
    def __init__(self, size_in, size_out, m, bias=True):
        super().__init__()
        self.linear = nn.Linear(m * size_in, m * size_out, bias=bias)
        self.register_buffer("mask", functools.reduce(torch.block_diag, [torch.ones(size_out, size_in) for _ in range(m)]))

    def forward(self, x):
        self.linear.weight.data *= self.mask
        return self.linear(x)


class Generator_2(nn.Module):
    def __init__(self, input_dimension, output_dimension, noise_dimension, hidden_layer_size,
                 BN_type=False, ReLU_coef=0.1, hidden_layer_depth=1, ntargets_k=5):
        super().__init__()
        self.input_dimension = input_dimension + noise_dimension
        self.output_dimension = output_dimension
        self.ntargets_k = ntargets_k
        self.hidden_layer_sizes = [hidden_layer_size] * hidden_layer_depth
        self.BN_type = BN_type
        self.leakyrelu = nn.LeakyReLU(ReLU_coef)
        self.linear_layers_from_input = nn.Linear(self.input_dimension, ntargets_k * self.hidden_layer_sizes[0])
        self.linear_layers_between = nn.ModuleList([
            NonFullyConnected_1(self.hidden_layer_sizes[0], self.hidden_layer_sizes[0], ntargets_k)
            for _ in range(len(self.hidden_layer_sizes))
        ])
        self.linear8 = nn.Linear(self.hidden_layer_sizes[0] * ntargets_k, self.output_dimension)
        if BN_type:
            self.BN1 = nn.BatchNorm1d(hidden_layer_size, momentum=0.8, affine=False)

    def forward(self, input_tensor):
        output = self.linear_layers_from_input(input_tensor)
        output = self.leakyrelu(output)
        for linear_layers_between in self.linear_layers_between:
            output = linear_layers_between(output)
            output = self.leakyrelu(output)
        return self.linear8(output)


# =========================
# MMD loss and test statistic
# =========================


def find_loss(y_torch, gen_y_all_torch, z_torch, sigma_w, sigma_u, M):
    """
    Conditional MMD loss with Laplace kernels.

    y_torch:          (B, d)
    gen_y_all_torch:  (B, M, d) or (B, M) for d = 1
    z_torch:          (B, dz)
    """
    if gen_y_all_torch.dim() == 2:
        gen_y_all_torch = gen_y_all_torch.unsqueeze(-1)
    if y_torch.dim() == 1:
        y_torch = y_torch.reshape(-1, 1)

    n = z_torch.shape[0]
    sigma_w = max(float(sigma_w), 1e-6)
    sigma_u = max(float(sigma_u), 1e-6)

    w_mx = torch.exp(-pairwise_l1(z_torch, z_torch) / sigma_w)

    u_mx_1 = torch.exp(-pairwise_l1(y_torch, y_torch) / sigma_u)

    # u_mx_2[i,j] = mean_m K(y_i, generated_y_jm)
    dist_rg = torch.sum(torch.abs(y_torch[:, None, None, :] - gen_y_all_torch[None, :, :, :]), dim=-1)
    u_mx_2 = torch.mean(torch.exp(-dist_rg / sigma_u), dim=2)
    u_mx_3 = u_mx_2.T

    # u_mx_4[i,j] = mean_{m1,m2} K(generated_y_im1, generated_y_jm2)
    u_mx_4 = torch.zeros(n, n, device=device)
    for m1 in range(M):
        dist_gg = torch.sum(
            torch.abs(gen_y_all_torch[:, m1, :][:, None, None, :] - gen_y_all_torch[None, :, :, :]),
            dim=-1,
        )
        u_mx_4 += torch.mean(torch.exp(-dist_gg / sigma_u), dim=2)
    u_mx_4 = u_mx_4 / M

    u_mx = u_mx_1 - u_mx_2 - u_mx_3 + u_mx_4
    eye = torch.eye(n, device=device)
    FF_mx = u_mx * w_mx * (1.0 - eye)
    if n <= 1:
        return torch.sum(FF_mx) * 0.0
    loss = torch.sum(FF_mx) / (n * (n - 1))
    return loss


def compute_residual_kernel_matrix(real, gen, sigma, chunk_size=64):
    """
    Compute U or V residual kernel matrix.

    real: (n, d)
    gen:  (n, M, d)
    return: (n, n)
    """
    if real.dim() == 1:
        real = real.reshape(-1, 1)
    if gen.dim() == 2:
        gen = gen.unsqueeze(-1)

    n, M, d = gen.shape
    sigma = max(float(sigma), 1e-6)

    rr = torch.exp(-pairwise_l1(real, real) / sigma)

    # rg[i,j] = mean_m K(real_i, gen_jm)
    dist_rg = torch.sum(torch.abs(real[:, None, None, :] - gen[None, :, :, :]), dim=-1)
    rg = torch.mean(torch.exp(-dist_rg / sigma), dim=2)
    gr = rg.T

    # gg[i,j] = mean_{m1,m2} K(gen_im1, gen_jm2). Chunked to reduce memory.
    gg = torch.zeros(n, n, device=device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c = end - start
        gg_chunk = torch.zeros(c, n, device=device)
        gen_chunk = gen[start:end]
        for m1 in range(M):
            dist = torch.sum(
                torch.abs(gen_chunk[:, m1, :][:, None, None, :] - gen[None, :, :, :]),
                dim=-1,
            )
            gg_chunk += torch.mean(torch.exp(-dist / sigma), dim=2)
        gg[start:end] = gg_chunk / M

    return rr - rg - gr + gg


def get_p_value_stat_1(
    boot_num,
    M,
    n,
    gen_x_all_torch,
    gen_y_all_torch,
    x_torch,
    y_torch,
    z_torch,
    sigma_w,
    sigma_u=1,
    sigma_v=1,
    boor_rv_type="gaussian",
):
    """Compute statistic and bootstrap statistics for one test fold."""
    if x_torch.dim() == 1:
        x_torch = x_torch.reshape(-1, 1)
    if y_torch.dim() == 1:
        y_torch = y_torch.reshape(-1, 1)
    if gen_x_all_torch.dim() == 2:
        gen_x_all_torch = gen_x_all_torch.unsqueeze(-1)
    if gen_y_all_torch.dim() == 2:
        gen_y_all_torch = gen_y_all_torch.unsqueeze(-1)

    sigma_w = max(float(sigma_w), 1e-6)
    w_mx = torch.exp(-pairwise_l1(z_torch, z_torch) / sigma_w)

    # Original code names: u for Y, v for X.
    u_mx = compute_residual_kernel_matrix(y_torch, gen_y_all_torch, sigma_u)
    v_mx = compute_residual_kernel_matrix(x_torch, gen_x_all_torch, sigma_v)

    eye = torch.eye(n, device=device)
    FF_mx = u_mx * v_mx * w_mx * (1.0 - eye)
    stat = torch.sum(FF_mx).item() / (n * (n - 1))

    if boor_rv_type == "rademacher":
        eboot = torch.sign(torch.randn(n, boot_num, device=device))
    elif boor_rv_type == "gaussian":
        eboot = torch.randn(n, boot_num, device=device)
    else:
        raise ValueError("boor_rv_type must be 'rademacher' or 'gaussian'.")

    # Vectorized wild bootstrap: stat_b = e_b^T FF e_b / (n - 1)
    boot_vals = torch.sum(eboot * (FF_mx @ eboot), dim=0) / (n * (n - 1))
    boottemp = boot_vals.detach().cpu().numpy()
    return stat, boottemp


# =========================
# Training procedure with mean alignment
# =========================


def train_ver3(
    G_zx,
    G_zy,
    X,
    Y,
    Z,
    X_test,
    Y_test,
    Z_test,
    noise_dimension,
    noise_type,
    G_lr,
    hidden_layer_size,
    DataLoader,
    BN_type,
    ReLU_coef,
    lambda_mean=0.5,
    mean_samples=20,
    epochs_num=50,
    patience=20,
    min_delta=1e-5,
    sigma_z=1,
    sigma_x=1,
    sigma_y=1,
    normal_ini=False,
    lambda_1=1,
    lambda_2=0,
    using_Gen="1",
    wgt_decay=0,
    lambda_3=0,
    lambda_4=0,
    drop_out_p=0.2,
    M_train=3,
):
    """Train G_zx and G_zy using MMD loss + mean alignment."""
    input_dimension = Z.shape[1]
    output_dimension_y = Y.shape[1]
    output_dimension_x = X.shape[1]

    if G_zy is None or G_zx is None:
        if using_Gen == "1":
            G_zy = Generator(input_dimension, output_dimension_y, noise_dimension, hidden_layer_size,
                             BN_type, ReLU_coef, drop_out_p).to(device)
            G_zx = Generator(input_dimension, output_dimension_x, noise_dimension, hidden_layer_size,
                             BN_type, ReLU_coef, drop_out_p).to(device)
        elif using_Gen == "2":
            G_zy = Generator_2(input_dimension, output_dimension_y, noise_dimension, hidden_layer_size,
                               BN_type, ReLU_coef).to(device)
            G_zx = Generator_2(input_dimension, output_dimension_x, noise_dimension, hidden_layer_size,
                               BN_type, ReLU_coef).to(device)
        else:
            raise ValueError("using_Gen must be '1' or '2'.")

        if normal_ini:
            for p in G_zy.parameters():
                p.data = torch.randn(p.shape, device=device) / np.sqrt(float(hidden_layer_size * 2))
            for p in G_zx.parameters():
                p.data = torch.randn(p.shape, device=device) / np.sqrt(float(hidden_layer_size * 2))

    G_zy_solver = optim.Adam(G_zy.parameters(), lr=G_lr, betas=(0.5, 0.999), weight_decay=wgt_decay)
    G_zx_solver = optim.Adam(G_zx.parameters(), lr=G_lr, betas=(0.5, 0.999), weight_decay=wgt_decay)

    best_loss = float("inf")
    counter = 0

    for epoch in range(epochs_num):
        G_zy.train()
        G_zx.train()
        last_epoch_loss = None

        for X_real, Y_real, Z_real, Z_fake in DataLoader:
            X_real = X_real.to(device)
            Y_real = Y_real.to(device)
            Z_real = Z_real.to(device)
            batch_size = Z_real.shape[0]

            # -------------------------
            # Mean alignment
            # -------------------------
            Z_repeated_mean = Z_real.repeat_interleave(mean_samples, dim=0)

            Noise_mean_y = sample_noise(Z_repeated_mean.shape[0], noise_dimension, noise_type, input_var=1.0 / 3.0)
            Y_generated_group = G_zy(torch.cat((Z_repeated_mean, Noise_mean_y), dim=1))
            Y_mean_pred = torch.mean(Y_generated_group.reshape(batch_size, mean_samples, -1), dim=1)
            loss_mean_y = nn.functional.mse_loss(Y_mean_pred, Y_real)

            Noise_mean_x = sample_noise(Z_repeated_mean.shape[0], noise_dimension, noise_type, input_var=1.0 / 3.0)
            X_generated_group = G_zx(torch.cat((Z_repeated_mean, Noise_mean_x), dim=1))
            X_mean_pred = torch.mean(X_generated_group.reshape(batch_size, mean_samples, -1), dim=1)
            loss_mean_x = nn.functional.mse_loss(X_mean_pred, X_real)

            # -------------------------
            # Conditional MMD loss
            # Fixed: repeat_interleave, not repeat.
            # -------------------------
            Z_real_repeat = Z_real.repeat_interleave(M_train, dim=0)

            Noise_fake_y = sample_noise(Z_real_repeat.shape[0], noise_dimension, noise_type, input_var=1.0 / 3.0)
            Y_fake = G_zy(torch.cat((Z_real_repeat, Noise_fake_y), dim=1))
            Y_fake = Y_fake.reshape(batch_size, M_train, -1)

            Noise_fake_x = sample_noise(Z_real_repeat.shape[0], noise_dimension, noise_type, input_var=1.0 / 3.0)
            X_fake = G_zx(torch.cat((Z_real_repeat, Noise_fake_x), dim=1))
            X_fake = X_fake.reshape(batch_size, M_train, -1)

            # Generator step for Y
            G_zy_solver.zero_grad()
            l1_first_y = torch.tensor(0.0, device=device)
            l1_rest_y = torch.tensor(0.0, device=device)
            for name, par in G_zy.named_parameters():
                if "fc1" in name:
                    l1_first_y = l1_first_y + torch.linalg.vector_norm(par, ord=1)
                else:
                    l1_rest_y = l1_rest_y + torch.linalg.vector_norm(par, ord=1)

            mmd_loss_y = (
                lambda_1 * find_loss(Y_real, Y_fake, Z_real, sigma_z, sigma_y, M_train)
                + lambda_3 * l1_rest_y
                + lambda_4 * l1_first_y
            )
            g_zy_error = mmd_loss_y + lambda_mean * loss_mean_y
            g_zy_error.backward()
            torch.nn.utils.clip_grad_norm_(G_zy.parameters(), max_norm=0.5)
            G_zy_solver.step()

            # Generator step for X
            G_zx_solver.zero_grad()
            l1_first_x = torch.tensor(0.0, device=device)
            l1_rest_x = torch.tensor(0.0, device=device)
            for name, par in G_zx.named_parameters():
                if "fc1" in name:
                    l1_first_x = l1_first_x + torch.linalg.vector_norm(par, ord=1)
                else:
                    l1_rest_x = l1_rest_x + torch.linalg.vector_norm(par, ord=1)

            mmd_loss_x = (
                lambda_1 * find_loss(X_real, X_fake, Z_real, sigma_z, sigma_x, M_train)
                + lambda_3 * l1_rest_x
                + lambda_4 * l1_first_x
            )
            g_zx_error = mmd_loss_x + lambda_mean * loss_mean_x
            g_zx_error.backward()
            torch.nn.utils.clip_grad_norm_(G_zx.parameters(), max_norm=0.5)
            G_zx_solver.step()

            last_epoch_loss = (g_zx_error.detach() + g_zy_error.detach()).item()

        if last_epoch_loss is not None:
            if last_epoch_loss < best_loss - min_delta:
                best_loss = last_epoch_loss
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                break

    return G_zy, G_zx


# =========================
# Main statistic function
# =========================


def mGAN(
    Ax,
    Ay,
    n=2000,
    z_dim=100,
    simulation="type1error",
    batch_size=64,
    epochs_num=1000,
    nstd=0.5,
    z_dist="gaussian",
    x_dims=1,
    y_dims=1,
    a_x=0.05,
    M=100,
    k=2,
    boot_num=1000,
    noise_dimension=10,
    hidden_layer_size=512,
    normal_ini=False,
    preprocess="None",
    G_lr=1e-5,
    using_orcale=False,
    lambda_1=1,
    lambda_2=0,
    using_Gen="1",
    boor_rv_type="gaussian",
    wgt_decay=0,
    lambda_3=1,
    lambda_4=0,
    drop_out_p=0.2,
    exp_num=0,
    M_train=3,
    lambda_mean=0.3,
    mean_samples=20,
):
    """Compute one p-value for the high-dimensional mean-signal MMDCI test."""
    if simulation == "type1error":
        sim_x, sim_y, sim_z = generate_samples_random(
            Ax, Ay, size=n, sType="CI", dx=x_dims, dy=y_dims, dz=z_dim,
            nstd=nstd, alpha_x=0.0, dist_z=z_dist, preprocess=preprocess,
        )
    elif simulation == "power":
        sim_x, sim_y, sim_z = generate_samples_random(
            Ax, Ay, size=n, sType="dependent", dx=x_dims, dy=y_dims, dz=z_dim,
            nstd=nstd, alpha_x=a_x, dist_z=z_dist, preprocess=preprocess,
        )
    else:
        raise ValueError("simulation must be 'type1error' or 'power'.")

    x, y, z = sim_x.to(device), sim_y.to(device), sim_z.to(device)
    Ax, Ay = Ax.to(device), Ay.to(device)

    test_size = int(n / k)
    stat_all = torch.zeros(k, 1)
    boot_temp_all = torch.zeros(k, boot_num)

    # Strict cross-fitting: do not initialize generators outside the fold loop.
    # Each fold gets newly initialized generators trained only on its training folds.

    for k_fold in range(k):
        k_fold_start = int(n / k * k_fold)
        k_fold_end = int(n / k * (k_fold + 1))

        X_test = x[k_fold_start:k_fold_end]
        Y_test = y[k_fold_start:k_fold_end]
        Z_test = z[k_fold_start:k_fold_end]
        X_train = torch.cat((x[:k_fold_start], x[k_fold_end:]), dim=0)
        Y_train = torch.cat((y[:k_fold_start], y[k_fold_end:]), dim=0)
        Z_train = torch.cat((z[:k_fold_start], z[k_fold_end:]), dim=0)

        if k == 1:
            X_train, Y_train, Z_train = X_test, Y_test, Z_test

        train_xyz = DatasetSelect_GAN(X_train, Y_train, Z_train, batch_size)
        dataloader_xyz = DataLoader(train_xyz, batch_size=batch_size, shuffle=True, drop_last=False)

        if not using_orcale:
            # Strict cross-fitting: reinitialize both generators inside every fold.
            # No weight reuse / warm start across folds.
            G_zy = None
            G_zx = None

            # Training bandwidths are computed from the training folds only.
            sigma_w_train = safe_median_bandwidth(pairwise_l1(Z_train, Z_train))
            sigma_u_train = safe_median_bandwidth(pairwise_l1(Y_train, Y_train))
            sigma_v_train = safe_median_bandwidth(pairwise_l1(X_train, X_train))

            G_zy, G_zx = train_ver3(
                G_zx=G_zx,
                G_zy=G_zy,
                X=X_train,
                Y=Y_train,
                Z=Z_train,
                X_test=X_test,
                Y_test=Y_test,
                Z_test=Z_test,
                noise_dimension=noise_dimension,
                noise_type="normal",
                G_lr=G_lr,
                hidden_layer_size=hidden_layer_size,
                DataLoader=dataloader_xyz,
                BN_type=False,
                ReLU_coef=0.1,
                epochs_num=epochs_num,
                sigma_z=sigma_w_train,
                sigma_x=sigma_v_train,
                sigma_y=sigma_u_train,
                normal_ini=normal_ini,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                using_Gen=using_Gen,
                wgt_decay=wgt_decay,
                lambda_3=lambda_3,
                lambda_4=lambda_4,
                drop_out_p=drop_out_p,
                M_train=M_train,
                lambda_mean=lambda_mean,
                mean_samples=mean_samples,
            )

        dataset_test = DatasetSelect(X_test, Y_test, Z_test)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

        gen_x_all = torch.zeros(test_size, M, x_dims, device=device)
        gen_y_all = torch.zeros(test_size, M, y_dims, device=device)
        z_all = torch.zeros(test_size, z_dim, device=device)
        x_all = torch.zeros(test_size, x_dims, device=device)
        y_all = torch.zeros(test_size, y_dims, device=device)

        if not using_orcale:
            G_zx.eval()
            G_zy.eval()

        cur_itr = 0
        for x_test, y_test, z_test in dataloader_test:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            z_test = z_test.to(device)
            z_test_temp = z_test.repeat(M, 1)

            if not using_orcale:
                with torch.no_grad():
                    Noise_fake_x = sample_noise(M, noise_dimension, "normal", input_var=1.0 / 3.0)
                    fake_x = G_zx(torch.cat((z_test_temp, Noise_fake_x), dim=1)).reshape(M, x_dims)
                    Noise_fake_y = sample_noise(M, noise_dimension, "normal", input_var=1.0 / 3.0)
                    fake_y = G_zy(torch.cat((z_test_temp, Noise_fake_y), dim=1)).reshape(M, y_dims)
            else:
                oracle_type = "CI" if simulation == "type1error" else "dependent"
                fake_x, fake_y = generate_samples_from_fixed_Z_random(
                    Ax, Ay, z_test_temp, size=M, sType=oracle_type,
                    dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd, alpha_x=a_x,
                )

            gen_x_all[cur_itr] = fake_x.detach()
            gen_y_all[cur_itr] = fake_y.detach()
            x_all[cur_itr] = x_test.reshape(-1)
            y_all[cur_itr] = y_test.reshape(-1)
            z_all[cur_itr] = z_test.reshape(-1)
            cur_itr += 1

        # Fixed: do not separately standardize true and generated samples.
        standardise = False
        if standardise:
            raise RuntimeError("standardise must remain False for Section 4.2 reproduction.")

        sigma_w = safe_median_bandwidth(pairwise_l1(z_all, z_all))
        sigma_u = safe_median_bandwidth(pairwise_l1(y_all, y_all))
        sigma_v = safe_median_bandwidth(pairwise_l1(x_all, x_all))

        cur_stat, cur_boot_temp = get_p_value_stat_1(
            boot_num,
            M,
            test_size,
            gen_x_all,
            gen_y_all,
            x_all,
            y_all,
            z_all,
            sigma_w,
            sigma_u,
            sigma_v,
            boor_rv_type,
        )
        stat_all[k_fold, 0] = cur_stat
        boot_temp_all[k_fold, :] = torch.from_numpy(cur_boot_temp)

    p_value = np.mean(torch.mean(boot_temp_all, dim=0).numpy() > torch.mean(stat_all).item())
    return p_value


# =========================
# Experiment runner with multi-GPU allocation
# =========================


def run_experiment(params):
    test = params["test"]
    sample_size = params["sample_size"]
    batch_size = params["batch_size"]
    z_dim = params["z_dim"]
    dx = params["dx"]
    dy = params["dy"]
    n_test = params["n_test"]
    epochs_num = params["epochs_num"]
    eps_std = params["eps_std"]
    dist_z = params["dist_z"]
    alpha_x = params["alpha_x"]
    m_value = params["m_value"]
    k_value = params["k_value"]
    j_value = params["j_value"]
    noise_dimension = params["noise_dimension"]
    hidden_layer_size = params["hidden_layer_size"]
    normal_ini = params["normal_ini"]
    preprocess = params["preprocess"]
    G_lr = params["G_lr"]
    alpha = params["alpha"]
    alpha1 = params["alpha1"]
    set_seeds = params["set_seeds"]
    using_orcale = params["using_orcale"]
    lambda_1 = params["lambda_1"]
    lambda_2 = params["lambda_2"]
    using_Gen = params["using_Gen"]
    boor_rv_type = params["boor_rv_type"]
    wgt_decay = params["wgt_decay"]
    lambda_3 = params["lambda_3"]
    lambda_4 = params["lambda_4"]
    drop_out_p = params["drop_out_p"]
    M_train = params["M_train"]
    lambda_mean = params.get("lambda_mean", 0.3)
    mean_samples = params.get("mean_samples", 20)
    workers_per_gpu = int(params.get("workers_per_gpu", 1))
    enable_cuda = bool(params.get("enable_cuda", True))

    gpu_ids = get_available_gpus(enable_cuda=enable_cuda)
    if gpu_ids:
        n_jobs = min(n_test, 12)
        gpu_assignment_msg = ", ".join([f"worker {i}->cuda:{gpu_ids[i % len(gpu_ids)]}" for i in range(n_jobs)])
    else:
        cpu_cores = max(1, multiprocessing.cpu_count() - 2)
        n_jobs = min(n_test, cpu_cores)
        gpu_assignment_msg = "CPU only"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Start parallel experiment")
    print(f"Mode: {test} | n: {sample_size} | z_dim: {z_dim} | folds: {k_value} | reps: {n_test}")
    print(f"Available GPUs: {len(gpu_ids)} | n_jobs: {n_jobs}")
    print(f"lambda_mean: {lambda_mean} | mean_samples: {mean_samples}")
    if test == "power":
        print(f"Alternative b = alpha_x: {alpha_x}")

    def single_trial_mGAN(exp_index):
        # Round-robin GPU assignment. This avoids all jobs crowding onto cuda:0.
        if gpu_ids:
            gpu_id = gpu_ids[exp_index % len(gpu_ids)]
        else:
            gpu_id = None
        set_process_device(gpu_id=gpu_id, enable_cuda=enable_cuda)

        seed = int(set_seeds + 10007 * exp_index)
        set_all_seeds(seed)
        rng = np.random.default_rng(seed)

        # High-dimensional mean-signal DGP.
        # Use the same four projection directions P in mX(Z) and mY(Z).
        P = make_mean_signal_projection(z_dim, k=4, rng=rng)
        Ax = P
        Ay = P.clone()

        p_val = mGAN(
            Ax=Ax,
            Ay=Ay,
            n=sample_size,
            z_dim=z_dim,
            simulation=test,
            batch_size=batch_size,
            epochs_num=epochs_num,
            nstd=eps_std,
            z_dist=dist_z,
            x_dims=dx,
            y_dims=dy,
            a_x=alpha_x,
            M=m_value,
            k=k_value,
            boot_num=j_value,
            noise_dimension=noise_dimension,
            hidden_layer_size=hidden_layer_size,
            normal_ini=normal_ini,
            preprocess=preprocess,
            G_lr=G_lr,
            using_orcale=using_orcale,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            using_Gen=using_Gen,
            boor_rv_type=boor_rv_type,
            wgt_decay=wgt_decay,
            lambda_3=lambda_3,
            lambda_4=lambda_4,
            drop_out_p=drop_out_p,
            exp_num=exp_index + 1,
            M_train=M_train,
            lambda_mean=lambda_mean,
            mean_samples=mean_samples,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return p_val

    p_values = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(single_trial_mGAN)(i) for i in range(n_test)
    )

    p_values = np.asarray(p_values, dtype=float)
    final_result = np.mean(p_values < alpha)
    final_result1 = np.mean(p_values < alpha1)

    print("\n" + "=" * 60)
    print(f"Experiment finished - Type: {test.upper()} | Z Dimension: {z_dim}")
    print(f"Emp Rej Rate: {final_result:.4f} at alpha = {alpha}")
    print(f"Emp Rej Rate: {final_result1:.4f} at alpha1 = {alpha1}")
    print("=" * 60 + "\n")
    return p_values

# =========================
# Run directly
# =========================

if __name__ == "__main__":
    p_val_list = run_experiment(param)
