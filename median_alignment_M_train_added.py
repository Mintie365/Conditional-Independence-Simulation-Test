# ============================================================
# 参数配置
# ============================================================

param = {
    "test": "power",
    "sample_size": 200,
    "batch_size": 128,
    "z_dim": 1,
    "dx": 1,
    "dy": 1,
    "n_test": 100,
    "epochs_num": 100,
    "eps_std": 1.0,
    "dist_z": 'gaussian',
    "alpha_x": 0.20,
    "m_value": 100,          # test statistic 阶段每个 Z 生成的样本数 M
    "M_train": 20,           # training MMD 阶段每个 Z 生成的样本数，可调
    "k_value": 8,
    "j_value": 1000,
    "noise_dimension": 5,
    "noise_dimension_type": "normal",
    "noise_dimension_var": 1,
    "hidden_layer_size": 1024,
    "normal_ini": False,
    "preprocess": 'scale',
    "G_lr": 7e-5,
    "alpha": 0.1,
    "alpha1": 0.05,
    "set_seeds": 0,
    "using_orcale": False,
    "lambda_1": 1,
    "lambda_2": 1,
    "using_Gen": '1',
    "boor_rv_type": 'gaussian',
    "wgt_decay": 1e-5,
    "lambda_3": 1e-4,
    "lambda_4": 2e-5,
    "drop_out_p": 0.2,
    "is_sparse": True,
    "sparse_ratio": 0.05,
    "lambda_median": 0.3,
    "median_samples": 30
}

import torch
import torch.distributions as TD
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from datetime import datetime
import functools
from tqdm import tqdm

# ============================================================
# 偏态数据生成辅助函数
# ============================================================

def _sigmoid_torch(x):
    return 1.0 / (1.0 + torch.exp(-x))


def _sample_shifted_lognormal_noise(num, dim, sigma=1.0, device=None):
    if device is None:
        device = torch.device("cpu")
    N = torch.randn(num, dim, device=device)
    denom = np.sqrt((np.exp(sigma ** 2) - 1.0) * np.exp(sigma ** 2))
    eps = (torch.exp(sigma * N) - 1.0) / denom
    return eps

# ============================================================
# 数据生成：替换为偏态 DGP
# ============================================================

def generate_samples_random(size=1000, sType='H0', dx=1, dy=1, dz=1, nstd=1.0, alpha_x=0.05,
                            preprocess="None", dist_z='gaussian'):
    num = size
    lognormal_sigma = 1.0

    if dist_z == 'gaussian':
        z_generator = TD.MultivariateNormal(torch.zeros(dz), 1 * torch.eye(dz))
        Z = z_generator.sample((num,))
    elif dist_z == 'laplace':
        z_generator = TD.Laplace(torch.zeros(dz), torch.ones(dz))
        Z = z_generator.sample((num,))
    else:
        raise ValueError("dist_z must be 'gaussian' or 'laplace'.")

    m = TD.Bernoulli(torch.tensor([alpha_x]))
    Axy = torch.ones((dx, dy)) * alpha_x

    eps_x = _sample_shifted_lognormal_noise(num, dx, sigma=lognormal_sigma, device=Z.device)
    eps_y = _sample_shifted_lognormal_noise(num, dy, sigma=lognormal_sigma, device=Z.device)

    z1 = Z[:, [0]]

    m_x = 0.8 * z1 + 0.5 * torch.sin(1.5 * z1)
    m_y = -0.6 * z1 + 0.4 * (z1 ** 2)

    s_x = 0.5 + 0.25 * torch.abs(z1) + 0.15 * _sigmoid_torch(z1)
    s_y = 0.5 + 0.20 * torch.abs(z1) + 0.15 * _sigmoid_torch(-z1)

    if dx > 1:
        m_x = m_x.repeat(1, dx)
        s_x = s_x.repeat(1, dx)

    if dy > 1:
        m_y = m_y.repeat(1, dy)
        s_y = s_y.repeat(1, dy)

    if sType == 'H0':
        X = m_x + nstd * s_x * eps_x
        Y = m_y + nstd * s_y * eps_y
    else:
        delta = m.sample((num,))

        if dx > 1:
            delta_x = delta.repeat(1, dx)
        else:
            delta_x = delta

        if dx == dy:
            shared_eps = eps_y
        elif dy == 1:
            shared_eps = eps_y.repeat(1, dx)
        else:
            shared_eps = eps_y[:, :dx]

        X = m_x + nstd * s_x * ((1.0 - delta_x) * eps_x + delta_x * shared_eps)
        Y = m_y + nstd * s_y * eps_y

    return X, Y, Z


def generate_samples_from_fixed_Z_random(Z, size=1000, sType='H0', dx=1, dy=1, dz=1, nstd=1.0,
                                         alpha_x=0.05, normalize=True, seed=None, dist_z='gaussian'):
    num = size
    lognormal_sigma = 1.0

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device_local = Z.device

    eps_x = _sample_shifted_lognormal_noise(num, dx, sigma=lognormal_sigma, device=device_local)
    eps_y = _sample_shifted_lognormal_noise(num, dy, sigma=lognormal_sigma, device=device_local)

    Axy = torch.ones((dx, dy), device=device_local) * alpha_x
    m = TD.Bernoulli(torch.tensor([alpha_x], device=device_local))

    z1 = Z[:, [0]]

    m_x = 0.8 * z1 + 0.5 * torch.sin(1.5 * z1)
    m_y = -0.6 * z1 + 0.4 * (z1 ** 2)

    s_x = 0.5 + 0.25 * torch.abs(z1) + 0.15 * _sigmoid_torch(z1)
    s_y = 0.5 + 0.20 * torch.abs(z1) + 0.15 * _sigmoid_torch(-z1)

    if dx > 1:
        m_x = m_x.repeat(1, dx)
        s_x = s_x.repeat(1, dx)

    if dy > 1:
        m_y = m_y.repeat(1, dy)
        s_y = s_y.repeat(1, dy)

    if sType == 'H0':
        X = m_x + nstd * s_x * eps_x
        Y = m_y + nstd * s_y * eps_y
    else:
        delta = m.sample((num,))

        if dx > 1:
            delta_x = delta.repeat(1, dx)
        else:
            delta_x = delta

        if dx == dy:
            shared_eps = eps_y
        elif dy == 1:
            shared_eps = eps_y.repeat(1, dx)
        else:
            shared_eps = eps_y[:, :dx]

        X = m_x + nstd * s_x * ((1.0 - delta_x) * eps_x + delta_x * shared_eps)
        Y = m_y + nstd * s_y * eps_y

    return X, Y


def get_p_value_stat_1(boot_num, M, n, gen_x_all_torch, gen_y_all_torch, x_torch, y_torch, z_torch,
                       sigma_w, sigma_u=1, sigma_v=1, boor_rv_type="gaussian",
                       device=None):
    w_mx = torch.linalg.vector_norm(
        z_torch.repeat(n, 1, 1) - torch.swapaxes(z_torch.repeat(n, 1, 1), 0, 1), ord=1, dim=2)
    w_mx = torch.exp(-w_mx / sigma_w)

    u_mx_1 = torch.exp(-torch.abs(y_torch.repeat(1, n) - y_torch.repeat(1, n).T) / sigma_u)
    u_mx_2 = torch.mean(
        torch.exp(-torch.abs(gen_y_all_torch.repeat(n, 1, 1) - y_torch.repeat(1, n).reshape(n, n, 1)) / sigma_u),
        dim=2)
    u_mx_3 = u_mx_2.T

    gen_y_all_torch_rep = gen_y_all_torch.repeat(n, 1, 1)
    temp_mx = gen_y_all_torch_rep[:, :, 0].T
    sum_mx = torch.mean(
        torch.exp(-torch.abs(gen_y_all_torch_rep - temp_mx.reshape(n, n, 1)) / sigma_u), dim=2)

    v_mx_1 = torch.exp(-torch.abs(x_torch.repeat(1, n) - x_torch.repeat(1, n).T) / sigma_v)
    v_mx_2 = torch.mean(
        torch.exp(-torch.abs(gen_x_all_torch.repeat(n, 1, 1) - x_torch.repeat(1, n).reshape(n, n, 1)) / sigma_v),
        dim=2)
    v_mx_3 = v_mx_2.T

    gen_x_all_torch_rep = gen_x_all_torch.repeat(n, 1, 1)
    temp2_mx = gen_x_all_torch_rep[:, :, 0].T
    sum2_mx = torch.mean(
        torch.exp(-torch.abs(gen_x_all_torch_rep - temp2_mx.reshape(n, n, 1)) / sigma_v), dim=2)

    for i in range(1, M):
        temp_mx = gen_y_all_torch_rep[:, :, i].T
        temp_add_mx = torch.mean(
            torch.exp(-torch.abs(gen_y_all_torch_rep - temp_mx.reshape(n, n, 1)) / sigma_u), dim=2)
        sum_mx = sum_mx + temp_add_mx

        temp2_mx = gen_x_all_torch_rep[:, :, i].T
        temp2_add_mx = torch.mean(
            torch.exp(-torch.abs(gen_x_all_torch_rep - temp2_mx.reshape(n, n, 1)) / sigma_v), dim=2)
        sum2_mx = sum2_mx + temp2_add_mx

    u_mx_4 = 1 / M * sum_mx
    u_mx = u_mx_1 - u_mx_2 - u_mx_3 + u_mx_4

    v_mx_4 = 1 / M * sum2_mx
    v_mx = v_mx_1 - v_mx_2 - v_mx_3 + v_mx_4

    FF_mx = u_mx * v_mx * w_mx * (1 - torch.eye(n).to(device))
    stat = 1 / (n - 1) * torch.sum(FF_mx).item()

    boottemp = np.array([])
    if boor_rv_type == "rademacher":
        eboot = torch.sign(torch.randn(n, boot_num)).to(device)
    elif boor_rv_type == "gaussian":
        eboot = torch.randn(n, boot_num).to(device)
    else:
        raise ValueError("boor_rv_type must be 'rademacher' or 'gaussian'.")

    for bb in range(boot_num):
        random_mx = torch.matmul(eboot[:, bb].reshape(-1, 1), eboot[:, bb].reshape(-1, 1).T)
        bootmatrix = FF_mx * random_mx
        stat_boot = 1 / (n - 1) * torch.sum(bootmatrix).item()
        boottemp = np.append(boottemp, stat_boot)

    return stat, boottemp


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


class DatasetSelect_GAN(torch.utils.data.Dataset):
    def __init__(self, X, Y, Z, batch_size):
        self.X_real = X
        self.Y_real = Y
        self.Z_real = Z
        self.batch_size = batch_size
        self.sample_size = X.shape[0]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        return self.X_real[index], self.Y_real[index], self.Z_real[index], \
               self.Z_real[(self.batch_size + index) % self.sample_size]


class DatasetSelect_GAN_ver2(torch.utils.data.Dataset):
    def __init__(self, Y, Z, batch_size):
        self.Y_real = Y
        self.Z_real = Z
        self.batch_size = batch_size
        self.sample_size = Z.shape[0]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        return self.Y_real[index], self.Z_real[index]


def sample_noise(sample_size, noise_dimension, noise_type, input_var=1,
                 device=None):
    if noise_type == "normal":
        noise_generator = TD.MultivariateNormal(
            torch.zeros(noise_dimension).to(device),
            input_var * torch.eye(noise_dimension).to(device))
        Z = noise_generator.sample((sample_size,))
    elif noise_type == "unif":
        Z = torch.rand(sample_size, noise_dimension).to(device)
    elif noise_type == "Cauchy":
        Z = TD.Cauchy(torch.tensor([0.0]), torch.tensor([1.0])).sample(
            (sample_size, noise_dimension)).squeeze(2).to(device)
    else:
        raise ValueError("noise_type must be 'normal', 'unif', or 'Cauchy'.")
    return Z


class Generator(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, noise_dimension, hidden_layer_size,
                 BN_type, ReLU_coef, drop_out_p, drop_input=False):
        super(Generator, self).__init__()
        self.BN_type = BN_type
        self.ReLU_coef = ReLU_coef
        self.fc1 = torch.nn.Linear(input_dimension + noise_dimension, hidden_layer_size, bias=True)
        if BN_type:
            self.BN1 = torch.nn.BatchNorm1d(hidden_layer_size, 0.8, affine=False)
            self.BN2 = torch.nn.BatchNorm1d(hidden_layer_size, 0.8, affine=False)
            self.BN3 = torch.nn.BatchNorm1d(hidden_layer_size, 0.8, affine=False)
        self.leakyReLU1 = torch.nn.LeakyReLU(ReLU_coef)
        self.fc2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size, bias=True)
        self.fc3 = torch.nn.Linear(hidden_layer_size, hidden_layer_size, bias=True)
        self.fc_last = torch.nn.Linear(hidden_layer_size, output_dimension, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.drop_out0 = torch.nn.Dropout(p=drop_out_p)
        self.drop_out1 = torch.nn.Dropout(p=drop_out_p)
        self.drop_out2 = torch.nn.Dropout(p=drop_out_p)
        self.drop_out3 = torch.nn.Dropout(p=drop_out_p)
        self.drop_input = drop_input

    def forward(self, x):
        if self.BN_type:
            if self.drop_input:
                x = self.drop_out0(x)
            x = self.drop_out1(self.leakyReLU1(self.BN1(self.fc1(x))))
            x = self.drop_out2(self.leakyReLU1(self.BN2(self.fc2(x))))
            x = self.fc_last(x)
        else:
            if self.drop_input:
                x = self.drop_out0(x)
            x = self.drop_out1(self.leakyReLU1(self.fc1(x)))
            x = self.drop_out2(self.leakyReLU1(self.fc2(x)))
            x = self.fc_last(x)
        return x


class NonFullyConnected_1(torch.nn.Module):
    def __init__(self, size_in, size_out, m, bias=True):
        super(NonFullyConnected_1, self).__init__()
        self.linear = torch.nn.Linear(m * size_in, m * size_out, bias=bias)
        self.mask = functools.reduce(
            torch.block_diag, [torch.ones(size_out, size_in) for i in range(m)]
        )

    def forward(self, x):
        self.linear.weight.data *= self.mask.to(self.linear.weight.device)
        return self.linear(x)


class Generator_2(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, noise_dimension, hidden_layer_size,
                 BN_type, ReLU_coef, hidden_layer_depth=1, ntargets_k=5):
        super(Generator_2, self).__init__()
        self.input_dimension = input_dimension + noise_dimension
        self.output_dimension = output_dimension
        self.ntargets_k = ntargets_k
        self.hidden_layer_sizes = [hidden_layer_size] * hidden_layer_depth
        self.BN_type = BN_type
        self.leakyrelu = torch.nn.LeakyReLU(ReLU_coef)
        self.linear_layers_from_input = torch.nn.Linear(
            self.input_dimension, ntargets_k * self.hidden_layer_sizes[0])
        self.linear_layers_between = torch.nn.ModuleList([
            NonFullyConnected_1(self.hidden_layer_sizes[0], self.hidden_layer_sizes[0], ntargets_k)
            for i in range(len(self.hidden_layer_sizes))
        ])
        self.linear8 = torch.nn.Linear(self.hidden_layer_sizes[0] * ntargets_k, self.output_dimension)
        if BN_type:
            self.BN1 = torch.nn.BatchNorm1d(hidden_layer_size, 0.8, affine=False)

    def forward(self, input):
        if self.BN_type:
            output = self.linear_layers_from_input(input)
            output = self.leakyrelu(self.BN1(output))
            for linear_layers_between in self.linear_layers_between:
                output = linear_layers_between(output)
                output = self.leakyrelu(self.BN1(output))
        else:
            output = self.linear_layers_from_input(input)
            output = self.leakyrelu(output)
            for linear_layers_between in self.linear_layers_between:
                output = linear_layers_between(output)
                output = self.leakyrelu(output)
        return self.linear8(output)


def _ensure_real_2d(Y):
    if Y.dim() == 1:
        Y = Y.reshape(-1, 1)
    return Y


def _ensure_generated_3d(hat_Y):
    # 训练阶段现在推荐输入 (B, M_train, d)。
    # 为了兼容旧代码，如果输入是 (B, d)，则视作 M_train = 1。
    if hat_Y.dim() == 1:
        hat_Y = hat_Y.reshape(-1, 1).unsqueeze(1)
    elif hat_Y.dim() == 2:
        hat_Y = hat_Y.unsqueeze(1)
    return hat_Y


def find_loss(Y, hat_Y, Z, sigma_z, sigma_y):
    """
    Laplace-kernel conditional MMD training loss.

    支持：
      Y:     (B, d) 或 (B,)
      hat_Y: (B, M_train, d)；兼容旧格式 (B, d)，此时 M_train = 1
      Z:     (B, dz)
    """
    Y = _ensure_real_2d(Y)
    hat_Y = _ensure_generated_3d(hat_Y)

    n = Z.shape[0]
    M_train = hat_Y.shape[1]
    sigma_z = max(float(sigma_z), 1e-8)
    sigma_y = max(float(sigma_y), 1e-8)

    dist_z = torch.linalg.vector_norm(
        Z.repeat(n, 1, 1) - torch.swapaxes(Z.repeat(n, 1, 1), 0, 1), ord=1, dim=2)
    Kz = torch.exp(-dist_z / sigma_z)

    dist_yy = torch.linalg.vector_norm(
        Y.repeat(n, 1, 1) - torch.swapaxes(Y.repeat(n, 1, 1), 0, 1), ord=1, dim=2)
    Kyy = torch.exp(-dist_yy / sigma_y)

    # K_y_fake[i, j] = mean_m K(Y_i, hat_Y_{j,m})
    dist_y_fake = torch.sum(torch.abs(Y[:, None, None, :] - hat_Y[None, :, :, :]), dim=-1)
    K_y_fake = torch.mean(torch.exp(-dist_y_fake / sigma_y), dim=2)
    K_fake_y = K_y_fake.T

    # K_fake_fake[i, j] = mean_{m1,m2} K(hat_Y_{i,m1}, hat_Y_{j,m2})
    K_fake_fake = torch.zeros(n, n, device=Y.device)
    for m1 in range(M_train):
        dist_ff = torch.sum(
            torch.abs(hat_Y[:, m1, :][:, None, None, :] - hat_Y[None, :, :, :]),
            dim=-1
        )
        K_fake_fake += torch.mean(torch.exp(-dist_ff / sigma_y), dim=2)
    K_fake_fake = K_fake_fake / M_train

    FF_mx = (Kyy - K_y_fake - K_fake_y + K_fake_fake) * Kz
    loss = torch.sum(FF_mx) / (n ** 2)
    return loss


def find_loss_2(Y, hat_Y, Z, sigma_z, sigma_y):
    """
    Gaussian-kernel conditional MMD training loss.

    支持：
      Y:     (B, d) 或 (B,)
      hat_Y: (B, M_train, d)；兼容旧格式 (B, d)，此时 M_train = 1
      Z:     (B, dz)
    """
    Y = _ensure_real_2d(Y)
    hat_Y = _ensure_generated_3d(hat_Y)

    n = Z.shape[0]
    M_train = hat_Y.shape[1]
    sigma_z = max(float(sigma_z), 1e-8)
    sigma_y = max(float(sigma_y), 1e-8)

    dist_z = torch.linalg.vector_norm(
        Z.repeat(n, 1, 1) - torch.swapaxes(Z.repeat(n, 1, 1), 0, 1), ord=2, dim=2)
    Kz = torch.exp(-(dist_z ** 2) / sigma_z)

    dist_yy = torch.sum((Y[:, None, :] - Y[None, :, :]) ** 2, dim=-1)
    Kyy = torch.exp(-dist_yy / sigma_y)

    # K_y_fake[i, j] = mean_m K(Y_i, hat_Y_{j,m})
    dist_y_fake = torch.sum((Y[:, None, None, :] - hat_Y[None, :, :, :]) ** 2, dim=-1)
    K_y_fake = torch.mean(torch.exp(-dist_y_fake / sigma_y), dim=2)
    K_fake_y = K_y_fake.T

    # K_fake_fake[i, j] = mean_{m1,m2} K(hat_Y_{i,m1}, hat_Y_{j,m2})
    K_fake_fake = torch.zeros(n, n, device=Y.device)
    for m1 in range(M_train):
        dist_ff = torch.sum(
            (hat_Y[:, m1, :][:, None, None, :] - hat_Y[None, :, :, :]) ** 2,
            dim=-1
        )
        K_fake_fake += torch.mean(torch.exp(-dist_ff / sigma_y), dim=2)
    K_fake_fake = K_fake_fake / M_train

    FF_mx = (Kyy - K_y_fake - K_fake_y + K_fake_fake) * Kz
    loss = torch.sum(FF_mx) / (n ** 2)
    return loss


def pinball_loss(y_true, y_pred, tau=0.5):
    diff = y_true - y_pred
    loss = torch.where(diff >= 0, tau * diff, (1.0 - tau) * (-diff))
    return loss.mean()


def train_ver3(
    G_zx, G_zy,
    X, Y, Z, X_test, Y_test, Z_test, M,
    noise_dimension, noise_type, G_lr, hidden_layer_size,
    DataLoader, BN_type, ReLU_coef,
    lambda_median=0.5, median_samples=20,
    M_train=1,
    epochs_num=50,
    patience=5, min_delta=1e-5,
    sigma_z=1, sigma_x=1, sigma_y=1,
    normal_ini=False,
    lambda_1=1, lambda_2=1, using_Gen='1', wgt_decay=0,
    lambda_3=0, drop_out_p=0.2, noise_dimension_var=1,
    lambda_4=0,
    device=None):
    input_dimension = Z.shape[1]
    output_dimension_y = Y.shape[1]
    output_dimension_x = X.shape[1]
    M_train = int(M_train)
    if M_train < 1:
        raise ValueError("M_train must be a positive integer.")

    if G_zy is None or G_zx is None:
        if using_Gen == '1':
            G_zy = Generator(input_dimension, output_dimension_y, noise_dimension,
                             hidden_layer_size, BN_type, ReLU_coef, drop_out_p).to(device)
            G_zx = Generator(input_dimension, output_dimension_x, noise_dimension,
                             hidden_layer_size, BN_type, ReLU_coef, drop_out_p).to(device)
        elif using_Gen == '2':
            G_zy = Generator_2(input_dimension, output_dimension_y, noise_dimension,
                               hidden_layer_size, BN_type, ReLU_coef).to(device)
            G_zx = Generator_2(input_dimension, output_dimension_x, noise_dimension,
                               hidden_layer_size, BN_type, ReLU_coef).to(device)
        else:
            raise ValueError("using_Gen must be '1' or '2'.")

    if normal_ini:
        for p in G_zy.parameters():
            p.data = torch.randn(p.shape, device=device) / np.sqrt(float(hidden_layer_size * 2))
        for p in G_zx.parameters():
            p.data = torch.randn(p.shape, device=device) / np.sqrt(float(hidden_layer_size * 2))

    G_zy_solver = optim.Adam(G_zy.parameters(), lr=G_lr, betas=(0.5, 0.999), weight_decay=wgt_decay)
    G_zx_solver = optim.Adam(G_zx.parameters(), lr=G_lr, betas=(0.5, 0.999), weight_decay=wgt_decay)

    iter_count = 0
    G_zy = G_zy.train()
    G_zx = G_zx.train()

    best_loss = float('inf')
    counter = 0

    for epoch in range(epochs_num):
        batch_count = 0
        G_zy = G_zy.train()
        G_zx = G_zx.train()

        for X_real, Y_real, Z_real, Z_fake in DataLoader:
            X_real = X_real.to(device)
            Y_real = Y_real.to(device)
            Z_real = Z_real.to(device)
            Z_fake = Z_fake.to(device)

            batch_size = Z_real.shape[0]

            # ============================================================
            # 中位数对齐：median_samples 控制每个 Z 的 Monte Carlo 样本数
            # ============================================================
            Z_repeated = Z_real.repeat_interleave(median_samples, dim=0)

            Noise_for_median_y = sample_noise(
                Z_repeated.shape[0], noise_dimension, noise_type,
                input_var=noise_dimension_var, device=device).to(device)
            Y_generated_group = G_zy(torch.cat((Z_repeated, Noise_for_median_y), dim=1))
            Y_generated_reshaped = Y_generated_group.reshape(batch_size, median_samples, -1)
            Y_median_pred = torch.median(Y_generated_reshaped, dim=1).values
            loss_median_y = pinball_loss(Y_real, Y_median_pred, tau=0.5)

            Noise_for_median_x = sample_noise(
                Z_repeated.shape[0], noise_dimension, noise_type,
                input_var=noise_dimension_var, device=device).to(device)
            X_generated_group = G_zx(torch.cat((Z_repeated, Noise_for_median_x), dim=1))
            X_generated_reshaped = X_generated_group.reshape(batch_size, median_samples, -1)
            X_median_pred = torch.median(X_generated_reshaped, dim=1).values
            loss_median_x = pinball_loss(X_real, X_median_pred, tau=0.5)

            # ============================================================
            # 训练 MMD：M_train 控制每个 Z 的 generated samples 数
            # ============================================================
            Z_real_mmd = Z_real.repeat_interleave(M_train, dim=0)

            Noise_fake_y = sample_noise(
                Z_real_mmd.shape[0], noise_dimension, noise_type,
                input_var=noise_dimension_var, device=device).to(device)
            Y_fake = G_zy(torch.cat((Z_real_mmd, Noise_fake_y), dim=1))
            Y_fake = Y_fake.reshape(batch_size, M_train, -1)

            Noise_fake_x = sample_noise(
                Z_real_mmd.shape[0], noise_dimension, noise_type,
                input_var=noise_dimension_var, device=device).to(device)
            X_fake = G_zx(torch.cat((Z_real_mmd, Noise_fake_x), dim=1))
            X_fake = X_fake.reshape(batch_size, M_train, -1)

            g_zy_error = None
            G_zy_solver.zero_grad()
            g_zx_error = None
            G_zx_solver.zero_grad()

            l1_regularization_first_layer = 0
            l1_regularization_rest_layers = 0
            for name, param_g in G_zy.named_parameters():
                if "fc1" in name:
                    l1_regularization_first_layer += torch.linalg.vector_norm(param_g, ord=1)
                else:
                    l1_regularization_rest_layers += torch.linalg.vector_norm(param_g, ord=1)

            mmd_loss_y = (lambda_1 * find_loss(Y_real, Y_fake, Z_real, sigma_z=sigma_z, sigma_y=sigma_y) +
                          lambda_2 * find_loss_2(Y_real, Y_fake, Z_real, sigma_z=sigma_z, sigma_y=sigma_y) +
                          lambda_3 * l1_regularization_rest_layers +
                          lambda_4 * l1_regularization_first_layer)

            g_zy_error = mmd_loss_y + lambda_median * loss_median_y
            g_zy_error.backward()
            torch.nn.utils.clip_grad_norm_(G_zy.parameters(), max_norm=0.5)
            G_zy_solver.step()

            l1_regularization_first_layer = 0
            l1_regularization_rest_layers = 0
            for name, param_g in G_zx.named_parameters():
                if "fc1" in name:
                    l1_regularization_first_layer += torch.linalg.vector_norm(param_g, ord=1)
                else:
                    l1_regularization_rest_layers += torch.linalg.vector_norm(param_g, ord=1)

            mmd_loss_x = (lambda_1 * find_loss(X_real, X_fake, Z_real, sigma_z=sigma_z, sigma_y=sigma_x) +
                          lambda_2 * find_loss_2(X_real, X_fake, Z_real, sigma_z=sigma_z, sigma_y=sigma_x) +
                          lambda_3 * l1_regularization_rest_layers +
                          lambda_4 * l1_regularization_first_layer)

            g_zx_error = mmd_loss_x + lambda_median * loss_median_x
            g_zx_error.backward()
            torch.nn.utils.clip_grad_norm_(G_zx.parameters(), max_norm=0.5)
            G_zx_solver.step()

            iter_count += 1
            batch_count += 1

            current_loss = g_zx_error + g_zy_error
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                counter = 0
            else:
                counter += 1

        if counter >= patience:
            break

    return G_zy, G_zx


def mGAN(n=500, z_dim=1, simulation='type1error', batch_size=64, epochs_num=1000,
         nstd=1.0, z_dist='gaussian', x_dims=1, y_dims=1, a_x=0.05, M=500, k=2, boot_num=1000,
         noise_dimension=10, hidden_layer_size=512, normal_ini=False, preprocess='normalize',
         G_lr=1e-5, using_orcale=False, lambda_1=1, lambda_2=1,
         lambda_median=0.3, median_samples=20, M_train=1,
         using_Gen='1',
         boor_rv_type="gaussian", wgt_decay=0, lambda_3=1, drop_out_p=0.2,
         noise_dimension_var=1, noise_dimension_type="normal", lambda_4=1,
         gpu_id=0):

    enable_cuda = True
    if torch.cuda.is_available() and enable_cuda:
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    if simulation == 'type1error':
        sim_x, sim_y, sim_z = generate_samples_random(
            size=n, sType='H0', dx=x_dims, dy=y_dims, dz=z_dim,
            nstd=nstd, alpha_x=a_x, dist_z=z_dist, preprocess=preprocess)
    elif simulation == 'power':
        sim_x, sim_y, sim_z = generate_samples_random(
            size=n, sType='H1', dx=x_dims, dy=y_dims, dz=z_dim,
            nstd=nstd, alpha_x=a_x, dist_z=z_dist, preprocess=preprocess)
    else:
        raise ValueError('Test does not exist.')

    x, y, z = sim_x.to(device), sim_y.to(device), sim_z.to(device)

    w_mx = torch.linalg.vector_norm(
        z.repeat(n, 1, 1) - torch.swapaxes(z.repeat(n, 1, 1), 0, 1), ord=1, dim=2)
    sigma_w = torch.median(w_mx).item()

    u_mx = torch.exp(-torch.abs(y.repeat(1, n) - y.repeat(1, n).T))
    sigma_u = torch.median(u_mx).item()

    v_mx = torch.exp(-torch.abs(x.repeat(1, n) - x.repeat(1, n).T))
    sigma_v = torch.median(v_mx).item()

    test_size = int(n / k)
    stat_all = torch.zeros(k, 1)
    boot_temp_all = torch.zeros(k, boot_num)
    cur_k = 0

    if not using_orcale:
        G_zy = Generator(z_dim, y_dims, noise_dimension, hidden_layer_size, False, 0.1, drop_out_p).to(device)
        G_zx = Generator(z_dim, x_dims, noise_dimension, hidden_layer_size, False, 0.1, drop_out_p).to(device)

    for k_fold in range(k):
        k_fold_start = int(n / k * k_fold)
        k_fold_end = int(n / k * (k_fold + 1))

        X_test = x[k_fold_start:k_fold_end]
        Y_test = y[k_fold_start:k_fold_end]
        Z_test = z[k_fold_start:k_fold_end]

        X_train = torch.cat((x[0:k_fold_start], x[k_fold_end:]))
        Y_train = torch.cat((y[0:k_fold_start], y[k_fold_end:]))
        Z_train = torch.cat((z[0:k_fold_start], z[k_fold_end:]))

        if k == 1:
            X_train, Y_train, Z_train = X_test, Y_test, Z_test

        train_xyz = DatasetSelect_GAN(X_train, Y_train, Z_train, batch_size)
        DataLoader_xyz = torch.utils.data.DataLoader(train_xyz, batch_size=batch_size, shuffle=True)

        if not using_orcale:
            current_epochs = epochs_num if k_fold == 0 else max(10, epochs_num // 5)

            G_zy, G_zx = train_ver3(
                G_zx=G_zx, G_zy=G_zy,
                X=X_train, Y=Y_train, Z=Z_train, M=M,
                X_test=X_test, Y_test=Y_test, Z_test=Z_test,
                noise_dimension=noise_dimension, noise_type=noise_dimension_type,
                G_lr=G_lr, hidden_layer_size=hidden_layer_size,
                DataLoader=DataLoader_xyz, BN_type=False, ReLU_coef=0.1,
                lambda_median=lambda_median, median_samples=median_samples,
                M_train=M_train,
                epochs_num=epochs_num,
                sigma_z=sigma_w, sigma_x=sigma_v, sigma_y=sigma_u,
                normal_ini=normal_ini, lambda_1=lambda_1, lambda_2=lambda_2,
                using_Gen=using_Gen, wgt_decay=wgt_decay, lambda_3=lambda_3,
                drop_out_p=drop_out_p, noise_dimension_var=noise_dimension_var,
                lambda_4=lambda_4,
                device=device)

        dataset_test = DatasetSelect(X_test, Y_test, Z_test)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

        gen_x_all = torch.zeros(test_size, M, device=device)
        gen_y_all = torch.zeros(test_size, M, device=device)

        z_all = torch.zeros(test_size, z_dim, device=device)
        x_all = torch.zeros(test_size, x_dims, device=device)
        y_all = torch.zeros(test_size, y_dims, device=device)

        cur_itr = 0

        if not using_orcale:
            G_zx = G_zx.eval()
            G_zy = G_zy.eval()

        for i, (x_test, y_test, z_test) in enumerate(dataloader_test):
            z_test_temp = z_test.repeat(M, 1)

            if not using_orcale:
                z_test_temp = z_test_temp.to(device)
                Noise_fake = sample_noise(z_test_temp.size()[0], noise_dimension,
                                          "normal", device=device).to(device)
                with torch.no_grad():
                    fake_x = G_zx(torch.cat((z_test_temp, Noise_fake), dim=1)).reshape(1, -1)

                Noise_fake = sample_noise(z_test_temp.size()[0], noise_dimension,
                                          "normal", device=device).to(device)
                with torch.no_grad():
                    fake_y = G_zy(torch.cat((z_test_temp, Noise_fake), dim=1)).reshape(1, -1)

            elif using_orcale:
                if simulation == 'type1error':
                    fake_x, fake_y = generate_samples_from_fixed_Z_random(
                        z_test_temp, size=M, sType='H0', dx=x_dims, dy=y_dims,
                        dz=z_dim, nstd=nstd, alpha_x=a_x, dist_z=z_dist)
                elif simulation == 'power':
                    fake_x, fake_y = generate_samples_from_fixed_Z_random(
                        z_test_temp, size=M, sType='H1', dx=x_dims, dy=y_dims,
                        dz=z_dim, nstd=nstd, alpha_x=a_x, dist_z=z_dist)

            gen_x_all[cur_itr, :] = fake_x.detach().reshape(-1)
            gen_y_all[cur_itr, :] = fake_y.detach().reshape(-1)
            x_all[cur_itr, :] = x_test
            y_all[cur_itr, :] = y_test
            z_all[cur_itr, :] = z_test

            cur_itr = cur_itr + 1

        cur_stat, cur_boot_temp = get_p_value_stat_1(
            boot_num, M, test_size,
            gen_x_all.to(device), gen_y_all.to(device),
            x_all.to(device), y_all.to(device), z_all.to(device),
            sigma_w, sigma_u, sigma_v, boor_rv_type,
            device=device)

        stat_all[cur_k, :] = cur_stat
        boot_temp_all[cur_k, :] = torch.from_numpy(cur_boot_temp)

        cur_k = cur_k + 1

    if using_orcale:
        gen_x_median = torch.median(gen_x_all, dim=1).values.reshape(-1, 1)
        gen_y_median = torch.median(gen_y_all, dim=1).values.reshape(-1, 1)

        mse_x = torch.mean((gen_x_median - x_all) ** 2).item()
        mse_y = torch.mean((gen_y_median - y_all) ** 2).item()

        print(f'Test MSE x (median diff) [{mse_x}], MSE y (median diff) [{mse_y}]')

    return np.mean(torch.mean(boot_temp_all, dim=0).numpy() > torch.mean(stat_all).item())


# ============================================================
# 并行部分
# ============================================================

from joblib import Parallel, delayed
import multiprocessing


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
    M_train = params.get("M_train", 1)
    k_value = params["k_value"]
    j_value = params["j_value"]
    noise_dimension = params["noise_dimension"]
    noise_dimension_type = params["noise_dimension_type"]
    noise_dimension_var = params["noise_dimension_var"]
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
    lambda_median = params["lambda_median"]
    median_samples = params["median_samples"]

    np.random.seed(set_seeds)
    torch.manual_seed(set_seeds)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(set_seeds)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_cores = min(20, n_test)

    if num_cores < 1:
        num_cores = 1

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始并行实验...")
    print(f"模式: {test}, 样本量: {sample_size}, 交叉验证折数: {k_value}, "
          f"模型中位数对齐超参数(pinball): {lambda_median}, "
          f"M_train: {M_train}, 实验次数: {n_test}, "
          f"并行核数: {num_cores}, 可用GPU数: {num_gpus}")

    if test == 'power':
        print(f"备择假设H_1下的模型参数 alpha_x: {alpha_x}")

    p_values = Parallel(n_jobs=num_cores)(
        delayed(mGAN)(
            n=sample_size, z_dim=z_dim, simulation=test, batch_size=batch_size,
            epochs_num=epochs_num, nstd=eps_std, z_dist=dist_z, x_dims=dx, y_dims=dy,
            a_x=alpha_x, M=m_value, k=k_value, boot_num=j_value,
            noise_dimension=noise_dimension, hidden_layer_size=hidden_layer_size,
            normal_ini=normal_ini, preprocess=preprocess, G_lr=G_lr,
            using_orcale=using_orcale, lambda_1=lambda_1, lambda_2=lambda_2,
            lambda_median=lambda_median, median_samples=median_samples,
            M_train=M_train,
            using_Gen=using_Gen, boor_rv_type=boor_rv_type, wgt_decay=wgt_decay,
            lambda_3=lambda_3, drop_out_p=drop_out_p,
            noise_dimension_var=noise_dimension_var,
            noise_dimension_type=noise_dimension_type, lambda_4=lambda_4,
            gpu_id=job_index % num_gpus
        )
        for job_index, _ in enumerate(range(n_test))
    )

    p_values = np.array(p_values)

    fp = [pval < alpha for pval in p_values]
    final_result = np.mean(fp)

    fp1 = [pval < alpha1 for pval in p_values]
    final_result1 = np.mean(fp1)

    print(f"\n" + "=" * 50)
    print(f"实验类型: {test.upper()}")
    print(f"Z Dimension: {z_dim}")
    print(f"Emp Rej Rate: {final_result:.4f} (at significance level {alpha})")
    print(f"Emp Rej Rate: {final_result1:.4f} (at significance level {alpha1})")
    print("=" * 50 + "\n")

    return p_values
