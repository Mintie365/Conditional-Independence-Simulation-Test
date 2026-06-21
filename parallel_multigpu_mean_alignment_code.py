param = {
  "test": "power", # ['type1error', 'power']
  "sample_size": 1000, # [1000, 1500]
  "batch_size": 256, # [32, 64, 128, 256]
  "z_dim": 200, # [50, 100, 150, 200, 250]
  "dx": 1,
  "dy": 1,
  "n_test": 100, # [200, 2000]
  "epochs_num": 200, # [1000, 1500]
  "eps_std" : 0.5,
  "dist_z" : 'gaussian', # ['laplace', 'gaussian']
  "alpha_x": 0.75, # only used under alternative [0.15, 0.30, 0.45, 0.60, 0.75]
  "m_value": 100, # [100, 200]
  "k_value": 2, # [1, 2, 4]
  "j_value": 1000, # [1000, 2000]
  "noise_dimension": 50, # [5, 10, 20]
  "hidden_layer_size": 512, # [64, 128, 256, 512, 1024]
  "normal_ini": False, # [True, False]
  "preprocess": 'normalize', # ['normalize',  'scale_Z', 'None' ]
  "G_lr": 2e-5, # [5e-6, 1e-5, 2e-5， 5e-5]
  "alpha": 0.1,
  "alpha1": 0.05,
  "set_seeds": 42,
  "using_orcale": False,
  "lambda_1": 1, # loss with Laplace kernel
  "lambda_2": 0,  # loss with Gaussian kernel
  "using_Gen": '1',  # ['1', '2'], types of generator "1" is fully connect, "2" is non fully
  "boor_rv_type":  'gaussian', # ['rademacher', 'gaussian']
  "wgt_decay": 1e-5, # weight decay for adam optimizer L2 regularization parameter
  "lambda_3": 1e-5, # L1 regularization parameter
  "lambda_4": 2e-5,
  "drop_out_p": 0.2, #  probability of an element to be zeroed. Default: 0.5, best 0.2
  "M_train": 10,
  "lambda_mean": 0.5,
  "mean_samples": 20,
  "workers_per_gpu": 2,
  "enable_cuda": True
}

import torch
import torch.nn as nn
import torch.distributions as TD
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from datetime import datetime
import functools
import os
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')


def get_available_gpus(enable_cuda=True):
    """Return visible CUDA GPU ids."""
    if enable_cuda and torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def set_process_device(gpu_id=None, enable_cuda=True):
    """Set the global torch device inside each joblib worker."""
    global device
    if enable_cuda and torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    return device


def set_all_seeds(seed):
    """Set numpy and torch seeds inside each worker."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_samples_random(Ax, Ay, size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=0.05, alpha_x=0.05,
               preprocess="None", dist_z='gaussian'):
    '''
    Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        we set f1 to be sin function and f2 to be cos function.
    Output:
        Samples X, Y, Z
    '''
    num = size

    if dist_z == 'gaussian':
        cov = np.eye(dz)
        mu = np.zeros(dz)
        Z = np.random.multivariate_normal(mu, cov, num)

    elif dist_z == 'laplace':
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num*dz)
        Z = np.reshape(Z, (num, dz))

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1)
    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / np.linalg.norm(Ay[:, i], ord=1)

    Axy = np.ones((dx, dy)) * alpha_x

    if sType == 'CI':
        X = np.sin(np.matmul(Z, Ax) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(np.matmul(Z, Ay) + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    elif sType == 'I':
        X = np.sin(nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    else:
        X = np.sin(np.matmul(Z, Ax) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(np.matmul(X, Axy) + np.matmul(Z, Ay) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))

    if preprocess == "normalize":
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())

    elif preprocess == "scale_Z":
        Z = Z / Z.max()

    elif preprocess == "None":
        X, Y, Z = X, Y, Z

    X, Y, Z = torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(Y)).float(), torch.from_numpy(np.array(Z)).float()
    return X, Y, Z

def generate_samples_from_fixed_Z_random(Ax, Ay, Z, size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=0.05, alpha_x=0.05,
                     normalize=True, seed=None, dist_z='gaussian'):
    '''
    Generate CI,I or NI post-nonlinear samples given fixed Z
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        we set f1 to be sin function and f2 to be cos function.
    Output:
        Samples X, Y, Z
    '''
    num = size

    error_generator_x = TD.MultivariateNormal(
        torch.zeros(dx), 1 * torch.eye(dx))

    error_generator_y = TD.MultivariateNormal(
        torch.zeros(dy), 1 * torch.eye(dy))

    Axy = torch.ones((dx, dy)) * alpha_x

    if sType == 'CI':
        X = torch.sin(torch.matmul(Z, Ax) + nstd * error_generator_x.sample((num,)) )##variance is 1, not 0.25, as mentioned in the paper
        Y = torch.cos(torch.matmul(Z, Ay) + nstd * error_generator_y.sample((num,)) )
    elif sType == 'I':
        X = torch.sin(nstd * error_generator_x.sample((num,)) )
        Y = torch.cos(nstd * error_generator_y.sample((num,)) )
    else:
        X = torch.sin(torch.matmul(Z, Ax) + nstd * error_generator_x.sample((num,)) )
        Y = torch.cos(torch.matmul(torch.sin(torch.matmul(Z, Ax) + nstd * error_generator_x.sample((num,)) ), Axy) + torch.matmul(Z, Ay) + nstd * error_generator_y.sample((num,)) )

    return X, Y

def get_p_value_stat_1(boot_num, M, n, gen_x_all_torch, gen_y_all_torch, x_torch, y_torch, z_torch, sigma_w, sigma_u = 1, sigma_v = 1,
                       boor_rv_type = "gaussian"):
    """
    Compute the p-value

    Input:
    - boot_num: Integer giving the number of bootstrap samples.
    - M: Integer giving the number of training samples per batch.
    - n: Integer giving the number of training samples.
    - gen_x_all_torch: PyTorch Tensor (batch_size, M) of generated data of X.
    - gen_y_all_torch: PyTorch Tensor (batch_size, M) of generated data of Y.
    - x_torch: PyTorch Tensor (batch_size) of training input X.
    - y_torch: PyTorch Tensor (batch_size) of training input Y.
    - z_torch: PyTorch Tensor (batch_size, dimension_Z) of training input Z
    - sigma_w: Float of the bandwith of the Laplace kernel.
    - sigma_u: Float of the bandwith of the Laplace kernel.
    - sigma_v: Float of the bandwith of the Laplace kernel.
    - boor_rv_type: "rademacher" or "gaussian", specifying the reference distribution.

    Output:
    - p_value: Float giving the p-value.
    """

    w_mx = torch.linalg.vector_norm(z_torch.repeat(n,1,1) - torch.swapaxes(z_torch.repeat(n,1,1), 0, 1), ord = 1, dim = 2)
    w_mx = torch.exp(-w_mx/sigma_w)

    u_mx_1 = torch.exp(-torch.abs(y_torch.repeat(1,n) - y_torch.repeat(1,n).T)/sigma_u)
    u_mx_2 = torch.mean(torch.exp(-torch.abs(gen_y_all_torch.repeat(n,1,1) - y_torch.repeat(1, n).reshape(n,n,1))/sigma_u), dim = 2)
    u_mx_3 = u_mx_2.T

    gen_y_all_torch_rep = gen_y_all_torch.repeat(n,1,1)

    temp_mx = gen_y_all_torch_rep[:,:,0].T
    sum_mx = torch.mean(torch.exp(-torch.abs(gen_y_all_torch_rep - temp_mx.reshape(n,n,1))/sigma_u), dim = 2)

    v_mx_1 = torch.exp(-torch.abs(x_torch.repeat(1,n) - x_torch.repeat(1,n).T)/sigma_v)
    v_mx_2 = torch.mean(torch.exp(-torch.abs(gen_x_all_torch.repeat(n,1,1) - x_torch.repeat(1, n).reshape(n,n,1))/sigma_v), dim = 2)
    v_mx_3 = v_mx_2.T

    gen_x_all_torch_rep = gen_x_all_torch.repeat(n,1,1)

    temp2_mx = gen_x_all_torch_rep[:,:,0].T
    sum2_mx = torch.mean(torch.exp(-torch.abs(gen_x_all_torch_rep - temp2_mx.reshape(n,n,1))/sigma_v), dim = 2)

    for i in range(1, M):
      temp_mx = gen_y_all_torch_rep[:,:,i].T
      temp_add_mx = torch.mean(torch.exp(-torch.abs(gen_y_all_torch_rep - temp_mx.reshape(n,n,1))/sigma_u), dim = 2)
      sum_mx = sum_mx + temp_add_mx

      temp2_mx = gen_x_all_torch_rep[:,:,i].T
      temp2_add_mx = torch.mean(torch.exp(-torch.abs(gen_x_all_torch_rep - temp2_mx.reshape(n,n,1))/sigma_v), dim = 2)
      sum2_mx = sum2_mx + temp2_add_mx

    u_mx_4 = 1/M*sum_mx
    u_mx = u_mx_1 - u_mx_2 - u_mx_3 + u_mx_4
    v_mx_4 = 1/M*sum2_mx
    v_mx = v_mx_1 - v_mx_2 - v_mx_3 + v_mx_4

    FF_mx = u_mx * v_mx *w_mx * (1-torch.eye(n).to(device))

    stat = 1/(n-1) * torch.sum(FF_mx).item()

    boottemp = np.array([])
    if boor_rv_type == "rademacher":
      eboot = torch.sign(torch.randn(n, boot_num)).to(device)
    elif boor_rv_type == "gaussian":
      eboot = torch.randn(n, boot_num).to(device)
    for bb in range(boot_num):
      random_mx = torch.matmul(eboot[:,bb].reshape(-1,1), eboot[:,bb].reshape(-1,1).T)
      bootmatrix = FF_mx * random_mx
      stat_boot = 1/(n-1) * torch.sum(bootmatrix).item()
      boottemp = np.append(boottemp, stat_boot)
    return stat, boottemp

class DatasetSelect(Dataset):
    """
    Create a DatasetSelect object to generate the DataLoader in the learning process.

    Input:
    - X: PyTorch Tensor of shape (N, input_dimension) giving the training data of X.
    - Y: PyTorch Tensor of shape (N, output_dimension) giving the training data of Y.
    - Z: PyTorch Tensor of shape (N, output_dimension) giving the training data of Z.
    """


    def __init__(self, X, Y, Z):
        self.X_real = X
        self.Y_real = Y
        self.Z_real = Z
        self.sample_size = X.shape[0]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        return self.X_real[index], self.Y_real[index], self.Z_real[index]

# Create a DataLoader for given (X, Y)

class DatasetSelect_GAN(torch.utils.data.Dataset):
    """
    Create a DatasetSelect object to generate the DataLoader in the learning process.

    Input:
    - X: PyTorch Tensor of shape (N, input_dimension) giving the training data of X.
    - Y: PyTorch Tensor of shape (N, output_dimension) giving the training data of Y.
    - batch_size: Integer giving the batch size.
    """

    def __init__(self, X, Y, Z, batch_size):
        self.X_real = X
        self.Y_real = Y
        self.Z_real = Z
        self.batch_size = batch_size
        self.sample_size = X.shape[0]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        return self.X_real[index], self.Y_real[index], self.Z_real[index], self.Z_real[(self.batch_size+index) % self.sample_size]

# Create a DataLoader for given (X, Y)

class DatasetSelect_GAN_ver2(torch.utils.data.Dataset):
  """
    Create a DatasetSelect object to generate the DataLoader in the learning process.

    Input:
    - X: PyTorch Tensor of shape (N, input_dimension) giving the training data of X.
    - Z: PyTorch Tensor of shape (N, output_dimension) giving the training data of Z.
    - batch_size: Integer giving the batch size.
  """

  def __init__(self, Y, Z, batch_size):
    self.Y_real = Y
    self.Z_real = Z
    self.batch_size = batch_size
    self.sample_size = Z.shape[0]

  def __len__(self):
    return self.sample_size

  def __getitem__(self, index):
    return self.Y_real[index], self.Z_real[index]

##### Auxilliary functions #####

def sample_noise(sample_size, noise_dimension, noise_type, input_var):
    """
    Generate a PyTorch Tensor of random noise from the specified reference distribution.

    Input:
    - sample_size: the sample size of noise to generate.
    - noise_dimension: the dimension of noise to generate.
    - noise_type: "normal", "unif" or "Cauchy", giving the reference distribution.

    Output:
    - A PyTorch Tensor of shape (sample_size, noise_dimension).
    """

    if (noise_type == "normal"):
      noise_generator = TD.MultivariateNormal(
        torch.zeros(noise_dimension).to(device), input_var * torch.eye(noise_dimension).to(device))

      Z = noise_generator.sample((sample_size,))
    if (noise_type == "unif"):
      Z = torch.rand(sample_size, noise_dimension)
    if (noise_type == "Cauchy"):
      Z = TD.Cauchy(torch.tensor([0.0]), torch.tensor([1.0])).sample((sample_size, noise_dimension)).squeeze(2)

    return Z

##### GAN architecture #####

class Generator(torch.nn.Module):
    """
    Specify the neural network architecture of the Generator.

    Here, we consider a FNN with a fully connected hidden layer with a width of 50,
    which is followed by a Leaky ReLU activation. The coefficient of Leaky ReLU needs to be
    specified. Batch normalization may be added prior to the activation function.
    The output layer a fully connected layer without activation.

    Inputs:
    - input_dimension: Integer giving the dimension of input Z.
    - output_dimension: Integer giving the dimension of output X or Y.
    - noise_dimension: Integer giving the dimension of random noise.
    - hidden_layer_size: Integer giving the size of the hidden layer of the generator.
    - BN_type: 'True' or 'False' specifying whether batch normalization is included.
    - ReLU_coef: Scalar giving the coefficient of the Leaky ReLU layer.
    - drop_out_p: Float giving the dropout probability.
    - drop_input: Boolean specifying whether to add dropout to the input layer.

    Returns:
    - x: PyTorch Tensor containing the (output_dimension,) output of the generator.
    """

    def __init__(self, input_dimension, output_dimension, noise_dimension, hidden_layer_size, BN_type, ReLU_coef, drop_out_p,
                 drop_input = False):
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
        # x = self.drop_out3(self.leakyReLU1(self.BN3(self.fc3(x))))
        x = self.fc_last(x)
      else:
        if self.drop_input:
            x = self.drop_out0(x)
        x = self.drop_out1(self.leakyReLU1(self.fc1(x)))
        x = self.drop_out2(self.leakyReLU1(self.fc2(x)))
        # x = self.drop_out3(self.leakyReLU1(self.fc3(x)))
        x = self.fc_last(x)
        x = self.sigmoid(x)
      return x

class NonFullyConnected_1(torch.nn.Module):

  def __init__(self, size_in, size_out, m, bias = True):
    super(NonFullyConnected_1, self).__init__()
    self.linear = torch.nn.Linear(m*size_in, m*size_out, bias = bias).to(device)
    self.mask = functools.reduce(torch.block_diag,[torch.ones(size_out, size_in) for i in range(m)]).to(device)

  def forward(self, x):

    self.linear.weight.data *= self.mask
    return self.linear(x)

class Generator_2(torch.nn.Module):
    def __init__(
            self,
            input_dimension,
            output_dimension,
            noise_dimension,
            hidden_layer_size,
            BN_type,
            ReLU_coef,
            hidden_layer_depth = 1,
            ntargets_k=5):
        super(Generator_2, self).__init__()
        self.input_dimension = input_dimension + noise_dimension
        self.output_dimension = output_dimension
        self.ntargets_k = ntargets_k
        self.hidden_layer_sizes = [hidden_layer_size] * hidden_layer_depth
        self.BN_type = BN_type
        self.leakyrelu = torch.nn.LeakyReLU(ReLU_coef)
        self.linear_layers_from_input = torch.nn.Linear(self.input_dimension, ntargets_k*self.hidden_layer_sizes[0])

        self.linear_layers_between = torch.nn.ModuleList([
            NonFullyConnected_1(self.hidden_layer_sizes[0],self.hidden_layer_sizes[0], ntargets_k)
            for i in range(len(self.hidden_layer_sizes))
        ])
        # self.linear8 = torch.nn.Linear(self.hidden_layer_sizes[0]*ntargets_k, self.hidden_layer_sizes[0]*ntargets_k)
        # self.linear8.weight = torch.nn.Parameter(torch.eye(self.hidden_layer_sizes[0]*ntargets_k), requires_grad=False)
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

        return self.linear8(output) # torch.mean(self.linear8(output)).reshape(1)

##### Training procedures #####


def find_loss(y_torch, gen_y_all_torch, z_torch, sigma_w, sigma_u, M ):
    """
    Compute the MMD loss via Laplace kernel.

    Inputs:
    - y_torch: PyTorch Tensor (batch_size) of training input. (X or Y)
    - gen_y_all_torch: PyTorch Tensor (batch_size, M) of generated data.
    - z_torch: PyTorch Tensor (batch_size, dimension_Z) of training input Z.
    - sigma_w: Float of the bandwith of the kernel.
    - sigma_u: Float of the bandwith of the kernel.
    - M: Number of training samples per batch.

    Outputs:
    - loss: PyTorch Tensor containing the MMD loss.
    """
    n = z_torch.shape[0]
    w_mx = torch.linalg.vector_norm(z_torch.repeat(n,1,1) - torch.swapaxes(z_torch.repeat(n,1,1), 0, 1), ord = 1, dim = 2)
    w_mx = torch.exp(-w_mx/sigma_w)

    u_mx_1 = torch.exp(-torch.abs(y_torch.repeat(1,n) - y_torch.repeat(1,n).T)/sigma_u)
    u_mx_2 = torch.mean(torch.exp(-torch.abs(gen_y_all_torch.repeat(n,1,1) - y_torch.repeat(1, n).reshape(n,n,1))/sigma_u), dim = 2)
    u_mx_3 = u_mx_2.T

    gen_y_all_torch_rep = gen_y_all_torch.repeat(n,1,1)

    temp_mx = gen_y_all_torch_rep[:,:,0].T
    sum_mx = torch.mean(torch.exp(-torch.abs(gen_y_all_torch_rep - temp_mx.reshape(n,n,1))/sigma_u), dim = 2)

    for i in range(1, M):
      temp_mx = gen_y_all_torch_rep[:,:,i].T
      temp_add_mx = torch.mean(torch.exp(-torch.abs(gen_y_all_torch_rep - temp_mx.reshape(n,n,1))/sigma_u), dim = 2)
      sum_mx = sum_mx + temp_add_mx


    u_mx_4 = 1/M*sum_mx
    u_mx = u_mx_1 - u_mx_2 - u_mx_3 + u_mx_4

    FF_mx = u_mx *w_mx * (1-torch.eye(n).to(device))

    loss =  1/(n) * torch.sum(FF_mx)
    return loss

def find_loss_2(Y, hat_Y, Z, sigma_z, sigma_y):
    """
    Compute the MMD loss via Gaussian kernel.

    Inputs:
    - y_torch: PyTorch Tensor (batch_size) of training input. (X or Y)
    - gen_y_all_torch: PyTorch Tensor (batch_size, M) of generated data.
    - z_torch: PyTorch Tensor (batch_size, dimension_Z) of training input Z.
    - sigma_w: Float of the bandwith of the kernel.
    - sigma_u: Float of the bandwith of the kernel.
    - M: Number of training samples per batch.

    Outputs:
    - loss: PyTorch Tensor containing the MMD loss.
    """

    n = Z.shape[0]
    mx_1_1 = torch.exp(-(Y.repeat(1,n) - Y.repeat(1,n).T) ** 2/sigma_y)
    mx_1_2 = torch.linalg.vector_norm(Z.repeat(n,1,1) - torch.swapaxes(Z.repeat(n,1,1), 0, 1), ord = 2, dim = 2)
    # sigma = torch.median(mx_1_2)
    mx_1_2 = torch.exp(-(mx_1_2 ** 2)/sigma_z)
    mx_1 = mx_1_1 * mx_1_2

    mx_2_1 = torch.exp(-(Y.repeat(1,n) - hat_Y.repeat(1,n).T) ** 2/sigma_y)
    mx_2 = mx_2_1 * mx_1_2

    mx_3 = mx_2.T

    mx_4_1 = torch.exp(-(hat_Y.repeat(1,n) - hat_Y.repeat(1,n).T) ** 2/sigma_y)
    mx_4 = mx_4_1 * mx_1_2

    FF_mx = (mx_1 - mx_2 - mx_3 + mx_4)
    loss =  1/(n**2) * torch.sum(FF_mx)
    return loss

def train_ver3(
    G_zx, G_zy, # 🌟 修改 1: 接收外部传入的模型，以便在 K 折之间重用权重
    X, Y, Z, X_test, Y_test, Z_test,
    noise_dimension, noise_type, G_lr, hidden_layer_size,
    DataLoader, BN_type, ReLU_coef,
    lambda_mean=0.5, mean_samples=20, # 🌟 修改 2: 新增均值对齐参数
    epochs_num=50,
    patience=20, min_delta=1e-5, # 🌟 修改 3: 新增 Early Stopping 参数
    sigma_z = 1, sigma_x = 1, sigma_y = 1,
    normal_ini = False,
    lambda_1 = 1, lambda_2 = 0, using_Gen = '1', wgt_decay = 0,
    lambda_3 = 0, lambda_4 = 0, drop_out_p = 0.2, M_train = 3):
    """
    Train loop for GAN (Adapted for Section 4.2 with Mean Alignment & Early Stopping)

    Inputs:
    - X: PyTorch Tensor (sample_size, dimension_X) of training input.
    - Y: PyTorch Tensor (sample_size, dimension_Y) of training input.
    - Z: PyTorch Tensor (sample_size, dimension_Z) of training input.
    - X_test: PyTorch Tensor (sample_size, dimension_X) of test input.
    - Y_test: PyTorch Tensor (sample_size, dimension_Y) of test input.
    - Z_test: PyTorch Tensor (sample_size, dimension_Z) of test input.
    - noise_dimension: Integer giving the dimension of random noise Z.
    - noise_type: "normal", "unif" or "Cauchy", giving the reference distribution.
    - G_lr: Float giving the learning rate of the generator.
    - hidden_layer_size: Integer giving the size of the hidden layer of the generator.
    - DataLoader: DataLoader object used to generate training batches.
    - BN_type: 'True' or 'False' specifying whether batch normalization is included.
    - ReLU_coef: Float giving the coefficient of the Leaky ReLU layer.
    - epochs_num: Number of epochs over the training dataset to use for training.
    - sigma_z: Float of the bandwith of the kernel.
    - sigma_x: Float of the bandwith of the kernel.
    - sigma_y: Float of the bandwith of the kernel.
    - normal_ini: Boolean specifying whether to initialize the generator with normal initialization.
    - lambda_1: Float giving the coefficient of the MMD loss using Laplace kernel.
    - lambda_2: Float giving the coefficient of the MMD loss using Gaussian kernel. (not using)
    - using_Gen: '1' or '2' specifying whether to use the first or second generator.(not using)
    - wgt_decay: Float giving the weight decay. (L2 regularization)
    - lambda_3: Scalar giving the coefficient of the L1 regularization.
    - drop_out_p: Float giving the dropout probability.
    - M_train: Number of training samples per batch used in the Laplace or Gaussian kernel.

    Outputs:
    - G_zy: PyTorch Net giving the trained generator.
    - G_zx: PyTorch Net giving the trained generator.
    """

    input_dimension = Z.shape[1]
    output_dimension_y = Y.shape[1]
    output_dimension_x = X.shape[1]

    # 🌟 修改 4: 只有当传入的 G_zy/G_zx 为 None 时才初始化，支持断点续训/权重重用
    if G_zy is None or G_zx is None:
        if using_Gen == '1':
            G_zy = Generator(input_dimension, output_dimension_y, noise_dimension, hidden_layer_size, BN_type, ReLU_coef, drop_out_p).to(device)
            G_zx = Generator(input_dimension, output_dimension_x, noise_dimension, hidden_layer_size, BN_type, ReLU_coef, drop_out_p).to(device)
        elif using_Gen == '2':
            G_zy = Generator_2(input_dimension, output_dimension_y, noise_dimension, hidden_layer_size, BN_type, ReLU_coef).to(device)
            G_zx = Generator_2(input_dimension, output_dimension_x, noise_dimension, hidden_layer_size, BN_type, ReLU_coef).to(device)

        if normal_ini:
            for p in G_zy.parameters():
                p.data = torch.randn(p.shape, device=device) / np.sqrt(float(hidden_layer_size * 2))
            for p in G_zx.parameters():
                p.data = torch.randn(p.shape, device=device) / np.sqrt(float(hidden_layer_size * 2))

    # 每一折都需要重新定义优化器，清空动量
    G_zy_solver = optim.Adam(G_zy.parameters(), lr=G_lr, betas=(0.5, 0.999), weight_decay=wgt_decay)
    G_zx_solver = optim.Adam(G_zx.parameters(), lr=G_lr, betas=(0.5, 0.999), weight_decay=wgt_decay)

    iter_count = 0
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

            # -------------------------------------------------------------
            # 🌟 修改 5: 均值对齐 (Mean Alignment) 数据准备
            # -------------------------------------------------------------
            Z_repeated_mean = Z_real.repeat_interleave(mean_samples, dim=0)
            Noise_for_mean = sample_noise(Z_repeated_mean.shape[0], noise_dimension, noise_type, input_var=1.0/3.0).to(device)

            # 计算 G_zy 的均值损失 (MSE)
            Y_generated_group = G_zy(torch.cat((Z_repeated_mean, Noise_for_mean), dim=1))
            Y_mean_pred = torch.mean(Y_generated_group.reshape(batch_size, mean_samples, -1), dim=1)
            loss_mean_y = torch.nn.functional.mse_loss(Y_mean_pred, Y_real)

            # 计算 G_zx 的均值损失 (MSE)
            X_generated_group = G_zx(torch.cat((Z_repeated_mean, Noise_for_mean), dim=1))
            X_mean_pred = torch.mean(X_generated_group.reshape(batch_size, mean_samples, -1), dim=1)
            loss_mean_x = torch.nn.functional.mse_loss(X_mean_pred, X_real)


            # -------------------------------------------------------------
            # 原有的 MMD Loss 计算逻辑 (保留 M_train)
            # -------------------------------------------------------------
            Z_real_repeat = Z_real.repeat(M_train, 1)

            # Generate fake data for MMD
            Noise_fake = sample_noise(Z_real_repeat.shape[0], noise_dimension, noise_type, input_var=1.0/3.0).to(device)
            Y_fake = G_zy(torch.cat((Z_real_repeat, Noise_fake), dim=1)).to(device)

            Noise_fake = sample_noise(Z_real_repeat.shape[0], noise_dimension, noise_type, input_var=1.0/3.0).to(device)
            X_fake = G_zx(torch.cat((Z_real_repeat, Noise_fake), dim=1)).to(device)

            Y_fake = Y_fake.reshape(batch_size, M_train)
            X_fake = X_fake.reshape(batch_size, M_train)

            # Generator step for Y
            G_zy_solver.zero_grad()
            l1_regularization_first_layer_y = 0
            l1_regularization_rest_layers_y = 0
            for name, param in G_zy.named_parameters():
                if "fc1" in name:
                    l1_regularization_first_layer_y += torch.linalg.vector_norm(param, ord=1)
                else:
                    l1_regularization_rest_layers_y += torch.linalg.vector_norm(param, ord=1)


            # 🌟 修改 6: 组合 MMD Loss 和 Mean Loss
            mmd_loss_y = (lambda_1 * find_loss(Y_real, Y_fake, Z_real, sigma_z, sigma_y, M_train) +
              lambda_3 * l1_regularization_rest_layers_y +
              lambda_4 * l1_regularization_first_layer_y)


            g_zy_error = mmd_loss_y + lambda_mean * loss_mean_y

            g_zy_error.backward()
            torch.nn.utils.clip_grad_norm_(G_zy.parameters(), max_norm=0.5)
            G_zy_solver.step()


            # Generator step for X
            G_zx_solver.zero_grad()
            l1_regularization_first_layer_x = 0
            l1_regularization_rest_layers_x = 0
            for name, param in G_zx.named_parameters():
                if "fc1" in name:
                    l1_regularization_first_layer_x += torch.linalg.vector_norm(param, ord=1)
                else:
                    l1_regularization_rest_layers_x += torch.linalg.vector_norm(param, ord=1)
            # 🌟 修改 6: 组合 MMD Loss 和 Mean Loss
            mmd_loss_x = (lambda_1 * find_loss(X_real, X_fake, Z_real, sigma_z, sigma_x, M_train) +
              lambda_3 * l1_regularization_rest_layers_x +
              lambda_4 * l1_regularization_first_layer_x)

            g_zx_error = mmd_loss_x + lambda_mean * loss_mean_x

            g_zx_error.backward()
            torch.nn.utils.clip_grad_norm_(G_zx.parameters(), max_norm=0.5)
            G_zx_solver.step()

            iter_count += 1
            batch_count += 1

        # -------------------------------------------------------------
        # 🌟 修改 7: Early Stopping 逻辑判定
        # -------------------------------------------------------------
        current_loss = g_zx_error + g_zy_error

        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    return G_zy, G_zx



def mGAN(Ax, Ay, n=500, z_dim=100, simulation='type1error', batch_size=64, epochs_num=1000,
         nstd=1.0, z_dist='gaussian', x_dims=1, y_dims=1, a_x=0.05, M=500, k=2, boot_num=1000,
         noise_dimension = 10, hidden_layer_size = 512, normal_ini = False, preprocess = 'normalize',
         G_lr = 1e-5, using_orcale = False, lambda_1 = 1, lambda_2 = 0, using_Gen = '1',
         boor_rv_type = "gaussian", wgt_decay = 0, lambda_3 = 1, lambda_4 = 0, drop_out_p = 0.2, exp_num = 0, M_train = 3,
         lambda_mean = 0.3, mean_samples = 20):  # <--- 创新点：增加均值对齐惩罚参数
    """
    Compute the test statistics
    """
    if simulation == 'type1error':
        sim_x, sim_y, sim_z = generate_samples_random(Ax, Ay, size=n, sType='CI', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd, alpha_x=a_x,
                                  dist_z=z_dist, preprocess = preprocess)
    elif simulation == 'power':
        sim_x, sim_y, sim_z = generate_samples_random(Ax, Ay, size=n, sType='dependent', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd,
                                  alpha_x=a_x, dist_z=z_dist, preprocess = preprocess)
    else:
        raise ValueError('Test does not exist.')

    # 优化点 1：数据集中放至 GPU
    x, y, z = sim_x.to(device), sim_y.to(device), sim_z.to(device)

    w_mx = torch.linalg.vector_norm(z.repeat(n,1,1) - torch.swapaxes(z.repeat(n,1,1), 0, 1), ord = 1, dim = 2)
    sigma_w_train = torch.median(w_mx).item()

    u_mx = torch.abs(y.repeat(1, n) - y.repeat(1, n).T)
    sigma_u_train = torch.median(u_mx).item()

    v_mx = torch.abs(x.repeat(1, n) - x.repeat(1, n).T)
    sigma_v_train = torch.median(v_mx).item()

    test_size = int(n/k)
    stat_all = torch.zeros(k, 1)
    boot_temp_all = torch.zeros(k, boot_num)
    cur_k = 0

    mse_x_list = np.array([])
    mse_y_list = np.array([])

    # 优化点 2：交叉验证外初始化模型，实现权重复用以加速收敛
    if not using_orcale:
        if using_Gen == '1':
            G_zy = Generator(z_dim, y_dims, noise_dimension, hidden_layer_size, False, 0.1, drop_out_p).to(device)
            G_zx = Generator(z_dim, x_dims, noise_dimension, hidden_layer_size, False, 0.1, drop_out_p).to(device)
        elif using_Gen == '2':
            G_zy = Generator_2(z_dim, y_dims, noise_dimension, hidden_layer_size, False, 0.1).to(device)
            G_zx = Generator_2(z_dim, x_dims, noise_dimension, hidden_layer_size, False, 0.1).to(device)

    for k_fold in range(k):
        k_fold_start = int(n/k * k_fold)
        k_fold_end = int(n/k * (k_fold+1))
        X_test, Y_test, Z_test = x[k_fold_start:k_fold_end], y[k_fold_start:k_fold_end], z[k_fold_start:k_fold_end]
        X_train, Y_train, Z_train = torch.cat((x[0:k_fold_start], x[k_fold_end:])), torch.cat((y[0:k_fold_start], y[k_fold_end:])), torch.cat((z[0:k_fold_start], z[k_fold_end:]))

        if (k == 1):
            X_train, Y_train, Z_train = X_test, Y_test, Z_test

        train_xyz = DatasetSelect_GAN(X_train, Y_train, Z_train, batch_size)
        DataLoader_xyz = torch.utils.data.DataLoader(train_xyz, batch_size=batch_size, shuffle=True)

        if not using_orcale:
            # 优化点 3：后续折使用较少的 epoch 动态微调
            current_epochs = epochs_num if k_fold == 0 else max(10, epochs_num // 5)

            # 将均值对齐参数透传给 train_ver3
            G_zy, G_zx = train_ver3(
                G_zx=G_zx, G_zy=G_zy,
                X = X_train, Y = Y_train, Z = Z_train,
                X_test = X_test, Y_test = Y_test, Z_test = Z_test,
                noise_dimension = noise_dimension, noise_type = "normal",
                G_lr = G_lr, hidden_layer_size = hidden_layer_size,
                DataLoader = DataLoader_xyz, BN_type = False, ReLU_coef = 0.1,
                epochs_num=current_epochs,
                sigma_z = sigma_w_train, sigma_x = sigma_v_train, sigma_y = sigma_u_train,
                normal_ini = normal_ini, lambda_1 = lambda_1, lambda_2 = lambda_2,
                using_Gen = using_Gen, wgt_decay = wgt_decay, lambda_3 = lambda_3,
                drop_out_p = drop_out_p, M_train = M_train,
                lambda_mean = lambda_mean, mean_samples = mean_samples) # <--- 创新点：向训练循环透传均值参数

        dataset_test = DatasetSelect(X_test, Y_test, Z_test)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

        # 优化点 4：提前放在 device 上
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
            z_test_temp = z_test.repeat(M,1)

            if not using_orcale:
                z_test_temp = z_test_temp.to(device)
                Noise_fake = sample_noise(z_test_temp.size()[0], noise_dimension, "normal", input_var = 1.0/3.0).to(device)
                # 优化点 5：推断时切断梯度图以节省显存
                with torch.no_grad():
                    fake_x = G_zx(torch.cat((z_test_temp, Noise_fake),dim=1)).reshape(1, -1)

                Noise_fake = sample_noise(z_test_temp.size()[0], noise_dimension, "normal", input_var = 1.0/3.0).to(device)
                with torch.no_grad():
                    fake_y = G_zy(torch.cat((z_test_temp, Noise_fake),dim=1)).reshape(1, -1)
            elif using_orcale:
                if simulation == 'type1error':
                    fake_x, fake_y = generate_samples_from_fixed_Z_random(Ax, Ay, z_test_temp, size=M, sType='CI', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd, alpha_x=a_x, dist_z=z_dist)
                elif simulation == 'power':
                    fake_x, fake_y = generate_samples_from_fixed_Z_random(Ax, Ay, z_test_temp, size=M, sType='dependent', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd, alpha_x=a_x, dist_z=z_dist)

            gen_x_all[cur_itr,:] = fake_x.detach().reshape(-1)
            gen_y_all[cur_itr,:] = fake_y.detach().reshape(-1)
            x_all[cur_itr,:] = x_test
            y_all[cur_itr,:] = y_test
            z_all[cur_itr,:] = z_test
            cur_itr = cur_itr + 1

        standardise = True

        if standardise:
            gen_x_all = (gen_x_all - torch.mean(gen_x_all, dim=0, keepdim=True)) / torch.std(gen_x_all, dim=0, keepdim=True)
            gen_y_all = (gen_y_all - torch.mean(gen_y_all, dim=0, keepdim=True)) / torch.std(gen_y_all, dim=0, keepdim=True)
            x_all = (x_all - torch.mean(x_all, dim=0, keepdim=True)) / torch.std(x_all, dim=0, keepdim=True)
            y_all = (y_all - torch.mean(y_all, dim=0, keepdim=True)) / torch.std(y_all, dim=0, keepdim=True)
            z_all = (z_all - torch.mean(z_all, dim=0, keepdim=True)) / torch.std(z_all, dim=0, keepdim=True)

        w_mx = torch.linalg.vector_norm(z_all.repeat(test_size,1,1) - torch.swapaxes(z_all.repeat(test_size,1,1), 0, 1), ord = 1, dim = 2)
        sigma_w = torch.median(w_mx).item()

        u_mx = torch.abs(y_all.repeat(1, test_size) - y_all.repeat(1, test_size).T)
        sigma_u = torch.median(u_mx).item()

        v_mx = torch.abs(x_all.repeat(1, test_size) - x_all.repeat(1, test_size).T)
        sigma_v = torch.median(v_mx).item()

        cur_stat, cur_boot_temp = get_p_value_stat_1(boot_num, M, test_size, gen_x_all, gen_y_all,
                                x_all, y_all, z_all, sigma_w, sigma_u, sigma_v, boor_rv_type)
        stat_all[cur_k,:] = cur_stat
        boot_temp_all[cur_k,:] = torch.from_numpy(cur_boot_temp)
        cur_k = cur_k + 1

    return np.mean(torch.mean(boot_temp_all, dim = 0).numpy() > torch.mean(stat_all).item() )

# =========================
# Parallel experiment runner
# =========================

def run_experiment(params):
    """
    Run repeated mGAN experiments in parallel.

    Multi-GPU scheduling rule:
        - If CUDA is available, trials are assigned to GPUs round-robin:
              trial i -> gpu_ids[i % num_gpus]
        - workers_per_gpu controls how many parallel joblib workers are launched per GPU.
        - Example: 4 GPUs and workers_per_gpu=2 -> n_jobs=8.

    Notes:
        - This function only adds multi-GPU parallel execution and keeps your mGAN / DGP logic unchanged.
        - Each trial gets an independent seed: set_seeds + exp_index * 100003.
    """
    # -------------------------
    # Extract parameters
    # -------------------------
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
    lambda_4 = params.get("lambda_4", 0.0)
    drop_out_p = params["drop_out_p"]
    M_train = params["M_train"]
    lambda_mean = params.get("lambda_mean", 0.3)
    mean_samples = params.get("mean_samples", 20)
    enable_cuda_local = params.get("enable_cuda", True)
    workers_per_gpu = int(params.get("workers_per_gpu", 1))
    requested_n_jobs = params.get("n_jobs", None)

    # -------------------------
    # Decide available devices and n_jobs
    # -------------------------
    gpu_ids = get_available_gpus(enable_cuda_local)
    if len(gpu_ids) > 0:
        if requested_n_jobs is None:
            n_jobs = min(n_test, max(1, len(gpu_ids) * workers_per_gpu))
        else:
            n_jobs = min(n_test, int(requested_n_jobs))
        device_msg = f"Available GPUs: {gpu_ids} | workers_per_gpu: {workers_per_gpu} | n_jobs: {n_jobs}"
    else:
        cpu_cores = max(1, multiprocessing.cpu_count() - 2)
        if requested_n_jobs is None:
            n_jobs = min(n_test, cpu_cores)
        else:
            n_jobs = min(n_test, int(requested_n_jobs), cpu_cores)
        device_msg = f"No CUDA GPU detected; using CPU | CPU workers: {n_jobs}"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始并行实验...")
    print(f"模式: {test} | 样本量: {sample_size} | z_dim: {z_dim} | 折数: {k_value} | 迭代数: {n_test}")
    print(device_msg)
    print(f"lambda_mean: {lambda_mean} | mean_samples: {mean_samples} | M_train: {M_train}")
    if test == "power":
        print(f"备择假设(Power) 下参数 alpha_x: {alpha_x}")

    # -------------------------
    # One independent Monte Carlo trial
    # -------------------------
    def single_trial_mGAN(exp_index):
        # Assign GPU by trial index. If no GPU, run on CPU.
        gpu_id = gpu_ids[exp_index % len(gpu_ids)] if len(gpu_ids) > 0 else None
        set_process_device(gpu_id=gpu_id, enable_cuda=enable_cuda_local)

        # Independent seed for this trial.
        trial_seed = int(set_seeds + exp_index * 100003)
        set_all_seeds(trial_seed)

        # Generate projection matrices for this trial.
        # Kept the same style as your current code.
        a_f = torch.rand((z_dim, dx), device=device)
        l1_norm_a_f = torch.linalg.vector_norm(a_f, ord=1)
        Ax = a_f / torch.clamp(l1_norm_a_f, min=1e-12)

        a_g = torch.rand((z_dim, dy), device=device)
        l1_norm_a_g = torch.linalg.vector_norm(a_g, ord=1)
        Ay = a_g / torch.clamp(l1_norm_a_g, min=1e-12)

        try:
            p_val = mGAN(
                Ax=Ax, Ay=Ay,
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
            return float(p_val)
        finally:
            # Release cached GPU memory inside this worker after each trial.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------
    # Parallel execution
    # -------------------------
    p_values = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(single_trial_mGAN)(i) for i in range(n_test)
    )
    p_values = np.array(p_values, dtype=float)

    # -------------------------
    # Result summary
    # -------------------------
    final_result = np.mean(p_values < alpha)
    final_result1 = np.mean(p_values < alpha1)

    print("\n" + "=" * 60)
    print(f"实验结束 - 类型: {test.upper()} | Z Dimension: {z_dim}")
    print(f"Emp Rej Rate: {final_result:.4f} at alpha = {alpha}")
    print(f"Emp Rej Rate: {final_result1:.4f} at alpha1 = {alpha1}")
    print("=" * 60 + "\n")

    return p_values


# Example:
# p_values = run_experiment(param)
