

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
from functions import estimate_operator_norm
from huber_TV import power_iteration
from datasets import NaturalDataset, TestDataset, ImageBlurDataset, my_collate
from nonsmooth_optimisers import
from nonsmooth_algorithms import PGD, FISTA, FISTA2, PDHG, AcceleratedPDHG, ADMM
from test_fns import get_boxplot
from torch.utils.data import ConcatDataset
from grad_x import grad, laplacian
import time
import pandas as pd
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_models = False
alpha = 0.01
NUM_IMAGES = 200
NUM_TEST = 10
#noise_list = list(np.linspace(0.1, 0.4, 10))
noise_list = [0.]
sigma_list = list(np.linspace(2, 8, 6))
noise_list_test = [0.]
#noise_list_test = list(np.linspace(0.2, 0.5, 10))
sigma_list_test = list(np.linspace(3, 10, 7))
#sigma_list_test = list(np.linspace(0.00001, 5, 10))
num_batch=8
wk_list = [1]
wk_list2 = [-1]

n_iters=10
## what about n_iters = 1, but then what about learned on this new generated one to get the next one? Will this just simplify to n_iters=10 for the learned one?
NUM_EPOCHS = 10

### FOR SOLVING f+g, f IS DIFF, g ISN'T

def f(x,y,A_func, alpha):
    return data_fidelity(x,y,A_func) + alpha * TotalVariation(x)

def data_fidelity(x,y,A_func):
    return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2)

def grad_data_fidelity(x, y ,A_func, A_adj):
    return A_adj(A_func(x) - y)

def Du(u):
    """Compute the discrete gradient of u using PyTorch."""
    diff_x = torch.diff(u, dim=1, prepend=u[:, :1])
    diff_y = torch.diff(u, dim=0, prepend=u[:1, :])
    return diff_x, diff_y

def TotalVariation(u):
    """Compute the total variation norm ||Du||_{2,1} for a given tensor u."""
    diff_x, diff_y = Du(u)
    norm_2_1 = torch.sum(torch.sqrt(diff_x**2 + diff_y**2))
    return norm_2_1


def prox_dual_norm(p):
    """Compute the proximal operator of the dual norm ||.||_{2,inf} for the given tensor p."""
    norm_p = torch.sqrt(p[:, :, 0] ** 2 + p[:, :, 1] ** 2).unsqueeze(-1)
    return torch.where(norm_p > 1, p / norm_p, p)


def prox_total_variation(v, tau):
    """Compute the proximal operator of the total variation norm for the given tensor v."""
    # Compute Dv
    diff_x, diff_y = Du(v)
    p = torch.stack([diff_x, diff_y], dim=-1)

    # Compute proximal operator of the dual norm
    prox_p = prox_dual_norm(p / tau)

    # Update v
    result = v - tau * (Du(prox_p[:, :, 0])[0] + Du(prox_p[:, :, 1])[1])
    return result


# Path to the folder containing the images
folder_path = 'Images_128/'
dataset = NaturalDataset(folder_path, num_imgs=NUM_IMAGES)
test_dataset = TestDataset(folder_path, num_imgs=NUM_TEST)

# blurred_list = ImageBlurDatasetGrad(dataset, alpha, noise_list, sigma_list)
blurred_list = ImageBlurDataset(dataset, alpha, noise_list, sigma_list, f, grad_f)

test_set = ImageBlurDataset(test_dataset, alpha, noise_list_test, sigma_list_test, f, grad_f)

# combined_dataset = ConcatDataset([dataset1, dataset2])

train_loader = DataLoader(dataset=blurred_list, batch_size=num_batch, shuffle=True, num_workers=0,
                          collate_fn=my_collate)