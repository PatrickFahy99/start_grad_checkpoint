

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
from functions import estimate_operator_norm
from huber_TV import power_iteration
from datasets import NaturalDataset, TestDataset, ImageBlurDataset, my_collate
from optimisers import TauFuncNet, TauFunc10Net, TauFuncUnboundedNet, TauFuncUnboundedAboveNet, UpdateModel, UnrollingFunc, GeneralUpdateModel, FixedModel, FixedMomentumModel, AdagradModel, RMSPropModel, AdamModel, CNN_LSTM
from algorithms import function_evals, gradient_descent_fixed, gradient_descent_backtracking, gradient_descent_function, gradient_descent_update, gradient_descent_modelfree, gradient_descent_unrolling, gradient_descent_fixed_nesterov, gradient_descent_fixed_momentum, gradient_descent_heavy_ball, adagrad, rmsprop, adam, accelerated_gradient_descent
from test_fns import get_boxplot
from torch.utils.data import ConcatDataset
from grad_x import grad, laplacian
import time
import pandas as pd
import seaborn as sns
from torch.nn import HuberLoss



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_models = False
alpha = 0.0001
NUM_IMAGES = 10*4
NUM_TEST = 10
#noise_list = list(np.linspace(0.1, 0.4, 10))
noise_list = [0.0196, 0.0392, 0.0588]
sigma_list = [4]#list(np.linspace(2, 8, 6))
noise_list_test = [0.]
#noise_list_test = list(np.linspace(0.2, 0.5, 10))
sigma_list_test = list(np.linspace(3, 10, 7))
#sigma_list_test = list(np.linspace(0.00001, 5, 10))
num_batch=4
wk_list = [1]
wk_list2 = [-1]

n_iters=1000
num_iters=1000
## what about n_iters = 1, but then what about learned on this new generated one to get the next one? Will this just simplify to n_iters=10 for the learned one?
NUM_EPOCHS = 1000

def huber(s, epsilon=0.01):
    return torch.where(torch.abs(s) <= epsilon, (0.5 * s ** 2) / epsilon, s - 0.5 * epsilon)

def f(x,y,A_func, alpha):
    return data_fidelity(x,y,A_func) + alpha * huber_total_variation(x)

def data_fidelity(x,y,A_func):
    return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2)

def grad_f(x, y ,A_func, A_adj, alpha):
    return A_adj(A_func(x) - y) + alpha * laplacian(x)


def huber_total_variation(u, eps=0.05):
    diff_x, diff_y = Du(u)
    #norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2), eps))
    zeros = torch.zeros_like(torch.sqrt(diff_x**2 + diff_y**2))
    norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2+1e-08)))
    #norm_2_1 = torch.sum(diff_x**2 + diff_y**2)
    return norm_2_1

def grad_data_fidelity(x, y ,A_func, A_adj):
    return A_adj(A_func(x) - y)

def reg(x, alpha):
    return alpha * huber_total_variation(x)

def Du(u):
    """Compute the discrete gradient of u using PyTorch."""
    diff_x = torch.diff(u, dim=1, prepend=u[:, :1])
    diff_y = torch.diff(u, dim=0, prepend=u[:1, :])
    return diff_x, diff_y

# Path to the folder containing the images
folder_path = 'lhq_256/'
dataset = NaturalDataset(folder_path, num_imgs=NUM_IMAGES)
test_dataset = TestDataset(folder_path, num_imgs=NUM_TEST)

# blurred_list = ImageBlurDatasetGrad(dataset, alpha, noise_list, sigma_list)
blurred_list = ImageBlurDataset(dataset, alpha, noise_list, sigma_list, f, grad_f)

test_set = ImageBlurDataset(test_dataset, alpha, noise_list_test, sigma_list_test, f, grad_f)

# combined_dataset = ConcatDataset([dataset1, dataset2])

train_loader = DataLoader(dataset=blurred_list, batch_size=num_batch, shuffle=True, num_workers=0,
                          collate_fn=my_collate)

tau_learned = torch.nn.Parameter(torch.tensor([2.0], requires_grad=True))
optimizer_fixed = torch.optim.Adam([tau_learned])

if not load_models:
    print('START TRAINING')
    for epoch in range(NUM_EPOCHS):  # Number of epochs
        torch.cuda.empty_cache()
        if epoch+1<num_iters:
            n_iters=epoch+1
        else:
            n_iters=num_iters
        epoch_obj = 0
        epoch_obj_update = 0
        epoch_obj_10 = 0
        epoch_obj_free = 0
        epoch_obj_fixed = 0
        epoch_obj_unrolling = 0
        epoch_obj_last = 0
        epoch_obj_ubd_above = 0
        epoch_obj_ubd = 0
        epoch_obj_fixed_momentum = 0
        epoch_obj_fixed_nesterov = 0
        epoch_obj_fixed_heavy = 0
        epoch_obj_fixed_adagrad = 0
        epoch_obj_fixed_rmsprop = 0
        epoch_obj_fixed_adam = 0
        epoch_obj_accelerated = 0
        epoch_obj_LBFGS = 0
        for i, batch in enumerate(train_loader):
            total_objective = 0
            total_objective_update = 0
            total_objective_ubd_above = 0
            total_objective_ubd = 0
            total_objective_free = 0
            total_objective_fixed = 0
            total_objective_unrolling = 0
            total_objective_last = 0
            total_objective_10 = 0
            total_objective_fixed_momentum = 0
            total_objective_fixed_nesterov = 0
            total_objective_fixed_heavy = 0
            total_objective_fixed_adagrad = 0
            total_objective_fixed_rmsprop = 0
            total_objective_fixed_adam = 0
            total_objective_accelerated = 0
            total_objective_LBFGS = 0
            img, y, x0, std, sig, epsilon, A_func, A_adj, f, grad_f = batch
            for j in range(img.shape[0]):  # iterate over items in the batch
                img_j = img[j].to(device)
                y_j = y[j].to(device)
                x0_j = x0[j].to(device)
                epsilon_j = epsilon[j].to(device)
                A_func_j = A_func[j]
                A_adj_j = A_adj[j]
                f_j = f[j]
                grad_f_j = grad_f[j]
                std_j = std[j].to(device)
                sig_j = sig[j].to(device)

                std_j = torch.tensor(0.)


                ## Fixed and learned as a function
                xs, taus = gradient_descent_fixed(f_j, x0_j, y_j, tau_learned, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_fixed += obj

            total_objective_fixed /= (num_batch)


            total_objective_fixed.backward()
            optimizer_fixed.step()
            optimizer_fixed.zero_grad()

            epoch_obj_fixed += total_objective_fixed

        epoch_obj_fixed /= (NUM_IMAGES/num_batch)


        print(f"Epoch: {epoch}, Fixed: {epoch_obj_fixed}, tau: {tau_learned}")