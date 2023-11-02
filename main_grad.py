

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


def KL_divergence(y, z):
    ## return the KL divergence between y and z (two vectors)
    return torch.sum(z * torch.log(z / y) - z + y)

def KL_data_fit(x, y, A_func):
    return KL_divergence(A_func(x), y)

def poisson_noise(mean, size):
    return torch.poisson(torch.ones(size)*mean)

def wk_dec(T):
    denom = (T/2)*(T+1)
    wk_list = [T*(T-i)/denom for i in range(T)]
    return wk_list

def wk_exp_dec(T, lambd = 0.99):
    a = (1-lambd)/(1-lambd**T)
    wk_list = [T*a*(lambd**i) for i in range(T)]
    return wk_list

def wk_increasing(T):
    denom = (T/2)*(T+1)
    wk_list = [i/denom for i in range(1,T+1)]
    return wk_list

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

n_iters=40
num_iters=40
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

if load_models == True:
    try:
        tau_model = torch.load('grad_tau_model.pth')
    except:
        tau_model = TauBetaFuncNet().to(device)
    try:
        hb_func_model = torch.load('grad_hb_func_model.pth')
    except:
        hb_func_model = TauBetaFuncNet().to(device)
    try:
        tau_model_10 = torch.load('grad_tau_model10.pth')
    except:
        tau_model_10 = TauFunc10Net().to(device)
    try:
        tau_model_last = torch.load('grad_tau_model_last.pth')
    except:
        tau_model_last = TauFuncNet().to(device)
    try:
        tau_model_ubd_above = torch.load('grad_tau_model_ubd_above.pth')
    except:
        tau_model_ubd_above = TauFuncUnboundedAboveNet().to(device)
    try:
        tau_model_ubd = torch.load('grad_tau_model_ubd.pth')
    except:
        tau_model_ubd = TauFuncUnboundedNet().to(device)
    try:
        update_model = torch.load('grad_update_model.pth')
    except:
        update_model = CNN_LSTM().to(device)#UpdateModel().to(device)
    try:
        free_model = torch.load('grad_free_model.pth')
    except:
        free_model = CNN_LSTM().to(device)#GeneralUpdateModel().to(device)
    try:
        unrolling_model = torch.load('grad_unrolling_func_model.pth')
    except:
        unrolling_model = UnrollingFunc().to(device)
    try:
        fixed_model = torch.load('grad_fixed_model.pth')
    except:
        fixed_model = FixedModel().to(device)
    try:
        fixed_momentum_model = torch.load('grad_fixed_momentum_model.pth')
    except:
        fixed_momentum_model = FixedMomentumModel().to(device)
    try:
        fixed_nesterov_model = torch.load('grad_fixed_nesterov_model.pth')
    except:
        fixed_nesterov_model = FixedMomentumModel().to(device)
    try:
        fixed_heavy_model = torch.load('grad_fixed_heavy_model.pth')
    except:
        fixed_heavy_model = FixedMomentumModel().to(device)
    try:
        fixed_adagrad_model = torch.load('grad_fixed_adagrad_model.pth')
    except:
        fixed_adagrad_model = AdagradModel().to(device)
    try:
        fixed_rmsprop_model = torch.load('grad_fixed_rmsprop_model.pth')
    except:
        fixed_rmsprop_model = RMSPropModel().to(device)
    try:
        fixed_adam_model = torch.load('grad_fixed_adam_model.pth')
    except:
        fixed_adam_model = AdamModel().to(device)
    try:
        accelerated_model = torch.load('grad_accelerated_model.pth')
    except:
        accelerated_model = FixedModel().to(device)
    try:
        LBFGS_model = torch.load('grad_LBFGS_model.pth')
    except:
        LBFGS_model = FixedModel().to(device)
else:
    tau_model = TauFuncNet().to(device)
    hb_func_model = TauBetaFuncNet().to(device)
    tau_model_10 = TauFunc10Net().to(device)
    tau_model_last = TauFuncNet().to(device)
    tau_model_ubd_above = TauFuncUnboundedAboveNet().to(device)
    tau_model_ubd = TauFuncUnboundedNet().to(device)
    update_model = CNN_LSTM().to(device)#UpdateModel().to(device)
    free_model = CNN_LSTM().to(device)#GeneralUpdateModel().to(device)
    unrolling_model = UnrollingFunc().to(device)
    fixed_model = FixedModel().to(device)
    lastModel = FixedModel().to(device)
    decreasingModel = FixedModel().to(device)
    increasingModel = FixedModel().to(device)
    fixed_momentum_model = FixedMomentumModel().to(device)
    fixed_nesterov_model = FixedMomentumModel().to(device)
    fixed_heavy_model = FixedMomentumModel().to(device)
    fixed_adagrad_model = AdagradModel().to(device)
    fixed_rmsprop_model = RMSPropModel().to(device)
    fixed_adam_model = AdamModel().to(device)
    accelerated_model = FixedModel().to(device)
    LBFGS_model = TauFuncNet().to(device)

optimizer = torch.optim.Adam(tau_model.parameters())
optimizer_hb_func = torch.optim.Adam(hb_func_model.parameters())
optimizer_10 = torch.optim.Adam(tau_model_10.parameters())
optimizer_last = torch.optim.Adam(tau_model_last.parameters())
optimizer_ubd_above = torch.optim.Adam(tau_model_ubd_above.parameters())
optimizer_ubd = torch.optim.Adam(tau_model_ubd.parameters())
optimizer_update = torch.optim.Adam(update_model.parameters())
optimizer_free = torch.optim.Adam(free_model.parameters())
optimizer_fixed = torch.optim.Adam(fixed_model.parameters())
optimizer_unrolling = torch.optim.Adam(unrolling_model.parameters())
optimizer_fixed_momentum = torch.optim.Adam(fixed_momentum_model.parameters())
optimizer_fixed_nesterov = torch.optim.Adam(fixed_nesterov_model.parameters())
optimizer_fixed_heavy = torch.optim.Adam(fixed_heavy_model.parameters())
optimizer_fixed_adagrad = torch.optim.Adam(fixed_adagrad_model.parameters())
optimizer_fixed_rmsprop = torch.optim.Adam(fixed_rmsprop_model.parameters())
optimizer_fixed_adam = torch.optim.Adam(fixed_adam_model.parameters())
optimizer_accelerated = torch.optim.Adam(accelerated_model.parameters())
optimizer_LBFGS = torch.optim.Adam(LBFGS_model.parameters())
optimizer_lastModel = torch.optim.Adam(lastModel.parameters())
optimizer_decreasingModel = torch.optim.Adam(decreasingModel.parameters())
optimizer_increasingModel = torch.optim.Adam(increasingModel.parameters())


if not load_models:
    print('START TRAINING')
    for epoch in range(NUM_EPOCHS):  # Number of epochs
        torch.cuda.empty_cache()
        if epoch+1<num_iters:
            n_iters=epoch+1
        else:
            n_iters=num_iters
        epoch_obj = 0
        epoch_obj_hb_func = 0
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
        epoch_obj_lastModel = 0
        epoch_obj_decreasingModel = 0
        epoch_obj_increasingModel = 0
        for i, batch in enumerate(train_loader):
            total_objective = 0
            total_objective_hb_func = 0
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
            total_objective_lastModel = 0
            total_objective_decreasingModel = 0
            total_objective_increasingModel = 0
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

                # op_tau = 1 / (power_iteration(A_func_j)**2 + 8*alpha/0.01)
                # tau_learned = fixed_model(torch.tensor([std_j, sig_j]).to(device))
                # start_time = time.time()
                # if (tau_learned < 2*op_tau) & (tau_learned  > op_tau):
                #     xs, taus = gradient_descent_fixed(f_j, x0_j.to(device), y_j.to(device), tau_learned, tol=1e-03)
                #     fxs = [f_j(x, y_j.to(device)) for x in xs]
                #     min_fx = min(fxs)
                #     fxs_diff = [(fx - min_fx).cpu().numpy() for fx in fxs]
                #     x_star = xs[-1]
                #     f_x_star = f_j(x_star, y_j.to(device))
                # else:
                #     xs, taus = gradient_descent_fixed(f_j, x0_j.to(device), y_j.to(device), op_tau, tol=1e-03)
                #     x_star = xs[-1]
                #     f_x_star = f_j(x_star, y_j.to(device))
                #     fxs = [f_j(x, y_j.to(device)) for x in xs]
                # end_time = time.time()
                # #print(end_time - start_time)
                # #print(f_x_star)

                # ## Fixed and learned as a function
                # tau_learned = fixed_model(torch.tensor([sig_j, std_j]).to(device))
                # xs, taus = gradient_descent_fixed(f_j, x0_j, y_j, tau_learned, max_iter=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, [1])
                # total_objective_fixed += obj

                # ## Fixed and learned as a function, lastmodel
                # tau_learned_lastmodel = lastModel(torch.tensor([sig_j, std_j]).to(device))
                # xs, taus = gradient_descent_fixed(f_j, x0_j, y_j, tau_learned_lastmodel, max_iter=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, [-1])
                # total_objective_lastModel += obj
                #
                # ## Fixed and learned as a function, decreasingModel
                # tau_learned_decreasing = decreasingModel(torch.tensor([sig_j, std_j]).to(device))
                # xs, taus = gradient_descent_fixed(f_j, x0_j, y_j, tau_learned_decreasing, max_iter=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_dec(n_iters))
                # total_objective_decreasingModel += obj
                #
                # ## Fixed and learned as a function, increasingModel
                # tau_learned_increasing = increasingModel(torch.tensor([sig_j, std_j]).to(device))
                # xs, taus = gradient_descent_fixed(f_j, x0_j, y_j, tau_learned_increasing, max_iter=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j,  wk_increasing(n_iters))
                # total_objective_increasingModel += obj

                # ## learned function
                # xs, taus = gradient_descent_function(f_j, x0_j, y_j, tau_model, sig_j, std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective += obj

                # ## learned function 10
                # xs, taus = gradient_descent_function(grad_f_j, x0_j, y_j, tau_model_10, sig_j, std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_10 += obj
                #
                # ## learned function last
                # xs, taus = gradient_descent_function(f_j, x0_j, y_j, tau_model_last, sig_j, std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, [-1])
                # total_objective_last += obj

                # ## learned function unbounded above
                # xs, taus = gradient_descent_function(f_j, x0_j, y_j, tau_model_ubd_above, sig_j, std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_ubd_above += obj

                # ## learned function unbounded
                # xs, taus = gradient_descent_function(grad_f_j, x0_j, y_j, tau_model_ubd, sig_j, std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_ubd += obj

                # update
                # xs = gradient_descent_update(grad_f_j, x0_j, y_j, update_model, float(sig_j), std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_update += obj

                # free model
                #xs = gradient_descent_modelfree(grad_f_j, x0_j, y_j, free_model, float(sig_j), std_j, iters=n_iters)
                #obj, fs = function_evals(f_j, xs, y_j, wk_list)
                #total_objective_free += obj

                # # unrolling
                # unrolling_taus = unrolling_model(sig_j, std_j).to(device)  ## not actually a func of x0
                # xs, taus = gradient_descent_unrolling(f_j, x0_j, y_j, unrolling_taus, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_unrolling += obj

                # # fixed momentum
                # mom_tau, mom_beta = fixed_momentum_model(torch.tensor([sig_j, std_j]).to(device))
                # xs, taus = gradient_descent_fixed_momentum(f_j, x0_j, y_j, mom_tau, mom_beta, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_fixed_momentum += obj
                #
                # # fixed nesterov
                # nes_tau, nes_beta = fixed_nesterov_model(torch.tensor([sig_j, std_j]).to(device))
                # xs, taus = gradient_descent_fixed_nesterov(f_j, x0_j, y_j, nes_tau, nes_beta, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_fixed_nesterov += obj

                # heavy ball
                hb_tau, hb_beta = fixed_heavy_model(torch.tensor([sig_j, std_j]).to(device))
                xs, taus = gradient_descent_heavy_ball(f_j, x0_j, y_j, hb_tau, hb_beta, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_fixed_heavy += obj

                # heavy ball function
                xs, taus = gradient_descent_function(f_j, x0_j, y_j, hb_func_model, sig_j, std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_fixed_heavy += obj
                #
                # # adagrad
                # adagrad_tau, adagrad_epsilon = fixed_adagrad_model(torch.tensor([sig_j, std_j]).to(device))
                # adagrad_epsilon = torch.exp(-1/adagrad_epsilon)
                # xs, taus = adagrad(f_j, x0_j, y_j, adagrad_tau, adagrad_epsilon, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_fixed_adagrad += obj
                #
                # # rmsprop
                # rmsprop_tau, rmsprop_epsilon, rmsprop_beta = fixed_rmsprop_model(torch.tensor([sig_j, std_j]).to(device))
                # rmsprop_epsilon = torch.exp(-1/rmsprop_epsilon)
                # xs, taus = rmsprop(f_j, x0_j, y_j, rmsprop_tau, rmsprop_epsilon, rmsprop_beta, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_fixed_rmsprop += obj
                #
                # # adam
                # adam_tau, adam_epsilon, adam_beta1, adam_beta2 = fixed_adam_model(torch.tensor([sig_j, std_j]).to(device))
                # adam_epsilon = torch.exp(-1/adam_epsilon)
                # xs, taus = adam(f_j, x0_j, y_j, adam_tau, adam_epsilon, adam_beta1, adam_beta2, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_fixed_adam += obj
                #
                # # accelerated
                # acc_tau = accelerated_model(torch.tensor([sig_j, std_j]).to(device))
                # xs, taus = accelerated_gradient_descent(f_j, x0_j, y_j, acc_tau, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_accelerated += obj

                # adamax
                #adamax_tau, adamax_epsilon, adamax_beta1, adamax_beta2 = fixed_adamax_model(torch.tensor([std_j, sig_j]).to(device))
                #adamax_epsilon = torch.exp(-adamax_epsilon)
                #xs, taus = adamax(grad_f_j, x0_j, y_j, adamax_tau, adamax_epsilon, adamax_beta1, adamax_beta2, iters=n_iters)
                #total_objective_fixed_adamax += obj

                # LBFGS
                # xs, taus = LBFGS(grad_f_j, x0_j, y_j, LBFGS_model, sig_j, std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_LBFGS += obj

                ## learned function unbounded above
                xs, taus = gradient_descent_function(f_j, x0_j, y_j, tau_model_ubd_above, sig_j, std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_ubd_above += obj
            #print(taus[-1])



            total_objective /= (num_batch)
            total_objective_hb_func /= (num_batch)
            total_objective_10 /= (num_batch)
            total_objective_ubd_above /= (num_batch)
            total_objective_ubd /= (num_batch)
            total_objective_update /= (num_batch)
            total_objective_free /= (num_batch)
            total_objective_fixed /= (num_batch)
            total_objective_unrolling /= (num_batch)
            total_objective_last /= (num_batch)
            total_objective_fixed_momentum /= (num_batch)
            total_objective_fixed_nesterov /= (num_batch)
            total_objective_fixed_heavy /= (num_batch)
            total_objective_fixed_adagrad /= (num_batch)
            total_objective_fixed_rmsprop /= (num_batch)
            total_objective_fixed_adam /= (num_batch)
            total_objective_accelerated /= (num_batch)
            total_objective_LBFGS /= (num_batch)
            total_objective_lastModel /= (num_batch)
            total_objective_decreasingModel /= (num_batch)
            total_objective_increasingModel /= (num_batch)

            # total_objective.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # # total_objective_10.backward()
            # # optimizer_10.step()
            # # optimizer_10.zero_grad()
            # #
            # total_objective_last.backward(retain_graph=True)
            # optimizer_last.step()
            # optimizer_last.zero_grad()
            #
            total_objective_ubd_above.backward()
            optimizer_ubd_above.step()
            optimizer_ubd_above.zero_grad()

            total_objective_hb_func.backward()
            optimizer_hb_func.step()
            optimizer_hb_func.zero_grad()

            #
            # # total_objective_ubd.backward()
            # # optimizer_ubd.step()
            # # optimizer_ubd.zero_grad()
            # #
            # # total_objective_update.backward()
            # # optimizer_update.step()
            # # optimizer_update.zero_grad()
            # #
            # # total_objective_free.backward()
            # # optimizer.step()
            # # optimizer.zero_grad()
            #
            # total_objective_fixed.backward()
            # optimizer_fixed.step()
            # optimizer_fixed.zero_grad()
            #
            # total_objective_lastModel.backward()
            # optimizer_lastModel.step()
            # optimizer_lastModel.zero_grad()
            #
            # total_objective_decreasingModel.backward()
            # optimizer_decreasingModel.step()
            # optimizer_decreasingModel.zero_grad()
            #
            # total_objective_increasingModel.backward()
            # optimizer_increasingModel.step()
            # optimizer_increasingModel.zero_grad()

            #
            # # total_objective_unrolling.backward()
            # # optimizer_unrolling.step()
            # # optimizer_unrolling.zero_grad()
            # #
            # total_objective_fixed_momentum.backward()
            # optimizer_fixed_momentum.step()
            # optimizer_fixed_momentum.zero_grad()
            #
            # total_objective_fixed_nesterov.backward()
            # optimizer_fixed_nesterov.step()
            # optimizer_fixed_nesterov.zero_grad()

            total_objective_fixed_heavy.backward()
            optimizer_fixed_heavy.step()
            optimizer_fixed_heavy.zero_grad()

            # total_objective_fixed_adagrad.backward()
            # optimizer_fixed_adagrad.step()
            # optimizer_fixed_adagrad.zero_grad()
            #
            # total_objective_fixed_rmsprop.backward()
            # optimizer_fixed_rmsprop.step()
            # optimizer_fixed_rmsprop.zero_grad()
            #
            # total_objective_fixed_adam.backward()
            # optimizer_fixed_adam.step()
            # optimizer_fixed_adam.zero_grad()
            #
            # total_objective_accelerated.backward()
            # optimizer_accelerated.step()
            # optimizer_accelerated.zero_grad()



            # total_objective_LBFGS.backward()
            # optimizer_LBFGS.step()
            # optimizer_LBFGS.zero_grad()


            epoch_obj += total_objective
            epoch_obj_hb_func += total_objective_hb_func
            epoch_obj_10 += total_objective_10
            epoch_obj_last += total_objective_last
            epoch_obj_update += total_objective_update
            epoch_obj_free += total_objective_free
            epoch_obj_fixed += total_objective_fixed
            epoch_obj_unrolling += total_objective_unrolling
            epoch_obj_ubd_above += total_objective_ubd_above
            epoch_obj_ubd += total_objective_ubd
            epoch_obj_fixed_momentum += total_objective_fixed_momentum
            epoch_obj_fixed_nesterov += total_objective_fixed_nesterov
            epoch_obj_fixed_heavy += total_objective_fixed_heavy
            epoch_obj_fixed_adagrad += total_objective_fixed_adagrad
            epoch_obj_fixed_rmsprop += total_objective_fixed_rmsprop
            epoch_obj_fixed_adam += total_objective_fixed_adam
            epoch_obj_accelerated += total_objective_accelerated
            epoch_obj_LBFGS += total_objective_LBFGS
            epoch_obj_lastModel += total_objective_lastModel
            epoch_obj_decreasingModel += total_objective_decreasingModel
            epoch_obj_increasingModel += total_objective_increasingModel


        epoch_obj /= (NUM_IMAGES/num_batch)
        epoch_obj_hb_func /= (NUM_IMAGES/num_batch)
        epoch_obj_10 /= (NUM_IMAGES/num_batch)
        epoch_obj_last /= (NUM_IMAGES/num_batch)
        epoch_obj_update /= (NUM_IMAGES/num_batch)
        epoch_obj_free /= (NUM_IMAGES/num_batch)
        epoch_obj_fixed /= (NUM_IMAGES/num_batch)
        epoch_obj_unrolling /= (NUM_IMAGES/num_batch)
        epoch_obj_ubd_above /= (NUM_IMAGES/num_batch)
        epoch_obj_ubd /= (NUM_IMAGES/num_batch)
        epoch_obj_fixed_momentum /= (NUM_IMAGES/num_batch)
        epoch_obj_fixed_nesterov /= (NUM_IMAGES/num_batch)
        epoch_obj_fixed_heavy /= (NUM_IMAGES/num_batch)
        epoch_obj_fixed_adagrad /= (NUM_IMAGES/num_batch)
        epoch_obj_fixed_rmsprop /= (NUM_IMAGES/num_batch)
        epoch_obj_fixed_adam /= (NUM_IMAGES/num_batch)
        epoch_obj_accelerated /= (NUM_IMAGES/num_batch)
        epoch_obj_LBFGS /= (NUM_IMAGES/num_batch)
        epoch_obj_lastModel /= (NUM_IMAGES/num_batch)
        epoch_obj_decreasingModel /= (NUM_IMAGES/num_batch)
        epoch_obj_increasingModel /= (NUM_IMAGES/num_batch)

        #
        # epoch_obj.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        #
        # # epoch_obj_10.backward()
        # # optimizer_10.step()
        # # optimizer_10.zero_grad()
        # #
        # epoch_obj_last.backward()
        # optimizer_last.step()
        # optimizer_last.zero_grad()
        #
        # epoch_obj_ubd_above.backward()
        # optimizer_ubd_above.step()
        # optimizer_ubd_above.zero_grad()
        #
        # # epoch_obj_ubd.backward()
        # # optimizer_ubd.step()
        # # optimizer_ubd.zero_grad()
        # #
        # # epoch_obj_update.backward()
        # # optimizer_update.step()
        # # optimizer_update.zero_grad()
        # #
        # # epoch_obj_free.backward()
        # # optimizer.step()
        # # optimizer.zero_grad()
        #
        # epoch_obj_fixed.backward()
        # optimizer_fixed.step()
        # optimizer_fixed.zero_grad()
        #
        # epoch_obj_unrolling.backward()
        # optimizer_unrolling.step()
        # optimizer_unrolling.zero_grad()
        #
        # epoch_obj_fixed_momentum.backward()
        # optimizer_fixed_momentum.step()
        # optimizer_fixed_momentum.zero_grad()
        #
        # epoch_obj_fixed_nesterov.backward()
        # optimizer_fixed_nesterov.step()
        # optimizer_fixed_nesterov.zero_grad()
        #
        # epoch_obj_fixed_heavy.backward()
        # optimizer_fixed_heavy.step()
        # optimizer_fixed_heavy.zero_grad()

        print(f"Epoch: {epoch}, Function: {epoch_obj_ubd_above}, Fixed: {epoch_obj_fixed}, Fixed Momentum: {epoch_obj_fixed_momentum}, Fixed Nesterov: {epoch_obj_fixed_nesterov}, Fixed Heavy: {epoch_obj_fixed_heavy}, Accelerated: {epoch_obj_accelerated}, Adagrad: {epoch_obj_fixed_adagrad}, RMSProp: {epoch_obj_fixed_rmsprop}, Adam: {epoch_obj_fixed_adam}")
        #print(f"Epoch: {epoch}, Objective: {epoch_obj.item()}, 10: {epoch_obj_10.item()}, Last: {epoch_obj_last.item()}, Unbounded: {epoch_obj_ubd.item()}, Unbounded Above: {epoch_obj_ubd_above.item()}, Update: {epoch_obj_update}, Free: {epoch_obj_free}, Fixed: {epoch_obj_fixed.item()}, Unrolling: {epoch_obj_unrolling.item()}")
        print(hb_tau, hb_beta)
        # print(taus[-1])
        # print('Learned', tau_learned)
        # print('Learned LastModel', tau_learned_lastmodel)
        # print('Learned Decreasing', tau_learned_decreasing)
        # print('Learned Increasing', tau_learned_increasing)
        if epoch % 10 == 0:
            if epoch > 0:
                print("SAVING MODELS")
                new_input = 'small_var_noise_4sig'
                # torch.save(fixed_model, f'grad_fixed_model{new_input}.pth')
                torch.save(tau_model_ubd_above, f'grad_tau_model_ubd_above{new_input}.pth')
                # torch.save(fixed_momentum_model, f'grad_fixed_momentum_model{new_input}.pth')
                # torch.save(fixed_nesterov_model, f'grad_fixed_nesterov_model{new_input}.pth')
                torch.save(fixed_heavy_model, f'grad_fixed_heavy_model{new_input}.pth')
                torch.save(hb_func_model, f'grad_hb_func_model{new_input}.pth')
                # torch.save(fixed_adagrad_model, f'grad_fixed_adagrad_model{new_input}.pth')
                # torch.save(fixed_rmsprop_model, f'grad_fixed_rmsprop_model{new_input}.pth')
                # torch.save(fixed_adam_model, f'grad_fixed_adam_model{new_input}.pth')
                # torch.save(accelerated_model, f'grad_accelerated_model{new_input}.pth')
                print("FINISHED SAVING MODELS")

    print("SAVING MODELS")
    new_input = 'small_var_noise_4sig'
    torch.save(tau_model_ubd_above, f'grad_tau_model_ubd_above{new_input}.pth')
    torch.save(fixed_heavy_model, f'grad_fixed_heavy_model{new_input}.pth')
    torch.save(hb_func_model, f'grad_hb_func_model{new_input}.pth')
    # new_input = 'no_noise_var_sig'
    # torch.save(fixed_model, f'grad_fixed_model{new_input}.pth')
    # torch.save(tau_model, f'grad_tau_model{new_input}.pth')
    # torch.save(tau_model_10, f'grad_tau_model_10{new_input}.pth')
    # torch.save(tau_model_last, f'grad_tau_model_last{new_input}.pth')
    # torch.save(update_model, f'grad_update_model{new_input}.pth')
    # torch.save(free_model, f'grad_free_model{new_input}.pth')
    # torch.save(unrolling_model, f'grad_unrolling_func_model{new_input}.pth')
    # torch.save(fixed_momentum_model, f'grad_fixed_momentum_model{new_input}.pth')
    # torch.save(fixed_nesterov_model, f'grad_fixed_nesterov_model{new_input}.pth')
    # torch.save(fixed_heavy_model, f'grad_fixed_heavy_model{new_input}.pth')
    # torch.save(fixed_adagrad_model, f'grad_fixed_adagrad_model{new_input}.pth')
    # torch.save(fixed_rmsprop_model, f'grad_fixed_rmsprop_model{new_input}.pth')
    # torch.save(fixed_adam_model, f'grad_fixed_adam_model{new_input}.pth')
    # torch.save(accelerated_model, f'grad_accelerated_model{new_input}.pth')
    print("FINISHED SAVING MODELS")


tau_optimal = lambda a_norm, grad_norm: 1 / (alpha*grad_norm**2 + a_norm ** 2)
tau_optimal2 = lambda a_norm, grad_norm: 2 / (alpha*grad_norm**2 + a_norm ** 2)

test_iters = 25
## Fixed and not learned
fnl_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
fnl_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
fnl_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
fnl_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## Fixed and learned as a function
flf_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
flf_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
flf_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
flf_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## learned function
l_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
l_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
l_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
l_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## learned function with 10
# l10_noise_obj_dict = {str(round(float(i), 3)):[] for i in noise_list_test}
# l10_sigma_obj_dict = {str(round(float(i), 3)):[] for i in sigma_list_test}
# l10_noise_list_taus = {str(round(float(i), 3)):[] for i in noise_list_test}
# l10_sigma_list_taus = {str(round(float(i), 3)):[] for i in sigma_list_test}
## learned function with last
ll_noise_obj_dict = {str(round(float(i), 3)):[] for i in noise_list_test}
ll_sigma_obj_dict = {str(round(float(i), 3)):[] for i in sigma_list_test}
ll_noise_list_taus = {str(round(float(i), 3)):[] for i in noise_list_test}
ll_sigma_list_taus = {str(round(float(i), 3)):[] for i in sigma_list_test}
## Unbounded above
ubd_above_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
ubd_above_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
ubd_above_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
ubd_above_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## Unbounded
# ubd_noise_obj_dict = {str(round(float(i), 3)):[] for i in noise_list_test}
# ubd_sigma_obj_dict = {str(round(float(i), 3)):[] for i in sigma_list_test}
# ubd_noise_list_taus = {str(round(float(i), 3)):[] for i in noise_list_test}
# ubd_sigma_list_taus = {str(round(float(i), 3)):[] for i in sigma_list_test}
## Backtracking
bt_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
bt_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
bt_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
bt_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## learned update
# upd_noise_obj_dict = {str(round(float(i), 3)):[] for i in noise_list_test}
# upd_sigma_obj_dict = {str(round(float(i), 3)):[] for i in sigma_list_test}
# upd_noise_list_taus = {str(round(float(i), 3)):[] for i in noise_list_test}
# upd_sigma_list_taus = {str(round(float(i), 3)):[] for i in sigma_list_test}
# ## learned entire update
# mf_noise_obj_dict = {str(round(float(i), 3)):[] for i in noise_list_test}
# mf_sigma_obj_dict = {str(round(float(i), 3)):[] for i in sigma_list_test}
# mf_noise_list_taus = {str(round(float(i), 3)):[] for i in noise_list_test}
# mf_sigma_list_taus = {str(round(float(i), 3)):[] for i in sigma_list_test}
## Unrolling
unr_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
unr_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
unr_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
unr_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## Fixed momentum
fixed_mom_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_mom_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
fixed_mom_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_mom_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## Fixed Nesterov
fixed_nes_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_nes_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
fixed_nes_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_nes_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## Fixed heavy
hb_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
hb_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
hb_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
hb_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## Fixed adagrad
fixed_adagrad_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_adagrad_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
fixed_adagrad_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_adagrad_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## Fixed rmsprop
fixed_rmsprop_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_rmsprop_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
fixed_rmsprop_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_rmsprop_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## Fixed adam
fixed_adam_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_adam_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
fixed_adam_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
fixed_adam_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}
## Accelerated
acc_noise_obj_dict = {str(round(float(i), 3)): [] for i in noise_list_test}
acc_sigma_obj_dict = {str(round(float(i), 3)): [] for i in sigma_list_test}
acc_noise_list_taus = {str(round(float(i), 3)): [] for i in noise_list_test}
acc_sigma_list_taus = {str(round(float(i), 3)): [] for i in sigma_list_test}

fs_list = []
fs_list_10 = []
fs_list_last = []
fs_list_update = []
fs_list_free = []
fs_list_fixed = []
fs_list_nlfixed = []
fs_list_unrolling = []
fs_list_bt = []
fs_list_unbounded = []
fs_list_unbounded_above = []
fs_list_fixed_mom = []
fs_list_fixed_nes = []
fs_list_fixed_heavy = []
fs_list_fixed_adagrad = []
fs_list_fixed_rmsprop = []
fs_list_fixed_adam = []
fs_list_acc = []

time_nlfixed = 0
time_fixed = 0
time_learned = 0
time_unbounded_above = 0
time_unrolling = 0
time_backtracking = 0
time_mom = 0
time_nes = 0
time_heavy = 0
time_adagrad = 0
time_rmsprop = 0
time_adam = 0
time_acc = 0
time_last = 0

taus_fixed_learned = []
taus_nlfixed_learned = []
taus_fixed_mom = []
taus_fixed_nes = []
taus_fixed_heavy = []
taus_fixed_adagrad = []
taus_fixed_rmsprop = []
taus_fixed_adam = []
taus_acc = []
taus_unrolling = []

xs_list = []
xs_list_10 = []
xs_list_last = []
xs_list_update = []
xs_list_free = []
xs_list_fixed = []
xs_list_nlfixed = []
xs_list_unrolling = []
xs_list_bt = []
xs_list_unbounded = []
xs_list_unbounded_above = []
xs_list_fixed_mom = []
xs_list_fixed_nes = []
xs_list_fixed_heavy = []
xs_list_fixed_adagrad = []
xs_list_fixed_rmsprop = []
xs_list_fixed_adam = []
xs_list_acc = []
xs_list_last = []

with torch.no_grad():
    num = 0
    for lst in test_set:
        print(num)
        num += 1


        ### ESTIMATING TRUE X^*
        tau_learned = fixed_model(torch.tensor([lst.sig, lst.std]).to(device))
        xs, taus = gradient_descent_fixed(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_learned, tol=1e-06, f=lst.f)
        x_star = xs[-1]
        f_x_star = lst.f(x_star, lst.y.to(device))

        op_tau = 1/power_iteration(lambda x: lst.A_adj(lst.A_func(x))+alpha*laplacian(x))
        xs, taus = gradient_descent_fixed(lst.grad_f, lst.x0.to(device), lst.y.to(device), op_tau, tol=1e-06, f=lst.f)




        ### fixed not learned
        start_time = time.time()
        #op_ATA = estimate_operator_norm(lambda x: lst.A_adj(lst.A_func(x)))
        #op_lap = estimate_operator_norm(laplacian)
        op_total = power_iteration(lambda x: lst.A_adj(lst.A_func(x))+alpha*laplacian(x))
        #op_adj = estimate_operator_norm(lst.A_adj)
        opt_tau = 1/float(op_total)#tau_optimal(op_A, op_grad)
        xs, taus = gradient_descent_fixed(lst.grad_f, lst.x0.to(device), lst.y.to(device), opt_tau, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        fnl_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        fnl_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        fnl_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        fnl_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_nlfixed.append(fs)
        end_time = time.time()
        time_nlfixed += (end_time - start_time)
        taus_nlfixed_learned.append(opt_tau)
        print('Guessed', opt_tau)
        xs_list_nlfixed.append(xs)

        ## Fixed and learned as a function
        start_time = time.time()
        tau_learned = fixed_model(torch.tensor([lst.sig, lst.std]).to(device))
        xs, taus = gradient_descent_fixed(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_learned,
                                          iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        flf_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        flf_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        flf_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        flf_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_fixed.append(fs)
        end_time = time.time()
        time_fixed += (end_time - start_time)
        taus_fixed_learned.append(tau_learned)
        print('Learned', tau_learned)
        xs_list_fixed.append(xs)

        ## learned function
        start_time = time.time()
        xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model,
                                             lst.sig.to(device), lst.std.to(device), iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        l_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        l_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        l_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        l_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list.append(fs)
        end_time = time.time()
        time_learned += (end_time - start_time)
        xs_list.append(xs)

        # ## learned function 10
        # xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model_10, lst.sig.to(device), lst.std.to(device), iters=test_iters)
        # obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        # l10_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        # l10_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        # l10_noise_list_taus[str(round(float(lst.std),3))].append(taus)
        # l10_sigma_list_taus[str(round(float(lst.sig),3))].append(taus)
        # fs_list_10.append(fs)
        #
        ## last
        xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model_last, lst.sig.to(device), lst.std.to(device), iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        ll_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        ll_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        ll_noise_list_taus[str(round(float(lst.std),3))].append(taus)
        ll_sigma_list_taus[str(round(float(lst.sig),3))].append(taus)
        fs_list_last.append(fs)
        #
        # ## unbounded
        # xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model_ubd, lst.sig.to(device), lst.std.to(device), iters=test_iters)
        # obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        # ubd_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        # ubd_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        # ubd_noise_list_taus[str(round(float(lst.std),3))].append(taus)
        # ubd_sigma_list_taus[str(round(float(lst.sig),3))].append(taus)
        # fs_list_unbounded.append(fs)

        ## unbounded above
        start_time = time.time()
        xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model_ubd_above,
                                             lst.sig.to(device), lst.std.to(device), iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        ubd_above_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        ubd_above_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        ubd_above_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        ubd_above_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_unbounded_above.append(fs)
        end_time = time.time()
        time_unbounded_above += (end_time - start_time)
        xs_list_unbounded_above.append(xs)

        # ## learned update
        # xs = gradient_descent_update(lst.grad_f, lst.x0.to(device), lst.y.to(device), update_model, lst.sig, lst.std, iters=test_iters)
        # obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        # upd_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        # upd_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        # fs_list_update.append(fs)
        #
        # ## learned entire update
        # xs = gradient_descent_modelfree(lst.grad_f, lst.x0.to(device), lst.y.to(device), free_model, lst.sig, float(lst.std), iters=test_iters)
        # obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        # mf_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        # mf_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        # fs_list_free.append(fs)

        ## Unrolling
        start_time = time.time()
        unrolling_taus = [round(float(i), 3) for i in
                          unrolling_model(lst.sig.to(device), lst.std.to(device))]  ## not actually a func of x0
        xs, taus = gradient_descent_unrolling(lst.grad_f, lst.x0.to(device), lst.y.to(device), unrolling_taus,
                                              iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        unr_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        unr_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        unr_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        unr_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_unrolling.append(fs)
        end_time = time.time()
        time_unrolling += (end_time - start_time)
        xs_list_unrolling.append(xs)

        ## Backtracking
        start_time = time.time()
        xs, taus = gradient_descent_backtracking(lst.grad_f, lst.x0, lst.y, lst.f, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y, wk_list)
        bt_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        bt_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        bt_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        bt_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_bt.append(fs)
        end_time = time.time()
        time_backtracking += (end_time - start_time)
        xs_list_bt.append(xs)

        ## Fixed momentum
        start_time = time.time()
        varis = fixed_momentum_model(torch.tensor([lst.sig, lst.std]).to(device))
        tau_mom = varis[0]
        beta_mom = varis[1]
        xs, taus = gradient_descent_fixed_momentum(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_mom, beta_mom, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        fixed_mom_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        fixed_mom_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        fixed_mom_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        fixed_mom_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_fixed_mom.append(fs)
        end_time = time.time()
        time_mom += (end_time - start_time)
        xs_list_fixed_mom.append(xs)
        taus_fixed_mom.append(tau_mom)

        ## Fixed Nesterov
        start_time = time.time()
        varis = fixed_nesterov_model(torch.tensor([lst.sig, lst.std]).to(device))
        tau_nes = varis[0]
        beta_nes = varis[1]
        xs, taus = gradient_descent_fixed_nesterov(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_nes, beta_nes, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        fixed_nes_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        fixed_nes_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        fixed_nes_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        fixed_nes_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_fixed_nes.append(fs)
        end_time = time.time()
        time_nes += (end_time - start_time)
        xs_list_fixed_nes.append(xs)

        ## Fixed Heavy
        start_time = time.time()
        varis = fixed_heavy_model(torch.tensor([lst.sig, lst.std]).to(device))
        tau_hb = varis[0]
        beta_hb = varis[1]
        xs, taus = gradient_descent_heavy_ball(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_hb, beta_hb, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        hb_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        hb_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        hb_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        hb_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_fixed_heavy.append(fs)
        end_time = time.time()
        time_heavy += (end_time - start_time)
        xs_list_fixed_heavy.append(xs)

        ## Fixed adagrad
        start_time = time.time()
        varis = fixed_adagrad_model(torch.tensor([lst.sig, lst.std]).to(device))
        tau_adagrad = varis[0]
        epsilon_adagrad = varis[1]
        #epsilon_adagrad = torch.exp(-epsilon_adagrad)
        xs, taus = adagrad(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_adagrad, epsilon_adagrad, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        fixed_adagrad_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        fixed_adagrad_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        fixed_adagrad_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        fixed_adagrad_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_fixed_adagrad.append(fs)
        end_time = time.time()
        time_adagrad += (end_time - start_time)
        xs_list_fixed_adagrad.append(xs)

        ## Fixed RMSProp
        start_time = time.time()
        varis = fixed_rmsprop_model(torch.tensor([lst.sig, lst.std]).to(device))
        tau_rmsprop = varis[0]
        beta_rmsprop = varis[1]
        epsilon_rmsprop = varis[2]
        #epsilon_rmsprop = torch.exp(-epsilon_rmsprop)
        xs, taus = rmsprop(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_rmsprop, beta_rmsprop, epsilon_rmsprop, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        fixed_rmsprop_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        fixed_rmsprop_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        fixed_rmsprop_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        fixed_rmsprop_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_fixed_rmsprop.append(fs)
        end_time = time.time()
        time_rmsprop += (end_time - start_time)
        xs_list_fixed_rmsprop.append(xs)

        ## Fixed Adam
        start_time = time.time()
        varis = fixed_adam_model(torch.tensor([lst.sig, lst.std]).to(device))
        tau_adam = varis[0]
        beta1_adam = varis[1]
        beta2_adam = varis[2]
        epsilon_adam = varis[3]
        #epsilon_adam = torch.exp(-epsilon_adam)
        xs, taus = adam(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_adam, beta1_adam, beta2_adam, epsilon_adam, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        fixed_adam_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        fixed_adam_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        fixed_adam_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        fixed_adam_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_fixed_adam.append(fs)
        end_time = time.time()
        time_adam += (end_time - start_time)
        xs_list_fixed_adam.append(xs)

        ## Accelerated
        start_time = time.time()
        tau_acc = accelerated_model(torch.tensor([lst.sig, lst.std]).to(device))
        xs, taus = accelerated_gradient_descent(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_acc, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list, f_x_star)
        acc_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        acc_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        acc_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        acc_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_acc.append(fs)
        end_time = time.time()
        time_acc += (end_time - start_time)
        xs_list_acc.append(xs)







# Calculate average times
avg_time_nlfixed = time_nlfixed / num
avg_time_fixed = time_fixed / num
avg_time_learned = time_learned / num
avg_time_unbounded_above = time_unbounded_above / num
avg_time_unrolling = time_unrolling / num
avg_time_backtracking = time_backtracking / num
avg_time_mom = time_mom / num
avg_time_nes = time_nes / num
avg_time_heavy = time_heavy / num
avg_time_adagrad = time_adagrad / num
avg_time_rmsprop = time_rmsprop / num
avg_time_adam = time_adam / num
avg_time_acc = time_acc / num
avg_time_last = time_last / num

print("Average Time for fixed not learned:", avg_time_nlfixed)
print("Average Time for Fixed and learned as a function:", avg_time_fixed)
print("Average Time for learned function:", avg_time_learned)
print("Average Time for unbounded above:", avg_time_unbounded_above)
print("Average Time for Unrolling:", avg_time_unrolling)
print("Average Time for Backtracking:", avg_time_backtracking)
print("Average Time for Momentum:", avg_time_mom)
print("Average Time for Nesterov:", avg_time_nes)
print("Average Time for Heavy Ball:", avg_time_heavy)
print("Average Time for Adagrad:", avg_time_adagrad)
print("Average Time for RMSProp:", avg_time_rmsprop)
print("Average Time for Adam:", avg_time_adam)
print("Average Time for Accelerated:", avg_time_acc)
print("Average Time for Last Iteration:", avg_time_last)


fs_list_agg = [[float(fs_list[i][k]) for i in range(len(fs_list))] for k in range(len(fs_list[0]))]
#fs_list_10_agg = [[float(fs_list_10[i][k]) for i in range(len(fs_list_10))] for k in range(len(fs_list_10[0]))]
fs_list_last_agg = [[float(fs_list_last[i][k]) for i in range(len(fs_list_last))] for k in range(len(fs_list_last[0]))]
fs_list_nlfixed_agg = [[float(fs_list_nlfixed[i][k]) for i in range(len(fs_list_nlfixed))] for k in
                       range(len(fs_list_nlfixed[0]))]
fs_list_fixed_agg = [[float(fs_list_fixed[i][k]) for i in range(len(fs_list_fixed))] for k in
                     range(len(fs_list_fixed[0]))]
#fs_list_update_agg = [[float(fs_list_update[i][k]) for i in range(len(fs_list_update))] for k in range(len(fs_list_update[0]))]
#fs_list_free_agg = [[float(fs_list_free[i][k]) for i in range(len(fs_list_free))] for k in range(len(fs_list_free[0]))]
fs_list_unrolling_agg = [[float(fs_list_unrolling[i][k]) for i in range(len(fs_list_unrolling))] for k in
                         range(len(fs_list_unrolling[0]))]
fs_list_bt_agg = [[float(fs_list_bt[i][k]) for i in range(len(fs_list_bt))] for k in range(len(fs_list_bt[0]))]
#fs_list_ubd_agg = [[float(fs_list_unbounded[i][k]) for i in range(len(fs_list_unbounded))] for k inrange(len(fs_list_unbounded[0]))]
fs_list_ubd_above_agg = [[float(fs_list_unbounded_above[i][k]) for i in range(len(fs_list_unbounded_above))] for k in
                         range(len(fs_list_unbounded_above[0]))]
fs_list_mom_agg = [[float(fs_list_fixed_mom[i][k]) for i in range(len(fs_list_fixed_mom))] for k in range(len(fs_list_fixed_mom[0]))]
fs_list_nes_agg = [[float(fs_list_fixed_nes[i][k]) for i in range(len(fs_list_fixed_nes))] for k in range(len(fs_list_fixed_nes[0]))]
fs_list_heavy_agg = [[float(fs_list_fixed_heavy[i][k]) for i in range(len(fs_list_fixed_heavy))] for k in range(len(fs_list_fixed_heavy[0]))]
fs_list_adagrad_agg = [[float(fs_list_fixed_adagrad[i][k]) for i in range(len(fs_list_fixed_adagrad))] for k in range(len(fs_list_fixed_adagrad[0]))]
fs_list_rmsprop_agg = [[float(fs_list_fixed_rmsprop[i][k]) for i in range(len(fs_list_fixed_rmsprop))] for k in range(len(fs_list_fixed_rmsprop[0]))]
fs_list_adam_agg = [[float(fs_list_fixed_adam[i][k]) for i in range(len(fs_list_fixed_adam))] for k in range(len(fs_list_fixed_adam[0]))]
fs_list_acc_agg = [[float(fs_list_acc[i][k]) for i in range(len(fs_list_acc))] for k in range(len(fs_list_acc[0]))]


f_avg = [np.mean(i) for i in fs_list_agg]
#f_avg_10 = [np.mean(i) for i in fs_list_10_agg]
f_avg_last = [np.mean(i) for i in fs_list_last_agg]
f_avg_nlfixed = [np.mean(i) for i in fs_list_nlfixed_agg]
f_avg_fixed = [np.mean(i) for i in fs_list_fixed_agg]
#f_avg_update = [np.mean(i) for i in fs_list_update_agg]
#f_avg_free = [np.mean(i) for i in fs_list_free_agg]
f_avg_unrolling = [np.mean(i) for i in fs_list_unrolling_agg]
f_avg_bt = [np.mean(i) for i in fs_list_bt_agg]
#f_avg_ubd = [np.mean(i) for i in fs_list_ubd_agg]
f_avg_ubd_above = [np.mean(i) for i in fs_list_ubd_above_agg]
f_avg_mom = [np.mean(i) for i in fs_list_mom_agg]
f_avg_nes = [np.mean(i) for i in fs_list_nes_agg]
f_avg_heavy = [np.mean(i) for i in fs_list_heavy_agg]
f_avg_adagrad = [np.mean(i) for i in fs_list_adagrad_agg]
f_avg_rmsprop = [np.mean(i) for i in fs_list_rmsprop_agg]
f_avg_adam = [np.mean(i) for i in fs_list_adam_agg]
f_avg_acc = [np.mean(i) for i in fs_list_acc_agg]




f_max = [np.max(i) for i in fs_list_agg]
#f_max_10 = [np.max(i) for i in fs_list_10_agg]
f_max_last = [np.max(i) for i in fs_list_last_agg]
f_max_nlfixed = [np.max(i) for i in fs_list_nlfixed_agg]
f_max_fixed = [np.max(i) for i in fs_list_fixed_agg]
#f_max_update = [np.max(i) for i in fs_list_update_agg]
#f_max_free = [np.max(i) for i in fs_list_free_agg]
f_max_unrolling = [np.max(i) for i in fs_list_unrolling_agg]
f_max_bt = [np.max(i) for i in fs_list_bt_agg]
#f_max_ubd = [np.max(i) for i in fs_list_ubd_agg]
f_max_ubd_above = [np.max(i) for i in fs_list_ubd_above_agg]
f_max_mom = [np.max(i) for i in fs_list_mom_agg]
f_max_nes = [np.max(i) for i in fs_list_nes_agg]
f_max_heavy = [np.max(i) for i in fs_list_heavy_agg]
f_max_adagrad = [np.max(i) for i in fs_list_adagrad_agg]
f_max_rmsprop = [np.max(i) for i in fs_list_rmsprop_agg]
f_max_adam = [np.max(i) for i in fs_list_adam_agg]
f_max_acc = [np.max(i) for i in fs_list_acc_agg]

f_min = [np.min(i) for i in fs_list_agg]
#f_min_10 = [np.min(i) for i in fs_list_10_agg]
f_min_last = [np.min(i) for i in fs_list_last_agg]
f_min_nlfixed = [np.min(i) for i in fs_list_nlfixed_agg]
f_min_fixed = [np.min(i) for i in fs_list_fixed_agg]
#f_min_update = [np.min(i) for i in fs_list_update_agg]
#f_min_free = [np.min(i) for i in fs_list_free_agg]
f_min_unrolling = [np.min(i) for i in fs_list_unrolling_agg]
f_min_bt = [np.min(i) for i in fs_list_bt_agg]
#f_min_ubd = [np.min(i) for i in fs_list_ubd_agg]
f_min_ubd_above = [np.min(i) for i in fs_list_ubd_above_agg]
f_min_mom = [np.min(i) for i in fs_list_mom_agg]
f_min_nes = [np.min(i) for i in fs_list_nes_agg]
f_min_heavy = [np.min(i) for i in fs_list_heavy_agg]
f_min_adagrad = [np.min(i) for i in fs_list_adagrad_agg]
f_min_rmsprop = [np.min(i) for i in fs_list_rmsprop_agg]
f_min_adam = [np.min(i) for i in fs_list_adam_agg]
f_min_acc = [np.min(i) for i in fs_list_acc_agg]


if False:
    plt.plot(f_avg, label='Learned Tau Function')
    #plt.plot(f_avg_10, label='10')
    #plt.plot(f_avg_last, label='Last')
    plt.plot(f_avg_nlfixed, label='Fixed')
    plt.plot(f_avg_fixed, label='Fixed Learned')
    #plt.plot(f_avg_update, label='Learned Update')
    #plt.plot(f_avg_free, label='Learned General')
    plt.plot(f_avg_unrolling, label='Unrolling')
    plt.plot(f_avg_bt, label='Backtracking')
    #plt.plot(f_avg_ubd, label='Unbounded')
    plt.plot(f_avg_ubd_above, label='Unbounded Above')
    plt.plot(f_avg_mom, label='Momentum')
    plt.plot(f_avg_nes, label='Nesterov')
    plt.plot(f_avg_heavy, label='Heavy Ball')
    plt.plot(f_avg_adagrad, label='Adagrad')
    plt.plot(f_avg_rmsprop, label='RMSProp')
    plt.plot(f_avg_adam, label='Adam')
    plt.legend()
    plt.show()


    plt.plot(f_avg, label='Learned Tau Function')
    plt.plot(f_avg_nlfixed, label='Fixed')
    plt.plot(f_avg_fixed, label='Fixed Learned')
    plt.plot(f_avg_bt, label='Backtracking')
    plt.plot(f_avg_ubd_above, label='Unbounded Above')
    plt.legend()
    plt.show()

    plt.plot(f_max, label='Learned Tau Function')
    #plt.plot(f_max_10, label='10')
    #plt.plot(f_max_last, label='Last')
    plt.plot(f_max_nlfixed, label='Fixed')
    plt.plot(f_max_fixed, label='Fixed Learned')
    #plt.plot(f_max_update, label='Learned Update')
    #plt.plot(f_max_free, label='Learned General')
    plt.plot(f_max_unrolling, label='Unrolling')
    plt.plot(f_max_bt, label='Backtracking')
    #plt.plot(f_max_ubd, label='Unbounded')
    plt.plot(f_max_ubd_above, label='Unbounded Above')
    plt.legend()
    plt.show()

    # get_boxplot(bt_noise_obj_dict, 'Noise Level')

    # get_boxplot(bt_sigma_obj_dict, 'Blurring')


    plt.plot(f_max, label='Learned Tau Function')
    #plt.plot(f_max_last, label='Last')
    plt.plot(f_max_fixed, label='Fixed Learned')
    plt.plot(f_max_bt, label='Backtracking')
    plt.legend()
    plt.show()





    # Plotting the boxplots
    plt.boxplot(fs_list_agg[:10])
    plt.xticks([1, 2, 3, 4], ['lst1', 'lst2', 'lst3', 'lst4'])
    plt.title("Boxplots of lst1, lst2, lst3, lst4")
    plt.show()


    plt.boxplot([[fs_list_bt_agg[i][j] - fs_list_agg[i][j] for j in range(len(fs_list_agg[i]))] for i in range(len(fs_list_agg))][:10])












    import pandas as pd
    import seaborn as sns
    # Create a new dataframe with the updated lists and index
    df_updated = pd.DataFrame({
        'Iteration': list(range(len(f_avg)))[:10],
        'Learned Fixed Tau': f_avg_fixed[:10],
        'Non learned Fixed Tau': f_avg_nlfixed[:10],
        'Backtracking': f_avg_bt[:10]
    })
    # Melt the dataframe to a long format, which is more suitable for seaborn
    df_melted = df_updated.melt(id_vars='Iteration', var_name='method', value_name='value')
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='Iteration', y='value', hue='method')
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Mean over f of f(x_k) over test set')
    plt.title('Comparison of Learned Tau Function and Backtracking')
    plt.show()

    df_updated = pd.DataFrame({
        'Iteration': list(range(len(f_avg)))[:10],
        'Learned Fixed Tau': f_max_fixed[:10],
        'Non learned Fixed Tau': f_max_nlfixed[:10],
        'Backtracking': f_max_bt[:10]
    })
    # Melt the dataframe to a long format, which is more suitable for seaborn
    df_melted = df_updated.melt(id_vars='Iteration', var_name='method', value_name='value')
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='Iteration', y='value', hue='method')
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Max over f of f(x_k) over test set')
    plt.title('Comparison of Learned Tau Function and Backtracking')
    plt.show()


    df_updated = pd.DataFrame({
        'Iteration': list(range(len(f_avg)))[:10],
        'Learned Fixed Tau': f_avg_fixed[:10],
        'Non learned Fixed Tau': f_avg_nlfixed[:10],
        'Backtracking': f_avg_bt[:10],
        'Tau Function': f_avg[:10],
        'Unrolling': f_avg_unrolling[:10],
        'Unbounded Above': f_avg_ubd_above[:10],
        'Momentum': f_avg_mom[:10],
        'Nesterov': f_avg_nes[:10],
        'Heavy Ball': f_avg_heavy[:10],
        'Adagrad': f_avg_adagrad[:10],
        #'RMSProp': f_avg_rmsprop[:10],
        #'Adam': f_avg_adam[:10],
        'Acceleration': f_avg_acc[:10],
        'Last': f_avg_last[:10]
    })
    # Melt the dataframe to a long format, which is more suitable for seaborn
    df_melted = df_updated.melt(id_vars='Iteration', var_name='method', value_name='value')
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='Iteration', y='value', hue='method')
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Max over f of f(x_k) over test set')
    plt.title('Comparison of Learned Tau Function and Backtracking')
    plt.show()


    df_updated = pd.DataFrame({
        'Iteration': list(range(len(f_avg)))[:10],
        'Learned Fixed Tau': f_min_fixed[:10],
        'Non learned Fixed Tau': f_min_nlfixed[:10],
        'Backtracking': f_min_bt[:10]
    })
    # Melt the dataframe to a long format, which is more suitable for seaborn
    df_melted = df_updated.melt(id_vars='Iteration', var_name='method', value_name='value')
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='Iteration', y='value', hue='method')
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Min over f of f(x_k) over test set')
    plt.title('Comparison of Learned Tau Function and Backtracking')
    plt.show()


    def plot_3(s1, s2, s3, label1 ='Learned Fixed Tau', label2='Non learned Fixed Tau', label3='Backtracking'):
        df_updated = pd.DataFrame({
            'Iteration': list(range(len(s1))),
            label1: s1,
            label2: s2,
            label3: s3
        })
        # Melt the dataframe to a long format, which is more suitable for seaborn
        df_melted = df_updated.melt(id_vars='Iteration', var_name='method', value_name='value')
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_melted, x='Iteration', y='value', hue='method')
        # Add labels and title
        plt.xlabel('Iteration')
        plt.ylabel('f of f(x_k) over test set')
        plt.title('Comparison of Learned Tau Function and Backtracking')
        plt.show()












    fs_list_seq = [[float(i) for i in lst] for lst in fs_list]
    #fs_list_seq_10 = [[float(i) for i in lst] for lst in fs_list_10]
    #fs_list_seq_last = [[float(i) for i in lst] for lst in fs_list_last]
    fs_list_seq_nlfixed = [[float(i) for i in lst] for lst in fs_list_nlfixed]
    fs_list_seq_fixed = [[float(i) for i in lst] for lst in fs_list_fixed]
    #fs_list_seq_update = [[float(i) for i in lst] for lst in fs_list_update]
    #fs_list_seq_free = [[float(i) for i in lst] for lst in fs_list_free]
    fs_list_seq_unrolling = [[float(i) for i in lst] for lst in fs_list_unrolling]
    fs_list_seq_bt = [[float(i) for i in lst] for lst in fs_list_bt]
    #fs_list_seq_ubd = [[float(i) for i in lst] for lst in fs_list_unbounded]
    fs_list_seq_ubd_above = [[float(i) for i in lst] for lst in fs_list_unbounded_above]


    min_learned_it = [fs_list_seq_fixed[i].index(min(fs_list_seq_fixed[i])) for i in range(len(fs_list_seq_fixed))]
    ratios_nl_l_fixed = [np.mean([fs_list_seq_nlfixed[seq_it][i]/fs_list_seq_fixed[seq_it][i] for i in range(min_learned_it[seq_it])]) for seq_it in range(len(fs_list_seq))]
    max_ratio_it = ratios_nl_l_fixed.index(max(ratios_nl_l_fixed))
    min_ratio_it = ratios_nl_l_fixed.index(min(ratios_nl_l_fixed))
    max_ratio_seq_nlfixed = fs_list_seq_nlfixed[max_ratio_it][:min_learned_it[max_ratio_it]]
    max_ratio_seq_fixed = fs_list_seq_fixed[max_ratio_it][:min_learned_it[max_ratio_it]]
    max_ratio_seq_backtracking = fs_list_seq_bt[max_ratio_it][:min_learned_it[max_ratio_it]]
    min_ratio_seq_nlfixed = fs_list_seq_nlfixed[min_ratio_it][:min_learned_it[min_ratio_it]]
    min_ratio_seq_fixed = fs_list_seq_fixed[min_ratio_it][:min_learned_it[min_ratio_it]]
    min_ratio_seq_backtracking = fs_list_seq_bt[min_ratio_it][:min_learned_it[min_ratio_it]]
    plot_3(max_ratio_seq_fixed, max_ratio_seq_nlfixed, max_ratio_seq_backtracking)
    plot_3(min_ratio_seq_fixed, min_ratio_seq_nlfixed, min_ratio_seq_backtracking)


    ratios_nl_l_fixed = [np.mean([fs_list_seq_bt[seq_it][i]/fs_list_seq_fixed[seq_it][i] for i in range(min_learned_it[seq_it])]) for seq_it in range(len(fs_list_seq))]
    max_ratio_it = ratios_nl_l_fixed.index(max(ratios_nl_l_fixed))
    min_ratio_it = ratios_nl_l_fixed.index(min(ratios_nl_l_fixed))
    max_ratio_seq_nlfixed = fs_list_seq_nlfixed[max_ratio_it][:min_learned_it[max_ratio_it]]
    max_ratio_seq_fixed = fs_list_seq_fixed[max_ratio_it][:min_learned_it[max_ratio_it]]
    max_ratio_seq_backtracking = fs_list_seq_bt[max_ratio_it][:min_learned_it[max_ratio_it]]
    min_ratio_seq_nlfixed = fs_list_seq_nlfixed[min_ratio_it][:min_learned_it[min_ratio_it]]
    min_ratio_seq_fixed = fs_list_seq_fixed[min_ratio_it][:min_learned_it[min_ratio_it]]
    min_ratio_seq_backtracking = fs_list_seq_bt[min_ratio_it][:min_learned_it[min_ratio_it]]
    plot_3(max_ratio_seq_fixed, max_ratio_seq_nlfixed, max_ratio_seq_backtracking)
    plot_3(min_ratio_seq_fixed, min_ratio_seq_nlfixed, min_ratio_seq_backtracking)

    ## visualise outputs - see if I need to change the regulariser
    ## see if theres a difference between different noises and different blurs
    ## implement the second 'optimal' step size

    for x in xs_list_fixed[min_ratio_it]:
        plt.imshow(x)
        plt.show()

    plt.imshow(test_dataset[max_ratio_it])

    for x in xs_list_fixed[max_ratio_it]:
        plt.imshow(x.to(device='cpu').detach().numpy())
        plt.show()


    plt.imshow(xs_list_fixed[max_ratio_it][-1].to(device='cpu').detach().numpy())
    plt.imshow(xs_list_nlfixed[max_ratio_it][-1].to(device='cpu').detach().numpy())


    plt.plot([float(i) for i in taus_nlfixed_learned])
    plt.plot([float(i) for i in taus_fixed_learned])
    plt.show()

    plt.plot([float(taus_nlfixed_learned[i])/float(taus_fixed_learned[i]) for i in range(len(taus_nlfixed_learned))])
    plt.show()

    for x in xs_list_fixed[min_ratio_it]:
        print(psnr(test_dataset[min_ratio_it], x.to(device='cpu').detach().numpy()))

    for x in xs_list_fixed[max_ratio_it]:
        print(psnr(test_dataset[max_ratio_it], x.to(device='cpu').detach().numpy()))