from torch.utils.data import DataLoader
import numpy as np
import torch
from huber_TV import power_iteration
from datasets import NaturalDataset, TestDataset, ImageBlurDataset, my_collate
from optimisers import CNN_LSTM, CNN_LSTM_Full, CNN_LSTM_CORRECTION, AdagradModel, RMSPropModel, AdamModel, TauFuncNet
from algorithms import function_evals, gradient_descent_fixed,  gradient_descent_update, gradient_descent_modelfree, gradient_descent_correctionNN, gradient_descent_post_correction
from grad_x import grad, laplacian
import time
from torch.nn import HuberLoss

import matplotlib.pyplot as plt
# xs, taus = gradient_descent_fixed(f_j, x0_j.to(device), y_j.to(device), 0.5, tol=1e-06)
# x_star = xs[-1]
# f_x_star = torch.tensor(f_j(x_star.to(device), y_j.to(device))).float()
# fs = [f_j(x, y_j).detach().cpu().numpy() for x in xs]

# xs2, taus = gradient_descent_fixed(f_j, x0_j.to(device), y_j.to(device), 1, tol=1e-06)
# x_star2 = xs[-1]
# f_x_star2 = torch.tensor(f_j(x_star.to(device), y_j.to(device))).float()
# fs2 = [f_j(x, y_j).detach().cpu().numpy() for x in xs2]


# plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
# plt.loglog(fs, marker='o', linestyle='-', color='b', label='Data')
# plt.loglog(fs2, marker='o', linestyle='-', color='b', label='Data2')
# plt.xlabel('X Values (log scale)')
# plt.ylabel('Y Values (log scale)')
# plt.title('Log-Log Plot Example')
# plt.legend()
# plt.grid(True)
# plt.show()



from functions import get_blurred_image

def get_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_x_star(f, x0, y, tau_choice):
    xs, taus = gradient_descent_fixed(f, x0.to(device), y.to(device), tau_choice, tol=1e-03, max_iter=2000)
    x_star = xs[-1]
    return x_star

def get_recon_psnr(f, x0, y, img, tau_choice):
    x_star = get_x_star(f, x0, y, tau_choice)
    return get_psnr(x_star, img.to(device))

def get_f_from_alpha(alpha, A_func):
    return lambda x,y: data_fidelity(x,y,A_func) + alpha * huber_total_variation(x)

def get_recon_psnr_from_alpha(alpha, y,A_func, img, tau_choice):
    f = get_f_from_alpha(alpha, A_func)
    return get_recon_psnr(f, x0, y, img, tau_choice)

def get_recon_psnrs_from_alpha(alpha, images, x,y,A_func, tau_choice):
    f = get_f_from_alpha(alpha, x,y,A_func)
    return sum([get_recon_psnr(f, x0, y, img, tau_choice)] for img in images)



def huber(s, epsilon=0.01):
    return torch.where(torch.abs(s) <= epsilon, (0.5 * s ** 2) / epsilon, s - 0.5 * epsilon)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_models = False
alpha = 0.0001
NUM_IMAGES = 40
NUM_TEST = 10

noise_list = [0.0196, 0.0392, 0.0588]
sigma_list = [4]#list(np.linspace(2, 8, 6))
noise_list_test = [0.4]
sigma_list_test = list(np.linspace(3, 10, 7))

num_batch = 4
wk_list = [1]
wk_list2 = [-1]

n_iters=40
num_iters=40

def wk_dec(T):
    denom = (T/2)*(T+1)
    wk_list = [(T-i)/denom for i in range(T)]
    return wk_list

def wk_exp_dec(T, lambd = 0.99):
    a = (1-lambd)/(1-lambd**T)
    wk_list = [a*(lambd**i) for i in range(T)]
    return wk_list

def wk_increasing(T):
    denom = (T/2)*(T+1)
    wk_list = [i/denom for i in range(1,T+1)]
    return wk_list

NUM_EPOCHS = 1000

## sig = 0 -> alpha = 0.0005
## sig=0.1 -> alpha = 0.01
## sig=0.2 -> alpha = 1, but when the noise strength is lower alpha higher is better - = 2 best, 1
## sig=0.3 -> alpha = 2
## sig=0.4 -> alpha = 0.1

#
# for alpha in [0.0005,0.001, 0.005, 0.01]:
#     def f_get(x, y, A_func, alpha):
#         return data_fidelity(x, y, A_func) + alpha * huber_total_variation(x)
#     total = 0
#     total_star = 0
#     for j in range(len(img)):
#         new_f = lambda x, y: f_get(x, y, lambda im: get_blurred_image(im, sig[j]), alpha)
#         tau_choice = 1/(1+800*alpha)
#         x_star = get_x_star(new_f, x0[j], y[j], tau_choice)
#         total_star+= new_f(x_star, y[j].to(device))
#         total += get_psnr(x_star, img[j].to(device))
#     total/=len(img)
#     print('Alpha PSNR', alpha, total)
#     mean_fx0 = np.mean([new_f(x0[j], y[j]) for j in range(len(x0))])
#     total_star/=len(img)
#     print('Alpha ratio of f_x_star to f_x_0', alpha, total_star/mean_fx0)

# def f(x,y,A_func, alpha):
#     return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2) + (alpha / 2) * (torch.linalg.norm(grad(x)) ** 2)
def f(x,y,A_func, alpha):
    return data_fidelity(x,y,A_func) + alpha * huber_total_variation(x)

def data_fidelity(x,y,A_func):
    return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2)

def grad_f(x, y ,A_func, A_adj, alpha):
    return A_adj(A_func(x) - y) + alpha * laplacian(x)


def huber_total_variation(u):
    diff_x, diff_y = Du(u)
    #norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2), eps))
    zeros = torch.zeros_like(torch.sqrt(diff_x**2 + diff_y**2))
    norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2+1e-08)))
    #norm_2_1 = torch.sum(diff_x**2 + diff_y**2)
    return norm_2_1

def non_negativity(u):
    ## return 9999 if there is a negative element in u, and zero otherwise

    ## first, count how many negative elements there are
    num_nonneg = torch.sum(torch.where(u<0, torch.ones_like(u), torch.zeros_like(u)))
    return 9999999999*num_nonneg

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
#folder_path = 'Images_128/'
dataset = NaturalDataset(folder_path, num_imgs=NUM_IMAGES)
test_dataset = TestDataset(folder_path, num_imgs=NUM_TEST)

# blurred_list = ImageBlurDatasetGrad(dataset, alpha, noise_list, sigma_list)
blurred_list = ImageBlurDataset(dataset, alpha, noise_list, sigma_list, f, grad_f)

test_set = ImageBlurDataset(test_dataset, alpha, noise_list_test, sigma_list_test, f, grad_f)

# combined_dataset = ConcatDataset([dataset1, dataset2])

train_loader = DataLoader(dataset=blurred_list, batch_size=num_batch, shuffle=True, num_workers=0,
                          collate_fn=my_collate)

new_input = 'small_var_noise_4sig'
#tau_model = torch.load(f'grad_tau_model_ubd_above{new_input}.pth')
if load_models == True:
    try:
        update_model = torch.load(f'grad_update_model{new_input}.pth')
    except:
        update_model = CNN_LSTM().to(device)
    try:
        free_model = torch.load(f'grad_free_model{new_input}.pth')
    except:
        free_model = CNN_LSTM_Full().to(device)
    try:
        correction_model = torch.load(f'grad_correction_model{new_input}.pth')
    except:
        correction_model = CNN_LSTM_CORRECTION().to(device)
    # try:
    #     post_correction_model = torch.load(f'grad_post_correction_model{new_input}.pth')
    # except:
    #     post_correction_model = CNN_LSTM().to(device)
else:
    update_model = CNN_LSTM().to(device)
    free_model = CNN_LSTM_Full().to(device)
    correction_model = CNN_LSTM_CORRECTION().to(device)
    #post_correction_model = CNN_LSTM().to(device)

optimizer_update = torch.optim.Adam(update_model.parameters())
optimizer_free = torch.optim.Adam(free_model.parameters())
optimizer_correction = torch.optim.Adam(correction_model.parameters())
#optimizer_post_correction = torch.optim.Adam(post_correction_model.parameters())

if not load_models:
    print('START TRAINING')
    for epoch in range(NUM_EPOCHS):  # Number of epochs
        if epoch+1<num_iters:
            n_iters=epoch+1
        else:
            n_iters=num_iters
        #n_iters = epoch+1
        epoch_obj_update = 0
        epoch_obj_free = 0
        epoch_obj_correction = 0
        epoch_obj_post_correction = 0
        for i, batch in enumerate(train_loader):
            total_objective_update = 0
            total_objective_free = 0
            total_objective_correction = 0
            total_objective_post_correction = 0
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
                #f_x_star_j = f_x_star[j].to(device)


                #
                # print('Init f', f_j(x0_j.to(device), y_j.to(device)))
                # xs, taus = gradient_descent_fixed(f_j, x0_j.to(device), y_j.to(device), 0.5, tol=1e-06)
                # x_star = xs[-1]
                # f_x_star = f_j(x_star, y_j.to(device))
                # print('Final f', f_x_star)
                f_x_star = 0

                # update
                xs = gradient_descent_update(f_j, x0_j, y_j, update_model, float(sig_j), std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list, f_x_star)
                total_objective_update += obj

                # free model
                # xs = gradient_descent_modelfree(f_j, x0_j, y_j, free_model, float(sig_j), std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_free += obj

                # correction model
                xs, taus = gradient_descent_correctionNN(f_j, x0_j, y_j, correction_model, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list, f_x_star)
                total_objective_correction += obj


                # # post correction
                # xs, taus2 = gradient_descent_post_correction(f_j, x0_j, y_j, post_correction_model, tau_model, sig_j, std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list, f_x_star)
                # total_objective_post_correction += obj

                # LBFGS
                #xs, taus = LBFGS(grad_f_j, x0_j, y_j, LBFGS_model, sig_j, std_j, iters=n_iters)
                #obj, fs = function_evals(f_j, xs, y_j, wk_list)
                #total_objective_LBFGS += obj

            total_objective_update /= (num_batch)
            total_objective_free /= (num_batch)
            total_objective_correction /= (num_batch)
            total_objective_post_correction /= (num_batch)

            total_objective_update.backward()
            optimizer_update.step()
            optimizer_update.zero_grad()

            # total_objective_free.backward()
            # optimizer_free.step()
            # optimizer_free.zero_grad()

            total_objective_correction.backward()
            optimizer_correction.step()
            optimizer_correction.zero_grad()

            # total_objective_adagrad.backward()
            # optimizer_adagrad.step()
            # optimizer_adagrad.zero_grad()
            #
            # total_objective_rmsprop.backward()
            # optimizer_rmsprop.step()
            # optimizer_rmsprop.zero_grad()
            #
            # total_objective_adam.backward()
            # optimizer_adam.step()
            # optimizer_adam.zero_grad()

            # total_objective_post_correction.backward()
            # optimizer_post_correction.step()
            # optimizer_post_correction.zero_grad()

            #total_objective_LBFGS.backward()
            #optimizer_LBFGS.step()
            #optimizer_LBFGS.zero_grad()

            epoch_obj_update += total_objective_update
            epoch_obj_free += total_objective_free
            epoch_obj_correction += total_objective_correction
            epoch_obj_post_correction += total_objective_post_correction

        epoch_obj_update /= (NUM_IMAGES/num_batch)
        epoch_obj_free /= (NUM_IMAGES/num_batch)
        epoch_obj_correction /= (NUM_IMAGES/num_batch)
        epoch_obj_post_correction /= (NUM_IMAGES/num_batch)

        if epoch % 10 == 0:
            print("SAVING MODELS")
            new_input = 'small_var_noise_4sig'
            torch.save(update_model, f'grad_update_model{new_input}.pth')
            #torch.save(free_model, f'grad_free_model{new_input}.pth')
            torch.save(correction_model, f'grad_correction_model{new_input}.pth')
            #torch.save(post_correction_model, f'grad_post_correction_model{new_input}.pth')
            print("FINISHED SAVING MODELS")

        print(taus[-1])
        #print(taus2[-1])
        print(f"Epoch: {epoch}, Model Free Update: {epoch_obj_update.item()}, Correction: {epoch_obj_correction.item()}")

    print("SAVING MODELS")
    new_input = 'small_var_noise_4sig'
    torch.save(update_model, f'grad_update_model{new_input}.pth')
    torch.save(free_model, f'grad_free_model{new_input}.pth')
    torch.save(correction_model, f'grad_correction_model{new_input}.pth')
    # torch.save(adagrad_model, f'grad_adagrad_model{new_input}.pth')
    # torch.save(rmsprop_model, f'grad_rmsprop_model{new_input}.pth')
    # torch.save(adam_model, f'grad_adam_model{new_input}.pth')
    #torch.save(LBFGS_model, f'grad_LBFGS_model{new_input}.pth')
    #torch.save(post_correction_model, f'grad_post_correction_model{new_input}.pth')
    print("FINISHED SAVING MODELS")
