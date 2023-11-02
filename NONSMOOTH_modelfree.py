from torch.utils.data import DataLoader
import numpy as np
import torch
from datasets import NaturalDataset, TestDataset, ImageBlurDatasetNonsmooth, my_collate_nonsmooth
from nonsmooth_optimisers import CNN_LSTM, CNN_LSTM_Full, CNN_LSTM_CORRECTION, TAU_NET
from nonsmooth_algorithms import function_evals, gradient_descent_update, gradient_descent_modelfree, gradient_descent_correctionNN, PGD
from torch.nn import HuberLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

delta= 0.05
huber = HuberLoss(reduction='none',delta=delta)

load_models = False
alpha = 0.01
NUM_IMAGES = 160
NUM_TEST = 10

noise_list = [0.]
sigma_list = list(np.linspace(2, 8, 6))
noise_list_test = [0.]
sigma_list_test = list(np.linspace(3, 10, 7))

num_batch = 8
wk_list = [1]
wk_list2 = [-1]

n_iters=10

NUM_EPOCHS = 10000

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

def f(x,y,A_func, alpha):
    return data_fidelity(x,y,A_func) + alpha * huber_total_variation(x)

def data_fidelity(x,y,A_func):
    return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2)

#def huber(x, epsilon=0.05):
#    return torch.where(torch.abs(x) < epsilon, 0.5 * x**2/epsilon,  (x - 0.5 * epsilon))

def huber_total_variation(u, eps=0.05):
    diff_x, diff_y = Du(u)
    #norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2), eps))
    zeros = torch.zeros_like(torch.sqrt(diff_x**2 + diff_y**2))
    norm_2_1 = torch.sum(huber(zeros, torch.sqrt(diff_x**2 + diff_y**2+1e-08)))
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

blurred_list = ImageBlurDatasetNonsmooth(dataset, alpha, noise_list, sigma_list, f, data_fidelity, grad_data_fidelity, reg)

test_set = ImageBlurDatasetNonsmooth(test_dataset, alpha, noise_list_test, sigma_list_test, f, data_fidelity, grad_data_fidelity, reg)

train_loader = DataLoader(dataset=blurred_list, batch_size=num_batch, shuffle=True, num_workers=0,
                          collate_fn=my_collate_nonsmooth)
tau_model = torch.load('grad_tau_model_noise.pth')
if load_models == True:
    try:
        update_model = torch.load('nonsmooth_update_model.pth')
    except:
        update_model = CNN_LSTM().to(device)
    try:
        free_model = torch.load('nonsmooth_free_model.pth')
    except:
        free_model = CNN_LSTM_Full().to(device)
    try:
        correction_model = torch.load('nonsmooth_correction_model.pth')
    except:
        correction_model = CNN_LSTM_CORRECTION().to(device)
    try:
        tau_model_PGD = torch.load('nonsmooth_tau_model_PGD.pth')
    except:
        tau_model_PGD = TAU_NET().to(device)
    try:
        pgd_model = torch.load('nonsmooth_pgd_model.pth')
    except:
        pgd_model = CNN_LSTM().to(device)
else:
    update_model = CNN_LSTM().to(device)
    free_model = CNN_LSTM_Full().to(device)
    correction_model = CNN_LSTM_CORRECTION().to(device)
    tau_model_PGD = TAU_NET().to(device)
    pgd_model = CNN_LSTM().to(device)

optimizer_update = torch.optim.Adam(update_model.parameters())
optimizer_free = torch.optim.Adam(free_model.parameters())
optimizer_correction = torch.optim.Adam(correction_model.parameters())
#optimizer_tau = torch.optim.Adam(tau_model_PGD.parameters())
optimizer_pgd = torch.optim.Adam(list(pgd_model.parameters())+list(tau_model_PGD.parameters()))

if not load_models:
    print('START TRAINING')
    for epoch in range(NUM_EPOCHS):  # Number of epochs
        epoch_obj_update = 0
        epoch_obj_free = 0
        epoch_obj_correction = 0
        epoch_tau = 0
        epoch_pgd = 0
        for i, batch in enumerate(train_loader):
            total_objective_update = 0
            total_objective_free = 0
            total_objective_correction = 0
            total_tau = 0
            total_pgd = 0
            img, y, x0, std, sig, epsilon, A_func, A_adj, f, data_fit, grad_data_fit, reg = batch
            for j in range(img.shape[0]):  # iterate over items in the batch
                img_j = img[j].to(device)
                y_j = y[j].to(device)
                x0_j = x0[j].to(device)
                epsilon_j = epsilon[j].to(device)
                A_func_j = A_func[j]
                A_adj_j = A_adj[j]
                f_j = f[j]
                data_fit_j = data_fit[j]
                grad_data_fit_j = grad_data_fit[j]
                reg_j = reg[j]
                std_j = std[j].to(device)
                sig_j = sig[j].to(device)


                # update
                xs = gradient_descent_update(grad_data_fit_j, reg_j, x0_j, y_j, update_model, float(sig_j), std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_update += obj

                #grad_data_fit, reg, x0, y, update_model, sig, std, iters = grad_data_fit_j, reg_j, x0_j, y_j, update_model, float(sig_j), std_j, n_iters

                # free model
                xs = gradient_descent_modelfree(grad_data_fit_j, reg_j, x0_j, y_j, free_model, float(sig_j), std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_free += obj

                # correction model
                xs, taus = gradient_descent_correctionNN(grad_data_fit_j, reg_j, x0_j, y_j, correction_model, float(sig_j), std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_correction += obj

                # PGD
                xs, taus = PGD(grad_data_fit_j, reg_j, x0_j, y_j, pgd_model, tau_model_PGD, float(sig_j), std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_pgd += obj


            total_objective_update /= (num_batch)
            total_objective_free /= (num_batch)
            total_objective_correction /= (num_batch)

            total_objective_update.backward()
            torch.nn.utils.clip_grad_norm_(update_model.parameters(), max_norm=1)
            optimizer_update.step()
            optimizer_update.zero_grad()

            total_objective_free.backward()
            optimizer_free.step()
            optimizer_free.zero_grad()

            total_objective_correction.backward()
            optimizer_correction.step()
            optimizer_correction.zero_grad()

            total_pgd.backward()
            optimizer_pgd.step()
            optimizer_pgd.zero_grad()

            epoch_obj_update += total_objective_update
            epoch_obj_free += total_objective_free
            epoch_obj_correction += total_objective_correction
            epoch_pgd += total_pgd


        epoch_obj_update /= (NUM_IMAGES/num_batch)
        epoch_obj_free /= (NUM_IMAGES/num_batch)
        epoch_obj_correction /= (NUM_IMAGES/num_batch)
        epoch_pgd /= (NUM_IMAGES/num_batch)

        if epoch % 10 == 0:
            print("SAVING MODELS")
            new_input = ''
            torch.save(update_model, f'nonsmooth_update_model{new_input}.pth')
            torch.save(free_model, f'nonsmooth_free_model{new_input}.pth')
            torch.save(correction_model, f'nonsmooth_correction_model{new_input}.pth')
            torch.save(tau_model_PGD, f'nonsmooth_tau_model_PGD{new_input}.pth')
            torch.save(pgd_model, f'nonsmooth_pgd_model{new_input}.pth')
            print("FINISHED SAVING MODELS")

        print(f"Epoch: {epoch}, Model Free Update: {round(epoch_obj_update.item(),4)}, Model Free Func: {round(epoch_obj_free.item(),4)}, Correction: {round(epoch_obj_correction.item(),4)}, PGD: {round(epoch_pgd.item(),4)}")

print("SAVING MODELS")
new_input = ''
torch.save(update_model, f'nonsmooth_update_model{new_input}.pth')
torch.save(free_model, f'nonsmooth_free_model{new_input}.pth')
torch.save(correction_model, f'nonsmooth_correction_model{new_input}.pth')
torch.save(tau_model_PGD, f'nonsmooth_tau_model_PGD{new_input}.pth')
torch.save(pgd_model, f'nonsmooth_pgd_model{new_input}.pth')
print("FINISHED SAVING MODELS")


