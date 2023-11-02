

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
from functions import estimate_operator_norm
from datasets import NaturalDataset, TestDataset, ImageBlurDataset, my_collate
from optimisers import TauFuncNet, TauFunc10Net, TauFuncUnboundedNet, TauFuncUnboundedAboveNet, UpdateModel, UnrollingFunc, GeneralUpdateModel, FixedModel
from algorithms import function_evals, gradient_descent_fixed, gradient_descent_backtracking, gradient_descent_function, gradient_descent_update, gradient_descent_modelfree, gradient_descent_unrolling
from test_fns import get_boxplot
from torch.utils.data import ConcatDataset
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_models = True
alpha = 0.0001
NUM_IMAGES = 100
NUM_TEST = 10
noise_list = list(np.linspace(0, 0.2, 10))
#noise_list = [0.]#list(np.linspace(0, 0.2, 10))
sigma_list = list(np.linspace(1, 5, 10))
noise_list_test = [0.]#list(np.linspace(0, 0.2, 10))
#noise_list_test = [0.]#list(np.linspace(0, 0.2, 10))
sigma_list_test = list(np.linspace(1, 7, 7))
num_batch=8
wk_list = [1]
wk_list2 = [-1]

n_iters=10
## what about n_iters = 1, but then what about learned on this new generated one to get the next one? Will this just simplify to n_iters=10 for the learned one?
NUM_EPOCHS = 5

def f(x,y,A_func, alpha):
    return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2) + (alpha / 2) * (torch.linalg.norm(x) ** 2)

def grad_f(x, y ,A_func, A_adj, alpha):
    return A_adj(A_func(x) - y) + alpha * x




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

if load_models == True:
    try:
        tau_model = torch.load('tau_model.pth')
    except:
        tau_model = TauFuncNet().to(device)
    try:
        tau_model_10 = torch.load('tau_model10.pth')
    except:
        tau_model_10 = TauFunc10Net().to(device)
    try:
        tau_model_last = torch.load('tau_model_last.pth')
    except:
        tau_model_last = TauFuncNet().to(device)
    try:
        tau_model_ubd_above = torch.load('tau_model_ubd_above.pth')
    except:
        tau_model_ubd_above = TauFuncUnboundedAboveNet().to(device)
    try:
        tau_model_ubd = torch.load('tau_model_ubd.pth')
    except:
        tau_model_ubd = TauFuncUnboundedNet().to(device)
    try:
        update_model = torch.load('update_model.pth')
    except:
        update_model = UpdateModel().to(device)
    try:
        free_model = torch.load('free_model.pth')
    except:
        free_model = GeneralUpdateModel().to(device)
    try:
        unrolling_model = torch.load('unrolling_func_model.pth')
    except:
        unrolling_model = UnrollingFunc().to(device)
    try:
        fixed_model = torch.load('fixed_model.pth')
    except:
        fixed_model = FixedModel().to(device)
else:
    tau_model = TauFuncNet().to(device)
    tau_model_10 = TauFunc10Net().to(device)
    tau_model_last = TauFuncNet().to(device)
    tau_model_ubd_above = TauFuncUnboundedAboveNet().to(device)
    tau_model_ubd = TauFuncUnboundedNet().to(device)
    update_model = UpdateModel().to(device)
    free_model = GeneralUpdateModel().to(device)
    unrolling_model = UnrollingFunc().to(device)
    fixed_model = FixedModel().to(device)

optimizer = torch.optim.Adam(tau_model.parameters())
optimizer_10 = torch.optim.Adam(tau_model_10.parameters())
optimizer_last = torch.optim.Adam(tau_model_last.parameters())
optimizer_ubd_above = torch.optim.Adam(tau_model_ubd_above.parameters())
optimizer_ubd = torch.optim.Adam(tau_model_ubd.parameters())
optimizer_update = torch.optim.Adam(update_model.parameters())
optimizer_free = torch.optim.Adam(free_model.parameters())
optimizer_fixed = torch.optim.Adam(fixed_model.parameters())
optimizer_unrolling = torch.optim.Adam(unrolling_model.parameters())

if not load_models:
    print('START TRAINING')
    for epoch in range(NUM_EPOCHS):  # Number of epochs
        epoch_obj = 0
        epoch_obj_update = 0
        epoch_obj_10 = 0
        epoch_obj_free = 0
        epoch_obj_fixed = 0
        epoch_obj_unrolling = 0
        epoch_obj_last = 0
        epoch_obj_ubd_above = 0
        epoch_obj_ubd = 0
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

                ## Fixed and learned as a function
                tau_learned = fixed_model(torch.tensor([std_j, sig_j]).to(device))
                xs, taus = gradient_descent_fixed(grad_f_j, x0_j, y_j, tau_learned, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_fixed += obj

                ## learned function
                xs, taus = gradient_descent_function(grad_f_j, x0_j, y_j, tau_model, sig_j, std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective += obj

                # ## learned function 10
                # xs, taus = gradient_descent_function(grad_f_j, x0_j, y_j, tau_model_10, sig_j, std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_10 += obj
                #
                ## learned function last
                xs, taus = gradient_descent_function(grad_f_j, x0_j, y_j, tau_model_last, sig_j, std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, [-1])
                total_objective_last += obj

                ## learned function unbounded above
                xs, taus = gradient_descent_function(grad_f_j, x0_j, y_j, tau_model_ubd_above, sig_j, std_j, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_ubd_above += obj

                # ## learned function unbounded
                # xs, taus = gradient_descent_function(grad_f_j, x0_j, y_j, tau_model_ubd, sig_j, std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_ubd += obj

                # # update
                # xs = gradient_descent_update(grad_f_j, x0_j, y_j, update_model, float(sig_j), std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_update += obj
                #
                # # free model
                # xs = gradient_descent_modelfree(grad_f_j, x0_j, y_j, free_model, float(sig_j), std_j, iters=n_iters)
                # obj, fs = function_evals(f_j, xs, y_j, wk_list)
                # total_objective_free += obj

                # unrolling
                unrolling_taus = unrolling_model(sig_j, std_j).to(device)  ## not actually a func of x0
                xs, taus = gradient_descent_unrolling(grad_f_j, x0_j, y_j, unrolling_taus, iters=n_iters)
                obj, fs = function_evals(f_j, xs, y_j, wk_list)
                total_objective_unrolling += obj

            total_objective /= (num_batch * n_iters * NUM_IMAGES)
            total_objective_10 /= (num_batch * n_iters * NUM_IMAGES)
            total_objective_ubd_above /= (num_batch * n_iters * NUM_IMAGES)
            total_objective_ubd /= (num_batch * n_iters * NUM_IMAGES)
            total_objective_update /= (num_batch * n_iters * NUM_IMAGES)
            total_objective_free /= (num_batch * n_iters * NUM_IMAGES)
            total_objective_fixed /= (num_batch * n_iters * NUM_IMAGES)
            total_objective_unrolling /= (num_batch * n_iters * NUM_IMAGES)
            total_objective_last /= (num_batch * NUM_IMAGES)

            total_objective.backward()
            optimizer.step()
            optimizer.zero_grad()

            # total_objective_10.backward()
            # optimizer_10.step()
            # optimizer_10.zero_grad()
            #
            total_objective_last.backward()
            optimizer_last.step()
            optimizer_last.zero_grad()

            total_objective_ubd_above.backward()
            optimizer_ubd_above.step()
            optimizer_ubd_above.zero_grad()

            # total_objective_ubd.backward()
            # optimizer_ubd.step()
            # optimizer_ubd.zero_grad()
            #
            # total_objective_update.backward()
            # optimizer_update.step()
            # optimizer_update.zero_grad()
            #
            # total_objective_free.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            total_objective_fixed.backward()
            optimizer_fixed.step()
            optimizer_fixed.zero_grad()

            total_objective_unrolling.backward()
            optimizer_unrolling.step()
            optimizer_unrolling.zero_grad()

            epoch_obj += total_objective
            epoch_obj_10 += total_objective_10
            epoch_obj_last += total_objective_last
            epoch_obj_update += total_objective_update
            epoch_obj_free += total_objective_free
            epoch_obj_fixed += total_objective_fixed
            epoch_obj_unrolling += total_objective_unrolling
            epoch_obj_ubd_above += total_objective_ubd_above
            epoch_obj_ubd += total_objective_ubd

        print(f"Epoch: {epoch}, Objective: {epoch_obj.item()}, Last: {epoch_obj_last.item()}, Unbounded Above: {epoch_obj_ubd_above.item()}, Fixed: {epoch_obj_fixed.item()}, Unrolling: {epoch_obj_unrolling.item()}")
        #print(f"Epoch: {epoch}, Objective: {epoch_obj.item()}, 10: {epoch_obj_10.item()}, Last: {epoch_obj_last.item()}, Unbounded: {epoch_obj_ubd.item()}, Unbounded Above: {epoch_obj_ubd_above.item()}, Update: {epoch_obj_update}, Free: {epoch_obj_free}, Fixed: {epoch_obj_fixed.item()}, Unrolling: {epoch_obj_unrolling.item()}")

    print("SAVING MODELS")
    new_input = '_noise'
    torch.save(fixed_model, f'fixed_model{new_input}.pth')
    torch.save(tau_model, f'tau_model{new_input}.pth')
    torch.save(tau_model_10, f'tau_model_10{new_input}.pth')
    torch.save(tau_model_last, f'tau_model_last{new_input}.pth')
    torch.save(update_model, f'update_model{new_input}.pth')
    torch.save(free_model, f'free_model{new_input}.pth')
    torch.save(unrolling_model, f'unrolling_func_model{new_input}.pth')
    print("FINISHED SAVING MODELS")

tau_optimal = lambda a_norm, a_adj_norm: 1 / (alpha + a_norm * a_adj_norm)

test_iters = 100
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
time_nlfixed = 0
time_fixed = 0
time_learned = 0
time_unbounded_above = 0
time_unrolling = 0
time_backtracking = 0
with torch.no_grad():
    num = 0
    for lst in test_set:
        print(num)
        num += 1
        ### fixed not learned
        start_time = time.time()
        op_A = estimate_operator_norm(lst.A_func)
        op_adj = estimate_operator_norm(lst.A_adj)
        opt_tau = tau_optimal(op_A, op_adj)
        xs, taus = gradient_descent_fixed(lst.grad_f, lst.x0.to(device), lst.y.to(device), opt_tau, iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        fnl_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        fnl_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        fnl_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        fnl_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_nlfixed.append(fs)
        end_time = time.time()
        time_nlfixed += (end_time - start_time)

        ## Fixed and learned as a function
        start_time = time.time()
        tau_learned = fixed_model(torch.tensor([lst.std, lst.sig]).to(device))
        xs, taus = gradient_descent_fixed(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_learned,
                                          iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        flf_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        flf_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        flf_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        flf_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_fixed.append(fs)
        end_time = time.time()
        time_fixed += (end_time - start_time)

        ## learned function
        start_time = time.time()
        xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model,
                                             lst.sig.to(device), lst.std.to(device), iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        l_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        l_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        l_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        l_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list.append(fs)
        end_time = time.time()
        time_learned += (end_time - start_time)

        # ## learned function 10
        # xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model_10, lst.sig.to(device), lst.std.to(device), iters=test_iters)
        # obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        # l10_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        # l10_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        # l10_noise_list_taus[str(round(float(lst.std),3))].append(taus)
        # l10_sigma_list_taus[str(round(float(lst.sig),3))].append(taus)
        # fs_list_10.append(fs)
        #
        # ## last
        # xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model_last, lst.sig.to(device), lst.std.to(device), iters=test_iters)
        # obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        # ll_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        # ll_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        # ll_noise_list_taus[str(round(float(lst.std),3))].append(taus)
        # ll_sigma_list_taus[str(round(float(lst.sig),3))].append(taus)
        # fs_list_last.append(fs)
        #
        # ## unbounded
        # xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model_ubd, lst.sig.to(device), lst.std.to(device), iters=test_iters)
        # obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        # ubd_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        # ubd_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        # ubd_noise_list_taus[str(round(float(lst.std),3))].append(taus)
        # ubd_sigma_list_taus[str(round(float(lst.sig),3))].append(taus)
        # fs_list_unbounded.append(fs)

        ## unbounded above
        start_time = time.time()
        xs, taus = gradient_descent_function(lst.grad_f, lst.x0.to(device), lst.y.to(device), tau_model_ubd_above,
                                             lst.sig.to(device), lst.std.to(device), iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        ubd_above_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        ubd_above_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        ubd_above_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        ubd_above_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_unbounded_above.append(fs)
        end_time = time.time()
        time_unbounded_above += (end_time - start_time)

        # ## learned update
        # xs = gradient_descent_update(lst.grad_f, lst.x0.to(device), lst.y.to(device), update_model, lst.sig, lst.std, iters=test_iters)
        # obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        # upd_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        # upd_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        # fs_list_update.append(fs)
        #
        # ## learned entire update
        # xs = gradient_descent_modelfree(lst.grad_f, lst.x0.to(device), lst.y.to(device), free_model, lst.sig, float(lst.std), iters=test_iters)
        # obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        # mf_noise_obj_dict[str(round(float(lst.std),3))].append(obj)
        # mf_sigma_obj_dict[str(round(float(lst.sig),3))].append(obj)
        # fs_list_free.append(fs)

        ## Unrolling
        start_time = time.time()
        unrolling_taus = [round(float(i), 3) for i in
                          unrolling_model(lst.sig.to(device), lst.std.to(device))]  ## not actually a func of x0
        xs, taus = gradient_descent_unrolling(lst.grad_f, lst.x0.to(device), lst.y.to(device), unrolling_taus,
                                              iters=test_iters)
        obj, fs = function_evals(lst.f, xs, lst.y.to(device), wk_list)
        unr_noise_obj_dict[str(round(float(lst.std), 3))].append(obj)
        unr_sigma_obj_dict[str(round(float(lst.sig), 3))].append(obj)
        unr_noise_list_taus[str(round(float(lst.std), 3))].append(taus)
        unr_sigma_list_taus[str(round(float(lst.sig), 3))].append(taus)
        fs_list_unrolling.append(fs)
        end_time = time.time()
        time_unrolling += (end_time - start_time)

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


# Calculate average times
avg_time_nlfixed = time_nlfixed / num
avg_time_fixed = time_fixed / num
avg_time_learned = time_learned / num
avg_time_unbounded_above = time_unbounded_above / num
avg_time_unrolling = time_unrolling / num
avg_time_backtracking = time_backtracking / num

print("Average Time for fixed not learned:", avg_time_nlfixed)
print("Average Time for Fixed and learned as a function:", avg_time_fixed)
print("Average Time for learned function:", avg_time_learned)
print("Average Time for unbounded above:", avg_time_unbounded_above)
print("Average Time for Unrolling:", avg_time_unrolling)
print("Average Time for Backtracking:", avg_time_backtracking)


fs_list_agg = [[float(fs_list[i][k]) for i in range(len(fs_list))] for k in range(len(fs_list[0]))]
#fs_list_10_agg = [[float(fs_list_10[i][k]) for i in range(len(fs_list_10))] for k in range(len(fs_list_10[0]))]
#fs_list_last_agg = [[float(fs_list_last[i][k]) for i in range(len(fs_list_last))] for k in range(len(fs_list_last[0]))]
fs_list_nlfixed_agg = [[float(fs_list_nlfixed[i][k]) for i in range(len(fs_list_nlfixed))] for k in
                       range(len(fs_list_nlfixed[0]))]
fs_list_fixed_agg = [[float(fs_list_fixed[i][k]) for i in range(len(fs_list_fixed))] for k in
                     range(len(fs_list_fixed[0]))]
#fs_list_update_agg = [[float(fs_list_update[i][k]) for i in range(len(fs_list_update))] for k inrange(len(fs_list_update[0]))]
#fs_list_free_agg = [[float(fs_list_free[i][k]) for i in range(len(fs_list_free))] for k in range(len(fs_list_free[0]))]
fs_list_unrolling_agg = [[float(fs_list_unrolling[i][k]) for i in range(len(fs_list_unrolling))] for k in
                         range(len(fs_list_unrolling[0]))]
fs_list_bt_agg = [[float(fs_list_bt[i][k]) for i in range(len(fs_list_bt))] for k in range(len(fs_list_bt[0]))]
#fs_list_ubd_agg = [[float(fs_list_unbounded[i][k]) for i in range(len(fs_list_unbounded))] for k in range(len(fs_list_unbounded[0]))]
fs_list_ubd_above_agg = [[float(fs_list_unbounded_above[i][k]) for i in range(len(fs_list_unbounded_above))] for k in
                         range(len(fs_list_unbounded_above[0]))]

f_avg = [np.mean(i) for i in fs_list_agg]
#f_avg_10 = [np.mean(i) for i in fs_list_10_agg]
#f_avg_last = [np.mean(i) for i in fs_list_last_agg]
f_avg_nlfixed = [np.mean(i) for i in fs_list_nlfixed_agg]
f_avg_fixed = [np.mean(i) for i in fs_list_fixed_agg]
#f_avg_update = [np.mean(i) for i in fs_list_update_agg]
#f_avg_free = [np.mean(i) for i in fs_list_free_agg]
f_avg_unrolling = [np.mean(i) for i in fs_list_unrolling_agg]
f_avg_bt = [np.mean(i) for i in fs_list_bt_agg]
#f_avg_ubd = [np.mean(i) for i in fs_list_ubd_agg]
f_avg_ubd_above = [np.mean(i) for i in fs_list_ubd_above_agg]




plt.plot(f_avg, label='Learned Tau Function')
plt.plot(f_avg_nlfixed, label='Fixed')
plt.plot(f_avg_fixed, label='Fixed Learned')
plt.plot(f_avg_unrolling, label='Unrolling')
plt.plot(f_avg_bt, label='Backtracking')
plt.plot(f_avg_ubd_above, label='Unbounded Above')
plt.legend()
plt.show()

import pandas as pd
import seaborn as sns
# Create a new dataframe with the updated lists and index
df_updated = pd.DataFrame({
    'Iteration': list(range(len(f_avg))),
    'Learned Tau Function': f_avg,
    'Backtracking': f_avg_bt
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



f_min = [np.min(i) for i in fs_list_agg]
#f_min_10 = [np.min(i) for i in fs_list_10_agg]
#f_min_last = [np.min(i) for i in fs_list_last_agg]
f_min_nlfixed = [np.min(i) for i in fs_list_nlfixed_agg]
f_min_fixed = [np.min(i) for i in fs_list_fixed_agg]
#f_min_update = [np.min(i) for i in fs_list_update_agg]
#f_min_free = [np.min(i) for i in fs_list_free_agg]
f_min_unrolling = [np.min(i) for i in fs_list_unrolling_agg]
f_min_bt = [np.min(i) for i in fs_list_bt_agg]
#f_min_ubd = [np.min(i) for i in fs_list_ubd_agg]
f_min_ubd_above = [np.min(i) for i in fs_list_ubd_above_agg]
df_updated = pd.DataFrame({
    'Iteration': list(range(len(f_min))),
    'Learned Tau Function': f_min_unrolling,
    'Backtracking': f_min_bt
})
# Melt the dataframe to a long format, which is more suitable for seaborn
df_melted = df_updated.melt(id_vars='Iteration', var_name='method', value_name='value')
# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted, x='Iteration', y='value', hue='method')
# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Minimum over f of f(x_k) over test set')
plt.title('Comparison of Learned Tau Function and Backtracking')
plt.show()




# Calculate the difference between 'Learned Tau Function' and 'Backtracking'
difference = [a - b for a, b in zip(f_avg, f_avg_bt)]
# Create a new dataframe with the updated list and index
df_difference = pd.DataFrame({
    'Iteration': list(range(len(difference))),
    'Difference': difference
})
# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_difference, x='Iteration', y='Difference')
# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Difference')
plt.title('Learned Tau Function - Backtracking')
plt.show()




plt.plot(f_avg, label='Learned Tau Function')
plt.plot(f_avg_bt, label='Backtracking')
plt.legend()
plt.show()



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
plt.legend()
plt.show()

f_max = [np.max(i) for i in fs_list_agg]
#f_max_10 = [np.max(i) for i in fs_list_10_agg]
#f_max_last = [np.max(i) for i in fs_list_last_agg]
f_max_nlfixed = [np.max(i) for i in fs_list_nlfixed_agg]
f_max_fixed = [np.max(i) for i in fs_list_fixed_agg]
#f_max_update = [np.max(i) for i in fs_list_update_agg]
#f_max_free = [np.max(i) for i in fs_list_free_agg]
f_max_unrolling = [np.max(i) for i in fs_list_unrolling_agg]
f_max_bt = [np.max(i) for i in fs_list_bt_agg]
#f_max_ubd = [np.max(i) for i in fs_list_ubd_agg]
f_max_ubd_above = [np.max(i) for i in fs_list_ubd_above_agg]

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








# Create a dataframe from the adjusted data
df_boxplot = pd.DataFrame(l_sigma_obj_dict)
# Melt the dataframe to a long format, which is more suitable for seaborn
df_boxplot_melted = df_boxplot.melt(var_name='Category', value_name='Value')
# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_boxplot_melted, x='Category', y='Value', palette='Set3')
# Add labels and title
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Boxplot of Categories')
# Set seaborn style for nicer plots
sns.set(style="whitegrid")
plt.show()




data = {i:[float(k) for k in  l_sigma_obj_dict[i]] for i in l_sigma_obj_dict}
# Prepare data for seaborn
data_for_plot = [val for sublist in data.values() for val in sublist]
labels_for_plot = [key for key, sublist in data.items() for _ in sublist]

# Create the boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x=labels_for_plot, y=data_for_plot)
plt.title('Boxplots for the learned model')
plt.xlabel('Gaussian blur standard deviation')
plt.ylabel('Objective function value at iteration 100')
plt.show()



data = {i:[float(-bt_sigma_obj_dict[i][k] + l_sigma_obj_dict[i][k]) for k in  range(len(bt_sigma_obj_dict[i]))] for i in bt_sigma_obj_dict}
# Prepare data for seaborn
data_for_plot = [val for sublist in data.values() for val in sublist]
labels_for_plot = [key for key, sublist in data.items() for _ in sublist]
# Create the boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x=labels_for_plot, y=data_for_plot)
plt.title('Boxplots for the learned model minus backtracking')
plt.xlabel('Gaussian blur standard deviation')
plt.ylabel('Difference in bjective function value at iteration 100')
plt.show()



l_sigma_obj_dict = {i:[float(k) for k in  l_sigma_obj_dict[i]] for i in l_sigma_obj_dict}
bt_sigma_obj_dict




