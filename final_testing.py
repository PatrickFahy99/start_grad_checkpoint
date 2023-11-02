import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
from functions import estimate_operator_norm
import time
from huber_TV import power_iteration
from datasets import NaturalDataset, TestDataset, ImageBlurDataset, my_collate
from optimisers import TauFuncNet, TauFunc10Net, TauFuncUnboundedNet, TauFuncUnboundedAboveNet, UpdateModel, \
    UnrollingFunc, GeneralUpdateModel, FixedModel, FixedMomentumModel, AdagradModel, RMSPropModel, AdamModel, CNN_LSTM
from algorithms import function_evals, gradient_descent_fixed, gradient_descent_backtracking, gradient_descent_function, \
    gradient_descent_update, gradient_descent_modelfree, gradient_descent_unrolling, gradient_descent_fixed_nesterov, \
    gradient_descent_fixed_momentum, gradient_descent_heavy_ball, adagrad, rmsprop, adam, accelerated_gradient_descent, heavy_ball_function, gradient_descent_correctionNN
from test_fns import get_boxplot
from torch.utils.data import ConcatDataset
from grad_x import grad, laplacian
import time
import pandas as pd
import seaborn as sns
from torch.nn import HuberLoss
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_models = False

alpha = 0.0001

NUM_TEST = 20

noise_list_test = [5 / 255, 10 / 255, 15 / 255]
sigma_list_test = [4]

wk_list = [1]
wk_list2 = [-1]


def huber(s, epsilon=0.01):
    return torch.where(torch.abs(s) <= epsilon, (0.5 * s ** 2) / epsilon, s - 0.5 * epsilon)


def f(x, y, A_func, alpha):
    return data_fidelity(x, y, A_func) + alpha * huber_total_variation(x)


def correct_fs(fs):
    return [float(f) for f in fs]


def data_fidelity(x, y, A_func):
    return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2)


def grad_f(x, y, A_func, A_adj, alpha):
    return A_adj(A_func(x) - y) + alpha * laplacian(x)


def huber_total_variation(u, eps=0.05):
    diff_x, diff_y = Du(u)
    # norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2), eps))
    zeros = torch.zeros_like(torch.sqrt(diff_x ** 2 + diff_y ** 2))
    norm_2_1 = torch.sum(huber(torch.sqrt(diff_x ** 2 + diff_y ** 2 + 1e-08)))
    # norm_2_1 = torch.sum(diff_x**2 + diff_y**2)
    return norm_2_1


def grad_data_fidelity(x, y, A_func, A_adj):
    return A_adj(A_func(x) - y)


def reg(x, alpha):
    return alpha * huber_total_variation(x)


def Du(u):
    """Compute the discrete gradient of u using PyTorch."""
    diff_x = torch.diff(u, dim=1, prepend=u[:, :1])
    diff_y = torch.diff(u, dim=0, prepend=u[:1, :])
    return diff_x, diff_y


# Path to the folder containing the images
folder_path = 'notumour/'#'lhq_256/'

'C:\Users\Patrick\PycharmProjects\TFR\notumor\Tr-no_0016.jpg'

test_dataset = TestDataset(folder_path, num_imgs=NUM_TEST)

test_set = ImageBlurDataset(test_dataset, alpha, noise_list_test, sigma_list_test, f, grad_f)

new_input = 'small_var_noise_4sig'
fixed_model = torch.load(f'grad_fixed_model{new_input}.pth')
tau_model = torch.load(f'grad_tau_model_ubd_above{new_input}.pth')
fixed_heavy_model = torch.load(f'grad_fixed_heavy_model{new_input}.pth')
hb_func_model = torch.load(f'grad_hb_func_model{new_input}.pth')

correction_model = torch.load(f'grad_correction_model{new_input}.pth')


# post_correction_model = torch.load(f'grad_post_correction_model{new_input}.pth')

def get_opt_tau(func, func_adj):
    both = lambda x: func_adj(func(x))
    max_eval = power_iteration(both)
    min_eval = power_iteration(lambda x: both(x) - max_eval * torch.identity(x.shape[0]))
    return 2 / (max_eval + min_eval + 8 * alpha / 0.01)



def power_iteration(A_func, num_iterations=1000):
    v = torch.randn((lst.img.shape[0], lst.img.shape[0]))  # Initialize a random vector

    for _ in range(num_iterations):
        Av = A_func(v)  # Compute Av using your function
        eigenvalue = torch.dot(v.flatten(), Av.flatten())  # Rayleigh quotient
        v = Av / torch.norm(Av, 2)  # Normalize the vector

    return eigenvalue


op_tau = 1 / (1 + 8 * alpha / 0.01)
op_tau2 = 2 / (1 + 8 * alpha / 0.01)

num = 0
taus_func_list = []
taus_bt_list = []

taus_hb_list = []
betas_hb_list = []

num_iters_func = []
num_iters_fixed = []
num_iters_fixed_heavy = []
num_iters_correction = []
num_iters_heavy_func = []

num_iters_nonlearn1 = []
num_iters_nonlearn2 = []

fs_func_list = []
fs_fixed = []
fs_fixed_heavy = []
fs_correction_list = []
fs_heavy_func = []

scale_nums_list_func = []
scale_nums_list_fixed = []
scale_nums_list_fixed_heavy = []
scale_nums_list_correction = []
scale_nums_list_heavy_func = []

execution_times_fixed = []
execution_times_nonlearn1 = []
execution_times_nonlearn2 = []
execution_times_backtracking = []
execution_times_learned_function = []
execution_times_fixed_heavy = []
execution_times_heavy_ball_function = []
execution_times_correction = []

fs_nonlearn1 = []
fs_nonlearn2 = []
fs_bt_list = []

scale_nums_list_nonlearn1 = []
scale_nums_list_nonlearn2 = []

blur_list = []
noise_list = []
one_done = False
two_done = False
three_done = False
for lst in test_set:
    print(num)
    num += 1

    if lst.std == 5 / 255:
        if one_done:
            continue
    if lst.std == 10 / 255:
        if two_done:
            continue
    if lst.std == 15 / 255:
        if three_done:
            continue

    if lst.std == 5 / 255:
        one_done = True
    if lst.std == 10 / 255:
        two_done = True
    if lst.std == 15 / 255:
        three_done = True

    print(lst.std)

    blur_list.append(lst.sig)
    noise_list.append(lst.std)

    lst.std = torch.tensor(0.).to(device)
    lst.sig = torch.tensor(4.).to(device)

    op_tau_new = op_tau
    op_tau_new2 = op_tau2
    print('Tau Opt', op_tau_new)
    print('Tau Opt 2', op_tau_new2)

    print('FIXED LEARNED')
    start_time = time.time()
    tau_learned = 1.99#float(fixed_model(torch.tensor([lst.sig, lst.std]).to(device)).detach())
    print(tau_learned)
    x_star, taus, fs, iters, scale_nums = gradient_descent_fixed(lst.f, lst.x0.to(device), lst.y.to(device),
                                                                 tau_learned, tol=0.005)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_fixed.append(execution_time)
    fs = correct_fs(fs)
    taus = correct_fs(taus)
    fs_fixed.append(fs)
    num_iters_fixed.append(iters)
    scale_nums_list_fixed.append(scale_nums)
    # print(fs[:10])
    # print(fs[1000])
    #
    # print(x_star)
    # print(noise_list[-1])

    ### ADD TIME!!!

    print('FIXED NOT LEARNED 1')
    start_time = time.time()
    x_star2, taus2, fs2, iters2, scale_nums2 = gradient_descent_fixed(lst.f, lst.x0.to(device), lst.y.to(device),
                                                                      op_tau_new, tol=0.005)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_nonlearn1.append(execution_time)
    fs2 = correct_fs(fs2)
    taus2 = correct_fs(taus2)
    fs_nonlearn1.append(fs2)
    num_iters_nonlearn1.append(iters2)
    scale_nums_list_nonlearn1.append(scale_nums2)

    torch.cuda.empty_cache()

    print('FIXED NOT LEARNED 2')
    start_time = time.time()
    x_star22, taus22, fs22, iters22, scale_nums22 = gradient_descent_fixed(lst.f, lst.x0.to(device), lst.y.to(device),
                                                                           op_tau_new2, tol=0.005)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_nonlearn2.append(execution_time)
    fs22 = correct_fs(fs22)
    taus22 = correct_fs(taus22)
    fs_nonlearn2.append(fs22)
    num_iters_nonlearn2.append(iters22)
    scale_nums_list_nonlearn2.append(scale_nums22)

    torch.cuda.empty_cache()

    ## BACKTRACKING
    print('BACKTRACKING')
    start_time = time.time()
    x_star_bt, taus_bt, fs_bt, iters_bt, scale_nums_bt = gradient_descent_backtracking(lst.f, lst.x0.to(device),
                                                                                       lst.y.to(device), tol=0.005)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_backtracking.append(execution_time)
    fs_bt = correct_fs(fs_bt)
    taus_bt = correct_fs(taus_bt)
    fs_bt_list.append(fs_bt)
    num_iters_fixed.append(iters_bt)
    scale_nums_list_fixed.append(scale_nums_bt)
    taus_bt_list.append(taus_bt)

    print('LEARNED FUNCTION')
    start_time = time.time()
    x_star_func, taus_func, fs_func, iters_func, scale_nums_func = gradient_descent_function(lst.f, lst.x0.to(device),
                                                                                             lst.y.to(device),
                                                                                             tau_model, lst.sig,
                                                                                             lst.std, tol=0.005)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_learned_function.append(execution_time)
    fs_func = correct_fs(fs_func)
    taus_func = correct_fs(taus_func)
    fs_func_list.append(fs_func)
    num_iters_func.append(iters_func)
    taus_func_list.append(taus_func)
    scale_nums_list_func.append(scale_nums_func)
    print(taus_func[0], taus_func[-1])
    print('MIN/MAX TAU', np.min(taus_func), np.max(taus_func))
    torch.cuda.empty_cache()


    print('FIXED HEAVY BALL')
    start_time = time.time()
    hb_tau, hb_beta = fixed_heavy_model(torch.tensor([lst.sig, lst.std]).to(device))
    print(hb_tau, hb_beta)
    x_star_hb, taus_hb, fs_hb, iters_hb, scale_nums_hb = gradient_descent_heavy_ball(lst.f, lst.x0.to(device),
                                                                                     lst.y.to(device),
                                                                                     float(hb_tau.detach()),
                                                                                     float(hb_beta.detach()), tol=0.005)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_fixed_heavy.append(execution_time)
    fs_hb = correct_fs(fs_hb)
    taus_hb = correct_fs(taus_hb)
    fs_fixed_heavy.append(fs_hb)
    num_iters_fixed_heavy.append(iters_hb)
    scale_nums_list_fixed_heavy.append(scale_nums_hb)
    taus_hb_list.append(taus_hb)
    betas_hb_list.append(hb_beta)

    print('HEAVY BALL FUNCTION')
    start_time = time.time()
    x_star_hb_func, taus_hb_func, betas_hb_func, fs_hb_func, iters_hb_func, scale_nums_hb_func = heavy_ball_function(lst.f, lst.x0.to(device),
                                                                                                lst.y.to(device),
                                                                                                hb_func_model, lst.sig,
                                                                                                lst.std, tol=0.005)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_heavy_ball_function.append(execution_time)
    fs_hb_func = correct_fs(fs_hb_func)
    taus_hb_func = correct_fs(taus_hb_func)
    fs_heavy_func.append(fs_hb_func)
    num_iters_heavy_func.append(iters_hb_func)
    print(taus_hb_func[0], taus_hb_func[-1])
    print(betas_hb_func[0], betas_hb_func[-1])
    print('MIN/MAX TAU', np.min(taus_hb_func), np.max(taus_hb_func))
    print('MIN/MAX BETA', np.min(correct_fs(betas_hb_func)), np.max(correct_fs(betas_hb_func)))

    torch.cuda.empty_cache()

    print('CORRECTION')
    start_time = time.time()
    x_star_correction, taus_correction, fs_correction, iters_correction, scale_nums_correction = gradient_descent_correctionNN(
                                                        lst.f, lst.x0.to(device), lst.y.to(device), correction_model, tol=0.005)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_correction.append(execution_time)
    fs_correction = correct_fs(fs_correction)
    taus_correction = correct_fs(taus_correction)
    fs_correction_list.append(fs_correction)
    num_iters_correction.append(iters_correction)
    scale_nums_list_correction.append(scale_nums_correction)
    print(taus_correction[0], taus_correction[-1])
    print('MIN/MAX TAU', np.min(taus_correction), np.max(taus_correction))
    torch.cuda.empty_cache()

def pad_list(input_list, desired_length=10000):
    if len(input_list) >= desired_length:
        return input_list

    num_zeros_to_add = desired_length - len(input_list)
    padded_list = input_list + [np.nan] * num_zeros_to_add
    return padded_list


def calculate_column_averages(fs_lists):
    num_elements = len(fs_lists[0])
    num_lists = len(fs_lists)
    averages = [np.nanmedian([fs_lists[j][i] for j in range(num_lists)]) for i in range(num_elements)]
    return averages


def find_max_difference_columns(fs_lists1, fs_lists2):
    num_elements = len(fs_lists1[0])
    num_lists = len(fs_lists1)

    # Calculate the average differences for each column
    average_diffs = [sum(abs(fs_lists1[j][i] - fs_lists2[j][i]) for j in range(num_lists)) / num_lists for i in
                     range(num_elements)]

    # Find the column index with the maximum average difference
    max_diff_column = average_diffs.index(max(average_diffs))

    # Get the corresponding columns from fs_list1 and fs_list2
    max_diff_col_fs_list1 = [fs_lists1[j][max_diff_column] for j in range(num_lists)]
    max_diff_col_fs_list2 = [fs_lists2[j][max_diff_column] for j in range(num_lists)]

    return max_diff_col_fs_list1, max_diff_col_fs_list2


def find_min_difference_columns(fs_lists1, fs_lists2):
    num_elements = len(fs_lists1[0])
    num_lists = len(fs_lists1)

    # Calculate the average differences for each column
    average_diffs = [sum(abs(fs_lists1[j][i] - fs_lists2[j][i]) for j in range(num_lists)) / num_lists for i in
                     range(num_elements)]

    # Find the column index with the maximum average difference
    max_diff_column = average_diffs.index(min(average_diffs))

    # Get the corresponding columns from fs_list1 and fs_list2
    max_diff_col_fs_list1 = [fs_lists1[j][max_diff_column] for j in range(num_lists)]
    max_diff_col_fs_list2 = [fs_lists2[j][max_diff_column] for j in range(num_lists)]

    return max_diff_col_fs_list1, max_diff_col_fs_list2


def group_columns_by_std(fs_list, std_list):
    grouped_columns = {}

    for fs_values, std_value in zip(fs_list, std_list):
        if std_value not in grouped_columns:
            grouped_columns[std_value] = []

        grouped_columns[std_value].append(fs_values)

    return grouped_columns


def f_at_termination(fs_lists, nums_scales):
    terms = [i[0] for i in nums_scales]
    all = [fs_lists[i][terms[i]] for i in range(len(terms))]
    mean_f_term = np.mean(all)
    mean_min_f = np.mean([np.min(fs) for fs in fs_lists])
    return mean_f_term - mean_min_f, np.mean(terms)


# fs_fixed = [[item for item in inner_list if item != 0] for inner_list in fs_fixed]
# fs_nonlearn1 = [[item for item in inner_list if item != 0] for inner_list in fs_nonlearn1]
# fs_nonlearn2 = [[item for item in inner_list if item != 0] for inner_list in fs_nonlearn2]

fs_fixed = [pad_list(fs) for fs in fs_fixed]
fs_nonlearn1 = [pad_list(fs) for fs in fs_nonlearn1]
fs_nonlearn2 = [pad_list(fs) for fs in fs_nonlearn2]
f_x_stars = [np.nanmin(fs) for fs in fs_fixed]

avgs_fixed = calculate_column_averages(fs_fixed)
avgs_nonlearn1 = calculate_column_averages(fs_nonlearn1)
avgs_nonlearn2 = calculate_column_averages(fs_nonlearn2)

min_diff_fixed_1 = find_min_difference_columns(fs_fixed, fs_nonlearn1)
min_diff_fixed_2 = find_min_difference_columns(fs_fixed, fs_nonlearn2)
min_diff_1_2 = find_min_difference_columns(fs_nonlearn1, fs_nonlearn2)

noise_list

[f_x_stars[i] / fs_fixed[i][0] for i in range(len(fs_fixed))]
[f_x_stars[i] / fs_fixed[i][0] for i in range(len(fs_fixed))]
[(fs_fixed[i][100] - f_x_stars[i]) / (fs_fixed[i][0] - f_x_stars[i]) for i in range(len(fs_fixed))]

[np.mean([fs_fixed[i][j] - f_x_stars[i] for j in range(100)]) for i in
 range(len(fs_fixed))]  ## this places more weight on less noise, although some require the f x star anyway
[np.mean([fs_fixed[i][j] for j in range(100)]) for i in
 range(len(fs_fixed))]  ## this places more weight on less noise, although some require the f x star anyway

## both of these ratios report higher scores for higher noise ratios than lower. So some though will have to be made to


plt.figure()
plt.xscale("log")
plt.yscale("log")
plt.title('For Noise 5/255')
plt.plot(fs_fixed[0], label='Learned')
plt.plot(fs_nonlearn1[0], label='1/L')
plt.plot(fs_nonlearn2[0], label='2/($\mu$+L)')
plt.legend()
plt.show()

plt.figure()
plt.xscale("log")
plt.yscale("log")
plt.title('For Noise 10/255')
plt.plot(fs_fixed[1], label='Learned')
plt.plot(fs_nonlearn1[1], label='1/L')
plt.plot(fs_nonlearn2[1], label='2/($\mu$+L)')
plt.legend()
plt.show()

plt.figure()
plt.xscale("log")
plt.yscale("log")
plt.title('For Noise 15/255')
plt.plot(fs_fixed[4], label='Learned')
plt.plot(fs_nonlearn1[4], label='1/L')
plt.plot(fs_nonlearn2[4], label='2/($\mu$+L)')
plt.legend()
plt.show()

plt.plot(avgs_fixed[:2000], label='Learned')
plt.plot(avgs_nonlearn1[:2000], label='1/L')
plt.plot(avgs_nonlearn2[:2000], label='2/($\mu$+L)')
plt.legend()

plt.imshow(x_star_hb.detach().cpu().numpy(), cmap='gray')  #
plt.axis('off')
plt.tight_layout()
plt.show()

plt.imshow(lst.y, cmap='gray')  #
plt.axis('off')
plt.tight_layout()
plt.show()

plt.imshow(lst.x0, cmap='gray')  #
plt.axis('off')
plt.tight_layout()
plt.show()

plt.imshow(lst.img, cmap='gray')  #
plt.axis('off')
plt.tight_layout()
plt.show()

minimum = np.nanmin(fs_nonlearn2[0])
plt.figure()
plt.xscale("log")
plt.yscale("log")
plt.title('For Noise 10/255')
plt.plot([i - minimum for i in fs_fixed[0] if i != np.nan], label='Learned')
plt.plot([i - minimum for i in fs_nonlearn1[0] if i != np.nan], label='1/L')
plt.plot([i - minimum for i in fs_nonlearn2[0] if i != np.nan], label='2/($\mu$+L)')
plt.legend()
plt.show()

minimum = np.nanmin(fs_nonlearn2[0])
data1 = [i - minimum for i in fs_nonlearn1[0] if not np.isnan(i)]
data2 = [i - minimum for i in fs_nonlearn2[0] if not np.isnan(i)]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title('Loglog plot of $f(x_k) - f(x^*)$ for an example function')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label

# Add occasional points for clarity
plt.loglog(data1, label='1/L', marker='o', markersize=5, linestyle='-')
plt.loglog(data2, label='2/($\mu$+L)', marker='s', markersize=5, linestyle='-')

# Show legend
plt.legend()

# Show the plot
plt.show()


