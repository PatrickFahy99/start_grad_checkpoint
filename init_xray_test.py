

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
from functions import estimate_operator_norm
import time
from huber_TV import power_iteration
from datasets import NaturalDataset, TestDataset, XRayDataset, my_collate
from optimisers import TauFuncNet, TauFunc10Net, TauFuncUnboundedNet, TauFuncUnboundedAboveNet, UpdateModel, \
    UnrollingFunc, GeneralUpdateModel, FixedModel, FixedMomentumModel, AdagradModel, RMSPropModel, AdamModel, CNN_LSTM
from algorithms import gradient_descent_fixed, gradient_descent_backtracking, gradient_descent_const
from torch.utils.data import ConcatDataset
from grad_x import grad, laplacian
import time
import pandas as pd
from torch.nn import HuberLoss
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from LGS_module import LGS, LGS2, BackTrackingTau
import odl

device = 'cuda' if torch.cuda.is_available() else 'cpu'


alpha = 0.002

NUM_TEST = 100


def huber(s, epsilon=1e-08):
    return torch.where(torch.abs(s) <= epsilon, (0.5 * s ** 2) / epsilon, s - 0.5 * epsilon)

def Du(u):
    """Compute the discrete gradient of u using PyTorch."""
    diff_x = torch.diff(u, dim=1, prepend=u[:, :1])
    diff_y = torch.diff(u, dim=0, prepend=u[:1, :])
    return diff_x, diff_y

def f(x, y, A_func, alpha):
    return data_fidelity(x, y, A_func) + reg(x, alpha)


def correct_fs(fs):
    return [float(f) for f in fs]


def data_fidelity(x, y, A_func):
    return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2)


def huber_total_variation(u, eps=1e-08):
    diff_x, diff_y = Du(u)
    norm_2_1 = torch.sum(huber(torch.sqrt(diff_x ** 2 + diff_y ** 2 + 1e-12), eps))
    return norm_2_1


def reg(x, alpha):
    if len(x.shape) == 2:
        return alpha * huber_total_variation(x)
    elif len(x.shape) == 3:
        return alpha * huber_total_variation(x.squeeze(0))
    elif len(x.shape) == 4:
        return alpha * huber_total_variation(x.squeeze(0).squeeze(0))
    else:
        print('UH OH, WRONG SHAPE')

def new_reg_func(x, alpha, eps):
    if len(x.shape) == 2:
        return alpha * huber_total_variation(x, eps)
    elif len(x.shape) == 3:
        return alpha * huber_total_variation(x.squeeze(0), eps)
    elif len(x.shape) == 4:
        return alpha * huber_total_variation(x.squeeze(0).squeeze(0), eps)
    else:
        print('UH OH, WRONG SHAPE')

reg_func = lambda x: reg(x, alpha)

# Path to the folder containing the images
folder_path = r'C:\Users\Patrick\XRayL2O\notumor'
#folder_path = r'C:\Users\Patrick\XRayL2O\lhq_256'

test_dataset = TestDataset(folder_path, num_imgs=NUM_TEST)

std = 0.05
test_set = XRayDataset(test_dataset, alpha, f, std, 'limited')


from datasets import Blur2Dataset, ImageDataBlur2

blur_level = 10
blur_size = 7
test_set = Blur2Dataset(test_dataset, alpha, std, blur_level, blur_size)


# new_input = 'small_var_noise_4sig'
# fixed_model = torch.load(f'grad_fixed_model{new_input}.pth')
# tau_model = torch.load(f'grad_tau_model_ubd_above{new_input}.pth')
# fixed_heavy_model = torch.load(f'grad_fixed_heavy_model{new_input}.pth')
# hb_func_model = torch.load(f'grad_hb_func_model{new_input}.pth')

# correction_model = torch.load(f'grad_correction_model{new_input}.pth')


# post_correction_model = torch.load(f'grad_post_correction_model{new_input}.pth')

num = 0

num_iters_nonlearn1 = []
execution_times_nonlearn1 = []
fs_nonlearn1 = []




import torch
from torchvision.transforms import GaussianBlur


def gradient_descent_fixed_odl(f, operator, adjoint, reg_func, x0, y, tau_in):
    tau = tau_in
    xo = x0.clone().detach().double()  # Convert to double precision
    xs = [xo.clone().detach().cpu().numpy()]
    regs = [reg_func(xo.squeeze(0)).detach().cpu().numpy()]

    new_f = lambda x: 0.5 * torch.norm(operator(x) - y).double()**2 + reg_func(x.squeeze(0)).double()  # Convert intermediate calculations to double


    fs = [float(new_f(xo))]


    xor = torch.tensor(xo.clone().detach().squeeze(0), requires_grad=True)

    reg_value = reg_func(xor).double()  # Convert intermediate calculations to double
    
    #grad_reg_new = torch.autograd.grad(reg_value, xo)[0].double()  # Convert gradients to double
    reg_value.backward()
    grad_reg_new = xor.grad

    grad_fit = adjoint(operator(xo) - y).double()  # Convert intermediate calculations to double
    grad_f_new = grad_fit + grad_reg_new

    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients
    f_new = 0
    f_old = 1

    num=0
    while f_new < f_old+100:
        torch.cuda.empty_cache()
        num += 1
        if num %100 == 0:
            print(num,torch.norm(grad_f_new), f_new)
        f_old = new_f(xo)

        xor = torch.tensor(xo.clone().detach().squeeze(0), requires_grad=True)
        reg_value = reg_func(xor).double()  # Convert intermediate calculations to double
        #grad_reg_new = torch.autograd.grad(reg_value, xo)[0].double()  # Convert gradients to double
        reg_value.backward()
        grad_reg_new = xor.grad

        grad_fit = adjoint(operator(xo) - y).double()  # Convert intermediate calculations to double
        grad_f_new = grad_fit + grad_reg_new

        

        xn = xo - tau * grad_f_new.to(device)
        xs.append(xn.detach().cpu().numpy())

        regs=reg_func(xn.squeeze(0))#regs.append(reg_func(xn).detach().cpu().numpy())
        xo = xn
        f_new = new_f(xn)
        fs.append(float(f_new))
        if f_new > f_old:
            print(float(f_new), float(f_old), float(torch.norm(grad_f_new)))
    return xs, fs, regs


def power_iteration(func, num_iterations=1000, N=256):
    u = torch.rand(1,N,N)
    for _ in range(num_iterations):
        v = func(u)
        norm_v = torch.linalg.norm(v.view(-1))
        u = v / norm_v
    operator_norm = torch.linalg.norm(func(u).view(-1)) / torch.linalg.norm(u.view(-1))
    return operator_norm.item()


def gradient_descent_fixed_odl_blur(operator, x0, y, tau_in, reg, alpha):

    xo = x0.clone().detach().double().to(device) # Convert to double precision
    xs = [xo.clone().detach().cpu().numpy()]

    tau = tau_in


    new_f = lambda x: (0.5 * torch.norm(operator(x) - y)**2 + reg(x, alpha)).double() # Convert intermediate calculations to double


    fs = [float(new_f(xo))]

    xor = torch.tensor(xo.clone().detach(), requires_grad=True)
    fit = new_f(xor).double()  # Convert intermediate calculations to double
    fit.backward()
    grad_fit = xor.grad
    grad_f_new = grad_fit #+ grad_reg_new

    xo = xo.clone().detach().to(device)  # Detach the tensor AFTER computing gradients
    f_new = 0
    f_old = 1

    num=0
    num_bigger = 0
    while f_new < f_old+100:
        torch.cuda.empty_cache()
        num += 1
        if num %100 == 0:
            print(num,torch.norm(grad_f_new), f_new)
        f_old = new_f(xo)

        xor = torch.tensor(xo.clone().detach(), requires_grad=True)
        fit = new_f(xor).double()  # Convert intermediate calculations to double
        fit.backward()
        grad_fit = xor.grad
        grad_f_new = grad_fit

        xn = xo - tau * grad_f_new.to(device)
        xs.append(xn.detach().cpu().numpy())

        xo = xn
        f_new = new_f(xn)
        fs.append(float(f_new))
        if f_new > f_old:
            num_bigger += 1
            print(float(f_new), float(f_old), float(torch.norm(grad_f_new)))
            if num_bigger == 1000:
                return xs, fs
    return xs, fs


def gradient_descent_fixed_iters_odl_blur(operator, x0, y, tau_in, reg, alpha, iters):

    xo = x0.clone().detach().double().to(device) # Convert to double precision
    xs = [xo.clone().detach().cpu().numpy()]

    tau = tau_in


    new_f = lambda x: (0.5 * torch.norm(operator(x) - y)**2 + reg(x, alpha)).double() # Convert intermediate calculations to double


    fs = [float(new_f(xo))]

    xor = torch.tensor(xo.clone().detach(), requires_grad=True)
    fit = new_f(xor).double()  # Convert intermediate calculations to double
    fit.backward()
    grad_fit = xor.grad
    grad_f_new = grad_fit #+ grad_reg_new

    xo = xo.clone().detach().to(device)  # Detach the tensor AFTER computing gradients
    f_new = 0
    f_old = 1

    num=0
    while num < iters:
        torch.cuda.empty_cache()
        num += 1

        xor = torch.tensor(xo.clone().detach(), requires_grad=True)
        fit = new_f(xor).double()  # Convert intermediate calculations to double
        fit.backward()
        grad_fit = xor.grad
        grad_f_new = grad_fit

        xn = xo - tau * grad_f_new.to(device)
        xs.append(xn.detach().cpu().numpy())

        xo = xn
        f_new = new_f(xn)
        fs.append(float(f_new))
    return xs, fs








def heavy_ball_fixed_iters_odl_blur(operator, x0, y, tau_in, beta, reg, alpha, iters):

    xo = x0.clone().detach().double().to(device) # Convert to double precision
    xs = [xo.clone().detach().cpu().numpy()]

    tau = tau_in


    new_f = lambda x: (0.5 * torch.norm(operator(x) - y)**2 + reg(x, alpha)).double() # Convert intermediate calculations to double


    fs = [float(new_f(xo))]

    xor = torch.tensor(xo.clone().detach(), requires_grad=True)
    fit = new_f(xor).double()  # Convert intermediate calculations to double
    fit.backward()
    grad_fit = xor.grad
    grad_f_new = grad_fit #+ grad_reg_new

    xo = xo.clone().detach().to(device)  # Detach the tensor AFTER computing gradients
    xm1 = xo
    f_new = 0
    f_old = 1

    num=0
    while num < iters:
        torch.cuda.empty_cache()
        num += 1

        xor = torch.tensor(xo.clone().detach(), requires_grad=True)
        fit = new_f(xor).double()  # Convert intermediate calculations to double
        fit.backward()
        grad_fit = xor.grad
        grad_f_new = grad_fit

        xn = xo - tau * grad_f_new.to(device) + beta*(xo-xm1)
        xs.append(xn.detach().cpu().numpy())

        xm1 = xo
        xo = xn
        f_new = new_f(xn)
        fs.append(float(f_new))
    return xs, fs






def gradient_descent_fixed_iters_odl(f, operator, adjoint, reg_func, x0, y, tau_in, iters):

    tau = tau_in
    xo = x0.clone().detach().double()  # Convert to double precision
    xs = [xo.clone().detach().cpu().numpy()]
    regs = [reg_func(xo).detach().cpu().numpy()]

    new_f = lambda x: 0.5 * torch.norm(operator(x) - y).double()**2 + reg_func(x).double()  # Convert intermediate calculations to double

    fs = [float(new_f(xo))]


    xo = torch.tensor(xo, requires_grad=True) # Set requires_grad to True
    reg_value = reg_func(xo).double()  # Convert intermediate calculations to double
    #grad_reg_new = torch.autograd.grad(reg_value, xo)[0].double()  # Convert gradients to double
    reg_value.backward()
    grad_reg_new = xo.grad

    grad_fit = adjoint(operator(xo) - y).double()  # Convert intermediate calculations to double
    grad_f_new = grad_fit + grad_reg_new

    

    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients
    f_new = 0

    num=0
    while num < iters:
        torch.cuda.empty_cache()
        num += 1
        if num %100 == 0:
            print(num,torch.norm(grad_f_new), f_new)

        xo = torch.tensor(xo, requires_grad=True)  # Set requires_grad to True
        reg_value = reg_func(xo).double()  # Convert intermediate calculations to double
        #grad_reg_new = torch.autograd.grad(reg_value, xo)[0].double()  # Convert gradients to double
        reg_value.backward()
        grad_reg_new = xo.grad

        grad_fit = adjoint(operator(xo) - y).double()  # Convert intermediate calculations to double
        grad_f_new = grad_fit + grad_reg_new

        xn = xo - tau * grad_f_new.to(device)
        xs.append(xn.detach().cpu().numpy())

        regs.append(reg_func(xn).detach().cpu().numpy())
        xo = xn
        f_new = new_f(xn)
        fs.append(float(f_new))
    return xs, fs, regs








def gradient_descent_fixed(f, operator, adjoint, reg_func, x0, y, tau_in, tol=1e-12):
    tau = tau_in
    xo = x0.clone().detach().double()  # Convert to double precision
    xs = [xo]
    num = 0
    taus = []

    new_f = lambda x: 0.5 * torch.norm(operator(x) - y).double()**2 + reg_func(x).double()  # Convert intermediate calculations to double

    f_value = new_f(xo)
    fs = [float(f_value)]

    xo = xo.requires_grad_(True)  # Set requires_grad to True
    reg_value = reg_func(xo).double()  # Convert intermediate calculations to double
    grad_reg_new = torch.autograd.grad(reg_value, xo)[0].double()  # Convert gradients to double

    grad_fit = adjoint(operator(xo) - y).double()  # Convert intermediate calculations to double
    grad_f_new = grad_fit + grad_reg_new

    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients

    scale_tau = False


    while torch.norm(grad_f_new) > tol:

        torch.cuda.empty_cache()

        if num % 100 == 0:
            print(num, f_value, torch.norm(grad_f_new))

        go = torch.norm(grad_f_new)
        if not scale_tau:
            tau = tau_in#.to(device)
        xn = xo.to(device) - tau * grad_f_new.to(device)

        xn.requires_grad_(True)
        f_value = new_f(xn)

        reg_value = reg_func(xn).double()  # Convert intermediate calculations to double
        grad_reg_new = torch.autograd.grad(reg_value, xn)[0].double()  # Convert gradients to double

        grad_fit = adjoint(operator(xn) - y).double()  # Convert intermediate calculations to double
        grad_f_new = grad_fit + grad_reg_new

        
        
        xn = xn.clone().detach()

        gn = torch.norm(grad_f_new)
        if go == gn:
            return xn, taus, fs
        if f(xn,y)>f(xo,y):
            res += 1
            tau *= 0.9
            scale_tau = True
            if res>1000:
                print('too many reductions')
                return xn, taus, fs
        else:
            scale_tau = False
            res = 0
            xs=xn#xs.append(xn)
            fs.append(float(f_value))
            taus.append(float(tau))
            xo = xn.clone().detach()
            num+=1
    return xs, taus, fs

def objective_function_hb(x):
    loss = 0
    num=0
    for lst in test_set:   
        num+=1
        if num<10:
            #xs, fs, reg_fs = gradient_descent_fixed_iters_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), torch.tensor([x]).to(device), 1000)
            xs, fs = heavy_ball_fixed_iters_odl_blur(lst.operator, lst.x0, lst.y, torch.tensor([x[0]]).to(device), torch.tensor([x[1]]).to(device), reg, alpha, 1000)
            loss += fs[-1]/NUM_TEST
    print(loss, x)
    # Your objective function here
    return loss



def objective_function_eps(x, reg_new):
    loss = 0
    num=0
    for lst in test_set:   
        num+=1
        if num<10:
            #xs, fs, reg_fs = gradient_descent_fixed_iters_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), torch.tensor([x]).to(device), 1000)
            xs, fs = gradient_descent_fixed_iters_odl_blur(lst.operator, lst.x0, lst.y, torch.tensor([x]).to(device), reg_new, alpha, 1000)
            loss += fs[-1]/NUM_TEST
    print(loss, x)
    # Your objective function here
    return loss

def objective_function(x):
    loss = 0
    num=0
    for lst in test_set:   
        num+=1
        if num<10:
            #xs, fs, reg_fs = gradient_descent_fixed_iters_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), torch.tensor([x]).to(device), 1000)
            xs, fs = gradient_descent_fixed_iters_odl_blur(lst.operator, lst.x0, lst.y, torch.tensor([x]).to(device), reg, alpha, 1000)
            loss += fs[-1]/NUM_TEST
    print(loss, x)
    # Your objective function here
    return loss

# from scipy.optimize import minimize
# # time method:
# start_time = time.time()
# op_tau_new = 1 / (test_set[0].operator_norm**2 + 8 * alpha / 1e-08)

# # print(1/(8 * alpha / 1e-08))
# # print(1 / test_set[0].operator_norm**2)
# #print(odl.power_method_opnorm(reg_func, atol=0, rtol=0, maxiter=10000))

# print(op_tau_new)
# result = minimize(objective_function, 2*op_tau_new, method='nelder-mead')
# end_time = time.time()
# execution_time = end_time - start_time
# print(execution_time)
# print(result.x[0])
# #learned 0.19617930103


# from scipy.optimize import minimize
# # time method:
# start_time = time.time()
# op_tau_new = 1 / (test_set[0].operator_norm**2 + 8 * alpha / 1e-08)
# print(alpha)
# print('OPT TAU', op_tau_new)
# for alpha in [0, 0.0000001, 0.000001,0.00001,0.0001,0.001,1e-08,0.1, 1]:
#     print('ALPHA', alpha)
#     result = minimize(objective_function, 1, method='nelder-mead', tol=0.001)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(execution_time)
#     print('FINAL LEARNED TAU FIXED')
#     print(result.x[0])
#learned 0.6309675766619465



# from scipy.optimize import minimize
# # time method:
# start_time = time.time()
# op_tau_new = 1 / (test_set[0].operator_norm**2 + 8 * alpha / 1e-08)
# print(alpha)
# print('OPT TAU', op_tau_new)
# result = minimize(objective_function_hb, (1,1), method='nelder-mead')
# end_time = time.time()
# execution_time = end_time - start_time
# print(execution_time)
# print('FINAL LEARNED TAU FIXED')
# print(result.x[0])
## learned: 1.09921875  0.8984375


# from scipy.optimize import minimize
# # alpha = 1e-03
# # # time method:
# start_time = time.time()
# print(alpha)
# #for eps in [1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1]:
# eps = 1e-08
# op_tau_new = 1 / (test_set[0].operator_norm**2 + 8 * alpha / eps)
# print('OPT TAU', op_tau_new)
# print('EPSILON', eps)
# new_reg = lambda x, alpha : new_reg_func(x, alpha, eps)
# new_obj = lambda x : objective_function_eps(x, new_reg)
# result = minimize(new_obj, op_tau_new, method='nelder-mead', tol=0.001)
# end_time = time.time()
# execution_time = end_time - start_time
# print(execution_time)
# print('FINAL LEARNED TAU FIXED')
# print(result.x[0])

# sdfsdf

import torch.nn as nn
import torch.nn.functional as F

class TauModel(nn.Module):
    def __init__(self, adjoint_operator_module, operator_module, reg, in_channels, out_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg = reg
        
        self.operator = operator_module
        self.adjoint_operator = adjoint_operator_module
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, y, n_iter=1):

        x_list = []
        tau_list = []
        
        for i in range(n_iter):

            adjoint_eval = self.adjoint_operator(self.operator(x) - y)

            x = x.requires_grad_(True)
            new_reg_value = self.reg(x)
            grad_reg_new = torch.autograd.grad(new_reg_value, x)[0]

            if i == 0:
                x_old = x
                adjoint_eval_old = adjoint_eval
                grad_reg_old = grad_reg_new

            #print(x.shape, adjoint_eval.shape, x_old.shape, adjoint_eval_old.shape, grad_reg_new.shape, grad_reg_old.shape)
            u = torch.cat([x, adjoint_eval, x_old, adjoint_eval_old, grad_reg_new, grad_reg_old], dim=0)
            u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.flatten()  # Flatten the tensor
            u = torch.relu(self.fc1(u))
            new_tau = F.softplus(self.fc2(u))

            #u = layers2[i].to(device)(u).to(device)

            #df = -self.step_length *u[:,0:1,:,:]
            df = - new_tau * (adjoint_eval + grad_reg_new)
            
            x = x + df

            x_old = x
            adjoint_eval_old = adjoint_eval
            grad_reg_old = grad_reg_new

            x_list.append(x.detach().cpu().numpy())
            tau_list.append(float(new_tau))
        
        return x, tau_list, x_list
    







#### BLUR


for lst in test_set:

    num += 1


    # num+=1
    if num<6:
        continue

    tau_opt = 1/(lst.operator_norm**2 + 8*alpha/1e-08)
    print('OPTIMAL TAU', tau_opt)
    print('FIXED NOT LEARNED 1')
    start_time = time.time()
    #xs, fs = gradient_descent_fixed_iters_odl_blur(lst.operator, lst.x0, lst.y, 1.0125, reg, alpha, 10)
    xs, fs = gradient_descent_fixed_odl_blur(lst.operator, lst.x0, lst.y, 1.0125, reg, alpha)
    plt.imshow(torch.tensor(xs[-1]).squeeze(0).cpu().numpy(), cmap='gray')
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_nonlearn1.append(execution_time)
    fs = correct_fs(fs)
    #taus2 = correct_fs(taus2)
    fs_nonlearn1.append(fs)
    print(fs[-1])
    torch.cuda.empty_cache()
    #xs = [xs[i] for i in range(len(xs))]
    plt.semilogy([i-min(fs) for i in fs])
    plt.show()


    # print('FIXED NOT LEARNED 2')
    # start_time = time.time()
    # xs2, fs2, regs = gradient_descent_fixed_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 2*op_tau_new)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # execution_times_nonlearn1.append(execution_time)
    # fs2 = correct_fs(fs2)
    # #taus2 = correct_fs(taus2)
    # fs_nonlearn1.append(fs2)
    # print(fs2[-1])
    # torch.cuda.empty_cache()


    print('FIXED LEARNED')
    start_time = time.time()
    xs22, fs22 = gradient_descent_fixed_odl_blur(lst.operator, lst.x0, lst.y, 0.6309675766619465, reg, alpha)
    plt.imshow(torch.tensor(xs22[-1]).squeeze(0).cpu().numpy(), cmap='gray')
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_nonlearn1.append(execution_time)
    fs = correct_fs(fs)
    #taus2 = correct_fs(taus2)
    fs_nonlearn1.append(fs)
    print(fs22[-1])
    torch.cuda.empty_cache()
    #xs = [xs[i] for i in range(len(xs))]
    plt.semilogy([i-min(fs22) for i in fs22])
    plt.show()


    print('EXACT')
    xse, taus, fs_exact = gradient_descent_fixed(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 0.19617930103)
    torch.cuda.empty_cache()
    plt.imshow(xs[-1].squeeze(0), cmap='gray')
    plt.show()
    plt.imshow(lst.img.squeeze(0), cmap='gray')

    print(fs_exact[-1])

    # xs_b, fs_b, regs = gradient_descent_fixed_iters_odl(lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 0.198,1000)                                                                     
    

    # dfgdfg

    # print('IMAGE REG, DATA', 'REG', reg_func(lst.img), 'FIT', data_fidelity(lst.img.unsqueeze(0).to(device), lst.y, lst.A_func), 'TOTAL', lst.f(lst.img.unsqueeze(0).to(device), lst.y))
    # print('APPROXIMATION', 'REG', reg_func(xs[-1]), 'FIT', data_fidelity(xs[-1].unsqueeze(0).to(device), lst.y, lst.A_func), 'TOTAL', lst.f(xs[-1].unsqueeze(0).to(device), lst.y))    #sdfsdfsdf

    # #xs, fs, regs = gradient_descent_fixed_iters_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 0.0987868702216, 200)
    
    plt.imshow(torch.tensor(xs[10]).squeeze(0).cpu().numpy(), cmap='gray')
    # ## 0.105 is divergent


    # plt.semilogy([i-min(fs_exact) for i in fs2], label='LEARNED')

    n_iter = 1000
    fig, ax = plt.subplots()
    ax.semilogy([i-min(fs_exact) for i in fs22 if i>=min(fs)], label='Learned Fixed')
    ax.semilogy([i-min(fs_exact) for i in fs], label='1/L', color='red')
    ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
    ax.axvspan(n_iter, len(fs), facecolor='yellow', alpha=0.5)
    ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
    ax.set_xlabel('Iteration Number')  # Replace with your actual label
    ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    ax.legend()
    plt.show()

    n_iter = 1000
    fig, ax = plt.subplots()
    ax.semilogy([i-min(fs_exact) for i in fs22 if i>=min(fs)], label='Learned Fixed')
    ax.semilogy([i-min(fs_exact) for i in fs], label='1/L', color='red')
    ax.semilogy([i-min(fs_exact) for i in fs2 if i>=min(fs)], label='2/L', color='black')
    ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
    ax.axvspan(n_iter, len(fs), facecolor='yellow', alpha=0.5)
    ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs 2/L vs learned $\phi$')
    ax.set_xlabel('Iteration Number')  # Replace with your actual label
    ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    ax.legend()
    plt.show()


    n_iter = 1000
    fig, ax = plt.subplots()
    ax.semilogy([i-min(fs_exact) for i in fs22 if i>min(fs)], label='Learned Fixed')
    ax.semilogy([i-min(fs_exact) for i in fs_b], label='$\phi = 0.198$', color='red')
    ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
    ax.axvspan(n_iter, len([i for i in fs22 if i>min(fs)]), facecolor='yellow', alpha=0.5)
    ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$.  learned $\phi$ vs $\phi = 0.198$')
    ax.set_xlabel('Iteration Number')  # Replace with your actual label
    ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    ax.legend()
    plt.show()


    # LEARNED TAU
    print('LEARNED TAU')
    tau_net = TauModel(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
    #tau_net.load_state_dict(torch.load('TAU_UNSUPERVISED.pth', map_location=device))
    tau_net.load_state_dict(torch.load('TAU_UNSUPERVISED_LIMITED.pth', map_location=device))

    # tau_net.eval() 

    x0_tensor = lst.x0[:, None, :, :]
    y_tensor = lst.y[:, None, :, :]

    # Now, you can use these 4D tensors as inputs to your network


    tau_reco, _, total_lst = tau_net(x0_tensor, y_tensor, n_iter=2000)
    plt.imshow(tau_reco.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='gray')
    plt.imshow(lst.img.squeeze(0).cpu().detach().numpy(), cmap='gray')
    fs_tau = [lst.f(torch.tensor(x).to(device), lst.y) for x in total_lst]
    fs_tau = correct_fs(fs_tau)
    taus_U = [float(t) for t in _]

    fs_tau.insert(0, fs22[0])




    n_iter = 15
    fig, ax = plt.subplots()
    ax.semilogy([i-min(fs_exact) for i in fs_tau[:2000]], label='Learned Function')
    ax.semilogy([i-min(fs_exact) for i in fs[:2000]], label='1/L', color='black')
    ax.semilogy([i-min(fs_exact) for i in fs22[:2000]], label='Learned Constant', color='red')
    ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
    ax.axvspan(n_iter, 2000, facecolor='yellow', alpha=0.5)
    ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$.  learned function vs learned constant vs 1/L')
    ax.set_xlabel('Iteration Number')  # Replace with your actual label
    ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    ax.legend()
    plt.show()

    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title("Learned $\phi$ function vs Iteration Number")
    plt.xlabel("Iteration Number")
    plt.ylabel("$\phi$")
    plt.scatter(range(len(taus_U)), taus_U, marker='.', color='red')
    plt.axhline(y=0.0852, color='blue', linestyle='--', label=f'1/L')
    plt.grid(True)
    plt.legend()
    plt.show()



    fghfgh

    print('SAFEGUARDED TAU MODEL')
    ### PROVABLY CONVERGENT IF THE F IS AT MOST WHAT GRADIENT DESCENT GIVES, BUT CHECKING AT EACH ITERATION IS COSTLY? MAYBE NOT
    fs_tau_sg = [lst.f(lst.x0, lst.y)]
    taus_sg = []
    xn = x0_tensor
    for i in range(400):
        tau_reco_sg, _, total_lst = tau_net(xn, y_tensor, n_iter=1)
        new_f = lst.f(tau_reco_sg, lst.y)
        grad_des_f = lst.f(xn, lst.y)
        xn = gradient_descent_fixed(lst.f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
        if new_f < grad_des_f:
            fs_tau_sg.append(new_f)
            xn = tau_reco_sg
            taus_sg.append(float(_[0]))
        else:
            fs_tau_sg.append(grad_des_f)
            taus_sg.append(op_tau_new)
    fs_tau_sg = correct_fs(fs_tau_sg)




    print('SAFEGUARDED UNSUPERVISED TAU MODEL')
    from LGS_train_module import TauModel
    tau_net = TauModel(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
    tau_net.load_state_dict(torch.load('TAU_UNSUPERVISED.pth', map_location=device))
    x0_tensor = lst.x0[:, None, :, :]
    y_tensor = lst.y[None, None, :, :]
    ### PROVABLY CONVERGENT IF THE F IS AT MOST WHAT GRADIENT DESCENT GIVES, BUT CHECKING AT EACH ITERATION IS COSTLY? MAYBE NOT
    fs_tau_Usg = [lst.f(lst.x0, lst.y)]
    taus_Usg = []
    xn = x0_tensor
    for i in range(400):
        tau_reco_Usg, _, total_lst = tau_net(xn, y_tensor, n_iter=1)
        new_f = lst.f(tau_reco_Usg, lst.y)
        grad_des_f = lst.f(xn, lst.y)
        xn = gradient_descent_fixed(lst.f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
        if new_f < grad_des_f:
            fs_tau_Usg.append(new_f)
            xn = tau_reco_Usg
            taus_Usg.append(float(_[0]))
        else:
            fs_tau_Usg.append(grad_des_f)
            taus_Usg.append(op_tau_new)
    fs_tau_Usg = correct_fs(fs_tau_Usg)

    break




#### XRAY
# for lst in test_set:

#     num += 1


#     #op_tau_new = get_opt_tau1(lst.A_func, lst.A_adj)
#     op_tau_new = 1 / (lst.operator_norm**2 + 8 * alpha / 1e-08)
#     #print(lst.operator_norm)
#     #print(op_tau_new)

#     ## learning constant tau:
#     ### ENCOPORATE ALL IN LST IN THIS: JUST AVERAG ERROR OVER ALL TEST SETS

#     # num+=1
#     if num<6:
#         continue
#     # print(num)

#     # data_fit = data_fidelity(lst.img.unsqueeze(0).to(device), lst.y, lst.A_func)
#     # reg_calc = reg_func(lst.img.unsqueeze(0).to(device))


#     # print('FIXED NOT LEARNED 1')
#     # start_time = time.time()
#     # x_star2, taus2, fs2, iters2, scale_nums2 = gradient_descent_fixed(lst.f, lst.x0.double().to(device), lst.y.double().to(device),
#     #                                                                   op_tau_new, tol=0.005)
#     # end_time = time.time()
#     # execution_time = end_time - start_time
#     # execution_times_nonlearn1.append(execution_time)
#     # fs2 = correct_fs(fs2)
#     # taus2 = correct_fs(taus2)
#     # fs_nonlearn1.append(fs2)
#     # num_iters_nonlearn1.append(iters2)

#     print('FIXED NOT LEARNED 1')
#     start_time = time.time()
#     #xs, fs, regs = gradient_descent_fixed_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), op_tau_new)
#     xs, fs = gradient_descent_fixed_odl(lst.img.unsqueeze(0))
#     plt.imshow(torch.tensor(xs[-1]).squeeze(0).cpu().numpy(), cmap='gray')
#     end_time = time.time()
#     execution_time = end_time - start_time
#     execution_times_nonlearn1.append(execution_time)
#     fs = correct_fs(fs)
#     #taus2 = correct_fs(taus2)
#     fs_nonlearn1.append(fs)
#     print(fs[-1])
#     torch.cuda.empty_cache()
#     #xs = [xs[i] for i in range(len(xs))]
#     plt.semilogy([i-min(fs) for i in fs])
#     plt.show()
#     asdfasdf

#     # print('FIXED NOT LEARNED 2')
#     # start_time = time.time()
#     # xs2, fs2, regs = gradient_descent_fixed_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 2*op_tau_new)
#     # end_time = time.time()
#     # execution_time = end_time - start_time
#     # execution_times_nonlearn1.append(execution_time)
#     # fs2 = correct_fs(fs2)
#     # #taus2 = correct_fs(taus2)
#     # fs_nonlearn1.append(fs2)
#     # print(fs2[-1])
#     # torch.cuda.empty_cache()


#     print('FIXED LEARNED')
#     start_time = time.time()
#     xs22, fs22, regs = gradient_descent_fixed_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 0.19617930103)                                                                     
#     end_time = time.time()
#     #fs22 = [lst.f(x, lst.y) for x in x_star2]
#     execution_time = end_time - start_time
#     execution_times_nonlearn1.append(execution_time)
#     #taus2 = correct_fs(taus2)
#     fs22 = correct_fs(fs22)
#     print(fs22[-1])
#     torch.cuda.empty_cache()


#     print('EXACT')
#     xse, taus, fs_exact = gradient_descent_fixed(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 0.19617930103)
#     torch.cuda.empty_cache()
#     plt.imshow(xs[-1].squeeze(0), cmap='gray')
#     plt.show()
#     plt.imshow(lst.img.squeeze(0), cmap='gray')

#     print(fs_exact[-1])

#     # xs_b, fs_b, regs = gradient_descent_fixed_iters_odl(lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 0.198,1000)                                                                     
    

#     # dfgdfg

#     # print('IMAGE REG, DATA', 'REG', reg_func(lst.img), 'FIT', data_fidelity(lst.img.unsqueeze(0).to(device), lst.y, lst.A_func), 'TOTAL', lst.f(lst.img.unsqueeze(0).to(device), lst.y))
#     # print('APPROXIMATION', 'REG', reg_func(xs[-1]), 'FIT', data_fidelity(xs[-1].unsqueeze(0).to(device), lst.y, lst.A_func), 'TOTAL', lst.f(xs[-1].unsqueeze(0).to(device), lst.y))    #sdfsdfsdf

#     # #xs, fs, regs = gradient_descent_fixed_iters_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 0.0987868702216, 200)
    
#     plt.imshow(torch.tensor(xs[10]).squeeze(0).cpu().numpy(), cmap='gray')
#     # ## 0.105 is divergent

#     dfgdfg

#     # plt.semilogy([i-min(fs_exact) for i in fs2], label='LEARNED')

#     n_iter = 1000
#     fig, ax = plt.subplots()
#     ax.semilogy([i-min(fs_exact) for i in fs22 if i>=min(fs)], label='Learned Fixed')
#     ax.semilogy([i-min(fs_exact) for i in fs], label='1/L', color='red')
#     ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
#     ax.axvspan(n_iter, len(fs), facecolor='yellow', alpha=0.5)
#     ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
#     ax.set_xlabel('Iteration Number')  # Replace with your actual label
#     ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     ax.legend()
#     plt.show()

#     n_iter = 1000
#     fig, ax = plt.subplots()
#     ax.semilogy([i-min(fs_exact) for i in fs22 if i>=min(fs)], label='Learned Fixed')
#     ax.semilogy([i-min(fs_exact) for i in fs], label='1/L', color='red')
#     ax.semilogy([i-min(fs_exact) for i in fs2 if i>=min(fs)], label='2/L', color='black')
#     ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
#     ax.axvspan(n_iter, len(fs), facecolor='yellow', alpha=0.5)
#     ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs 2/L vs learned $\phi$')
#     ax.set_xlabel('Iteration Number')  # Replace with your actual label
#     ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     ax.legend()
#     plt.show()


#     n_iter = 1000
#     fig, ax = plt.subplots()
#     ax.semilogy([i-min(fs_exact) for i in fs22 if i>min(fs)], label='Learned Fixed')
#     ax.semilogy([i-min(fs_exact) for i in fs_b], label='$\phi = 0.198$', color='red')
#     ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
#     ax.axvspan(n_iter, len([i for i in fs22 if i>min(fs)]), facecolor='yellow', alpha=0.5)
#     ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$.  learned $\phi$ vs $\phi = 0.198$')
#     ax.set_xlabel('Iteration Number')  # Replace with your actual label
#     ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     ax.legend()
#     plt.show()


#     # n_iter = 1000
#     # fig, ax = plt.subplots()
#     # ax.semilogy([i-min(fs_exact) for i in fs_exact], label='LEARNED')
#     # ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
#     # ax.axvspan(n_iter, len(fs_exact), facecolor='red', alpha=0.5)
#     # ax.set_xlabel('X-axis Label')
#     # ax.set_ylabel('Y-axis Label')
#     # ax.legend()
#     # plt.show()


#     # print('FIXED BIGGER 1')
#     # start_time = time.time()
#     # x_star2, taus2 = gradient_descent_fixed(lst.f, lst.x0.double().to(device), lst.y.double().to(device),
#     #                                                                   0.18, max_iter=1000)
#     # end_time = time.time()
#     # fs2b = [lst.f(x, lst.y) for x in x_star2]
#     # execution_time = end_time - start_time
#     # execution_times_nonlearn1.append(execution_time)
#     # taus2 = correct_fs(taus2)
#     # fs2b = correct_fs(fs2b)
#     # torch.cuda.empty_cache()

#     # print('FIXED BIGGER 2')
#     # start_time = time.time()
#     # x_star2, taus2 = gradient_descent_fixed(lst.f, lst.x0.double().to(device), lst.y.double().to(device),
#     #                                                                   0.19, max_iter=1000)
#     # end_time = time.time()
#     # fs2bb = [lst.f(x, lst.y) for x in x_star2]
#     # execution_time = end_time - start_time
#     # execution_times_nonlearn1.append(execution_time)
#     # taus2 = correct_fs(taus2)
#     # fs2bb = correct_fs(fs2bb)
#     # torch.cuda.empty_cache()



#     # print('FIXED 2/L')
#     # start_time = time.time()
#     # x_star2, taus2 = gradient_descent_fixed(lst.f, lst.x0.double().to(device), lst.y.double().to(device),
#     #                                                                   2*op_tau_new)
#     # end_time = time.time()
#     # fs2l = [lst.f(x, lst.y) for x in x_star2]
#     # execution_time = end_time - start_time
#     # execution_times_nonlearn1.append(execution_time)
#     # fs2l = correct_fs(fs2l)
#     # taus2 = correct_fs(taus2)

#     # print('FIXED 1/L')
#     # start_time = time.time()
#     # x_star2, taus2 = gradient_descent_fixed(lst.f, lst.x0.double().to(device), lst.y.double().to(device),
#     #                                                                   op_tau_new)
#     # end_time = time.time()
#     # fs2 = [lst.f(x, lst.y) for x in x_star2]
#     # execution_time = end_time - start_time
#     # execution_times_nonlearn1.append(execution_time)
#     # fs2 = correct_fs(fs2)
#     # taus2 = correct_fs(taus2)

#     # torch.cuda.empty_cache()

#     # print('Learned', len(fs22))
#     # print('2/L', len(fs2l))



#     # print(min(fs2l),min(fs22))
#     # if min(fs2l)>min(fs22):
#     #     print('bad')
#     #     break
#     # else:
#     #     print('next one')
#     #     continue

#     # min_f = np.min([min(fs22), min(fs2l), min(fs2)])
#     # sns.set(style="whitegrid")
#     # plt.figure(figsize=(10, 6))
#     # plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs vs 2/L vs learned $\phi$')
#     # plt.xlabel('Iteration Number')  # Replace with your actual label
#     # plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     # plt.semilogy([i - min_f for i in [i for i in fs2l if i - min_f>1e-04]], label='2/L')
#     # plt.semilogy([i - min_f for i in [i for i in fs2 if i - min_f>1e-04]], label='1/L')
#     # plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Fixed')
#     # plt.semilogy([i - min_f for i in [i for i in fs2b if i - min_f>1e-04]], label='Slightly Bigger')
#     # plt.semilogy([i - min_f for i in [i for i in fs2bb if i - min_f>1e-04]], label='More Slightly Bigger')
#     # plt.legend()
#     # plt.show()

#     # print('FIXED SMALL')
#     # start_time = time.time()
#     # x_star2, taus2, fs2small, iters2, scale_nums2 = gradient_descent_fixed(lst.f, lst.x0.double().to(device), lst.y.double().to(device),
#     #                                                                   1e-08, tol=0.005)
#     # end_time = time.time()
#     # execution_time = end_time - start_time
#     # execution_times_nonlearn1.append(execution_time)
#     # fs2small = correct_fs(fs2small)
#     # taus2 = correct_fs(taus2)
#     # fs_nonlearn1.append(fs2)
#     # num_iters_nonlearn1.append(iters2)

#     # print('FIXED LARGE')
#     # start_time = time.time()
#     # x_star2, taus2, fs2large, iters2, scale_nums2 = gradient_descent_fixed(lst.f, lst.x0.double().to(device), lst.y.double().to(device),
#     #                                                                   0.15, tol=0.005)
#     # end_time = time.time()
#     # execution_time = end_time - start_time
#     # execution_times_nonlearn1.append(execution_time)
#     # fs2large = correct_fs(fs2large)
#     # taus2 = correct_fs(taus2)
#     # fs_nonlearn1.append(fs2)
#     # num_iters_nonlearn1.append(iters2)

#     # print('FIXED LEARNED')
#     # start_time = time.time()
#     # x_star2, taus2, fs22, iters2, scale_nums2 = gradient_descent_fixed(lst.f, lst.x0.double().to(device), lst.y.double().to(device),
#     #                                                                   0.0626, tol=0.005)
#     # end_time = time.time()
#     # execution_time = end_time - start_time
#     # execution_times_nonlearn1.append(execution_time)
#     # fs22 = correct_fs(fs22)
#     # taus2 = correct_fs(taus2)
#     # fs_nonlearn1.append(fs2)
#     # num_iters_nonlearn1.append(iters2)

#     # torch.cuda.empty_cache()
    


#     # # #BACKTRACKING

#     # print('BACKTRACKING')
#     # start_time = time.time()
#     # x_star_bt, taus_bt, fs_bt, iters_bt, scale_nums_bt = gradient_descent_backtracking(lst.f, lst.x0.double().to(device),
#     #                                                                                    lst.y.double().to(device), tol=1e-08)
#     # end_time = time.time()
#     # execution_time = end_time - start_time
#     # #execution_times_backtracking.append(execution_time)
#     # fs_bt = correct_fs(fs_bt)
#     # taus_bt = correct_fs(taus_bt)
#     # #fs_bt_list.append(fs_bt)
#     # #num_iters_fixed.append(iters_bt)
#     # #scale_nums_list_fixed.append(scale_nums_bt)
#     # #taus_bt_list.append(taus_bt)

#     # ## LEARNED
#     # print('LEARNED')
#     # LGS_net = LGS(lst.A_adj, lst.A_func, lst.y, lst.x0, in_channels=2, out_channels=1, step_length=0.1, n_iter=100).to(device)
#     # LGS_net.load_state_dict(torch.load('Deep Learning Reconstructions\LGS1_005.pth', map_location=device))

#     # LGS_net.eval()

#     # x0_tensor = lst.x0[:, None, :, :]
#     # y_tensor = lst.y[None, None, :, :]

#     # # Now, you can use these 4D tensors as inputs to your network
#     # LGS_reco, _, total_lst = LGS_net(x0_tensor, y_tensor)
#     # fs_learn = [lst.f(x, lst.y) for x in total_lst]
#     # fs_learn = correct_fs(fs_learn)


#     # ## LEARNED BY ME
#     # print('LEARNED BY ME')
#     # from LGS_train_module import LGD2
#     # LGS_net = LGD2(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
#     # LGS_net.load_state_dict(torch.load('CORR_LG_SUPERVISED_GT.pth', map_location=device))

#     # LGS_net.eval()

#     # x0_tensor = lst.x0[:, None, :, :]
#     # y_tensor = lst.y[None, None, :, :]

#     # # Now, you can use these 4D tensors as inputs to your network
#     # LGS_reco, _, total_lst = LGS_net(x0_tensor, y_tensor, n_iter=100)
#     # fs_learn2 = [lst.f(x, lst.y) for x in total_lst]
#     # fs_learn2 = correct_fs(fs_learn2)


#     # ## LEARNED BY ME - UNSUPERVISED
#     # print('LEARNED BY ME - UNSUPERVISED')
#     # LGS_netu = LGD2(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
#     # LGS_netu.load_state_dict(torch.load('CORR_LG_UNSUPERVISED_GT.pth', map_location=device))

#     # LGS_netu.eval()

#     # x0_tensor = lst.x0[:, None, :, :]
#     # y_tensor = lst.y[None, None, :, :]

#     # # Now, you can use these 4D tensors as inputs to your network
#     # LGS_recou, _, total_lstu = LGS_netu(x0_tensor, y_tensor, n_iter=100)
#     # fs_learn2u = [lst.f(x, lst.y) for x in total_lstu]
#     # fs_learn2u = correct_fs(fs_learn2u)

#     # plt.semilogy([i for i in fs_learn2], label='SUPERVISED')
#     # plt.semilogy([i for i in fs_learn2u], label='UNSUPERVISED')
#     # plt.legend()

#     # LEARNED TAU
#     print('LEARNED TAU')
#     tau_net = TauModel(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
#     #tau_net.load_state_dict(torch.load('TAU_UNSUPERVISED.pth', map_location=device))
#     tau_net.load_state_dict(torch.load('TAU_UNSUPERVISED_LIMITED.pth', map_location=device))

#     # tau_net.eval() 

#     x0_tensor = lst.x0[:, None, :, :]
#     y_tensor = lst.y[:, None, :, :]

#     # Now, you can use these 4D tensors as inputs to your network


#     tau_reco, _, total_lst = tau_net(x0_tensor, y_tensor, n_iter=2000)
#     plt.imshow(tau_reco.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='gray')
#     plt.imshow(lst.img.squeeze(0).cpu().detach().numpy(), cmap='gray')
#     fs_tau = [lst.f(torch.tensor(x).to(device), lst.y) for x in total_lst]
#     fs_tau = correct_fs(fs_tau)
#     taus_U = [float(t) for t in _]

#     fs_tau.insert(0, fs22[0])




#     n_iter = 15
#     fig, ax = plt.subplots()
#     ax.semilogy([i-min(fs_exact) for i in fs_tau[:2000]], label='Learned Function')
#     ax.semilogy([i-min(fs_exact) for i in fs[:2000]], label='1/L', color='black')
#     ax.semilogy([i-min(fs_exact) for i in fs22[:2000]], label='Learned Constant', color='red')
#     ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
#     ax.axvspan(n_iter, 2000, facecolor='yellow', alpha=0.5)
#     ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$.  learned function vs learned constant vs 1/L')
#     ax.set_xlabel('Iteration Number')  # Replace with your actual label
#     ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     ax.legend()
#     plt.show()



#     n_iter = 15
#     fig, ax = plt.subplots()
#     ax.semilogy([i-min(fs_exact) for i in fs_tau[:50]], label='Learned Function')
#     ax.semilogy([i-min(fs_exact) for i in fs[:50]], label='1/L', color='black')
#     ax.semilogy([i-min(fs_exact) for i in fs22[:50]], label='Learned Constant', color='red')
#     ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
#     ax.axvspan(n_iter, len(fs_tau[:50]), facecolor='yellow', alpha=0.5)
#     ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$.  learned function vs learned constant vs 1/L')
#     ax.set_xlabel('Iteration Number')  # Replace with your actual label
#     ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     ax.legend()
#     plt.show()


#     ertert

    
#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title("Learned $\phi$ function vs Iteration Number")
#     plt.xlabel("Iteration Number")
#     plt.ylabel("$\phi$")
#     plt.scatter(range(len(taus_U)), taus_U, marker='.', color='red')
#     plt.axhline(y=0.0852, color='blue', linestyle='--', label=f'1/L')
#     plt.grid(True)
#     plt.legend()
#     plt.show()



#     n_iter = 15
#     fig, ax = plt.subplots()
#     ax.scatter(range(50), taus_U[:50], marker='.', color='red')
#     ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
#     ax.axvspan(n_iter, 50, facecolor='yellow', alpha=0.5)
#     ax.axhline(y=0.0852, color='blue', linestyle='--', label=f'1/L')
#     ax.set_title("Learned $\phi$ function vs Iteration Number")
#     ax.set_xlabel("Iteration Number")  # Replace with your actual label
#     ax.set_ylabel("$\phi$")  # Replace with your actual label
#     ax.legend()
#     plt.show()



#     fghfgh

#     print('SAFEGUARDED TAU MODEL')
#     ### PROVABLY CONVERGENT IF THE F IS AT MOST WHAT GRADIENT DESCENT GIVES, BUT CHECKING AT EACH ITERATION IS COSTLY? MAYBE NOT
#     fs_tau_sg = [lst.f(lst.x0, lst.y)]
#     taus_sg = []
#     xn = x0_tensor
#     for i in range(400):
#         tau_reco_sg, _, total_lst = tau_net(xn, y_tensor, n_iter=1)
#         new_f = lst.f(tau_reco_sg, lst.y)
#         grad_des_f = lst.f(xn, lst.y)
#         xn = gradient_descent_fixed(lst.f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
#         if new_f < grad_des_f:
#             fs_tau_sg.append(new_f)
#             xn = tau_reco_sg
#             taus_sg.append(float(_[0]))
#         else:
#             fs_tau_sg.append(grad_des_f)
#             taus_sg.append(op_tau_new)
#     fs_tau_sg = correct_fs(fs_tau_sg)




#     print('SAFEGUARDED UNSUPERVISED TAU MODEL')
#     from LGS_train_module import TauModel
#     tau_net = TauModel(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
#     tau_net.load_state_dict(torch.load('TAU_UNSUPERVISED.pth', map_location=device))
#     x0_tensor = lst.x0[:, None, :, :]
#     y_tensor = lst.y[None, None, :, :]
#     ### PROVABLY CONVERGENT IF THE F IS AT MOST WHAT GRADIENT DESCENT GIVES, BUT CHECKING AT EACH ITERATION IS COSTLY? MAYBE NOT
#     fs_tau_Usg = [lst.f(lst.x0, lst.y)]
#     taus_Usg = []
#     xn = x0_tensor
#     for i in range(400):
#         tau_reco_Usg, _, total_lst = tau_net(xn, y_tensor, n_iter=1)
#         new_f = lst.f(tau_reco_Usg, lst.y)
#         grad_des_f = lst.f(xn, lst.y)
#         xn = gradient_descent_fixed(lst.f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
#         if new_f < grad_des_f:
#             fs_tau_Usg.append(new_f)
#             xn = tau_reco_Usg
#             taus_Usg.append(float(_[0]))
#         else:
#             fs_tau_Usg.append(grad_des_f)
#             taus_Usg.append(op_tau_new)
#     fs_tau_Usg = correct_fs(fs_tau_Usg)


#         ## SAFEGUARDED ZONE TAU MODEL

#     # print('SAFEGUARDED ZONE TAU MODEL')

#     # tau_net_zone = TauModel(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
#     # tau_net_zone.load_state_dict(torch.load('ZONE_CORR_TAU_MODEL.pth', map_location=device))

#     # tau_net_zone.eval() 

#     # x0_tensor = lst.x0[:, None, :, :]
#     # y_tensor = lst.y[None, None, :, :]

#     # # Now, you can use these 4D tensors as inputs to your network


#     # #tau_reco_zone, _, total_lst_zone = tau_net_zone(x0_tensor, y_tensor, n_iter=100)
#     # #fs_tau_zone = [lst.f(x, lst.y) for x in total_lst_zone]
#     # #fs_tau_zone = correct_fs(fs_tau_zone)

#     # fs_tau_sg_zone = [lst.f(lst.x0, lst.y)]
#     # taus_sg_zone = []
#     # xn = x0_tensor
#     # for i in range(400):
#     #     tau_reco_sg_zone, _, total_lst = tau_net_zone(xn, y_tensor, n_iter=1)
#     #     new_f = lst.f(tau_reco_sg_zone, lst.y)
#     #     if new_f < fs_tau_sg_zone[-1]:
#     #         fs_tau_sg_zone.append(new_f)
#     #         xn = tau_reco_sg_zone
#     #         taus_sg_zone.append(float(_[0]))
#     #     else:
#     #         xn = gradient_descent_fixed(lst.f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
#     #         fs_tau_sg_zone.append(lst.f(xn, lst.y))
#     #         taus_sg_zone.append(op_tau_new)
#     # fs_tau_sg_zone = correct_fs(fs_tau_sg_zone)

#     # for i in range(400):
#     #     tau_reco_sg_zone, _, total_lst = tau_net_zone(xn, y_tensor, n_iter=1)
#     #     xn = gradient_descent_fixed(lst.f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
#     #     new_f = lst.f(tau_reco_sg_zone, lst.y)
#     #     grad_des_f = lst.f(xn, lst.y)
#     #     if new_f < grad_des_f:
#     #         fs_tau_sg_zone.append(new_f)
#     #         xn = tau_reco_sg_zone
#     #         taus_sg_zone.append(float(_[0]))
#     #     else:
#     #         fs_tau_sg_zone.append(grad_des_f)
#     #         taus_sg_zone.append(op_tau_new)
#     # fs_tau_sg_zone = correct_fs(fs_tau_sg_zone)


#     # ## BACKTRACKING TAU
#     # print('BACKTRACKING TAU')
#     # BT_net = BackTrackingTau(lst.A_adj, lst.A_func, reg_func, in_channels=4, out_channels=1).to(device)
#     # BT_net.load_state_dict(torch.load('BACKTRACKING_IMITATION_TAU.pth', map_location=device))

#     # from LGS_train_module import BackTrackingTauTrain
#     # net = BackTrackingTauTrain(lst.A_adj, lst.A_func, reg_func, in_channels=4, out_channels=1).to(device)
#     # net.load_state_dict(torch.load('BACKTRACKING_IMITATION_TAU.pth', map_location=device))
#     # new_tau = net(lst.x0, lst.x0, lst.y)

#     # BT_net.eval()

#     # x0_tensor = lst.x0[:, None, :, :]
#     # y_tensor = lst.y[None, None, :, :]

#     # # Now, you can use these 4D tensors as inputs to your network
#     # BT_reco, _, total_lst_BT = BT_net(lst.x0, lst.y, n_iter=100)
#     # fs_bt_tau = [lst.f(x, lst.y) for x in total_lst_BT]
#     # fs_bt_tau = correct_fs(fs_bt_tau)

#     # plt.semilogy([i for i in fs_tau_sg], label='SUPERVISED')
#     # plt.semilogy([i for i in fs_tau_Usg], label='UNSUPERVISED')
#     # plt.legend()
#     # plt.show()

#     ###### UNSUPERVISED PERFORMS BETTER!!!!!!!!!!!!!!!!!!
#     fs_learn2u

#         #plt.imshow(lst.y.squeeze(0).cpu().detach().numpy())
#     min_f = np.min([np.min(fs2), np.min(fs22), np.min(fs_bt)])
#     plt.semilogy([i - min_f for i in fs2], label='1/L')
#     plt.semilogy([i - min_f for i in fs22], label='Learned Fixed')
#     plt.semilogy([i - min_f for i in fs_learn2], label='Learned Correction Supervised')
#     plt.semilogy([i - min_f for i in fs_learn2u], label='Learned Correction Unsupervised')
#     plt.semilogy([i - min_f for i in fs_bt], label='Backtracking')
#     plt.semilogy([i - min_f for i in fs_tau], label='Learned Tau')
#     plt.semilogy([i - min_f for i in fs_tau_sg], label='Learned Tau Supervised Safeguarded')
#     plt.semilogy([i - min_f for i in fs_tau_Usg], label='Learned Tau Unsupervsied Safeguarded')
#     plt.legend()
#     plt.show()


#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
#     plt.xlabel('Iteration Number')  # Replace with your actual label
#     plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     plt.semilogy([i - min_f for i in [i for i in fs2 if i - min_f>1e-04]], label='1/L')
#     plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Fixed')
#     plt.legend()
#     plt.show()


#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
#     plt.xlabel('Iteration Number')  # Replace with your actual label
#     plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     plt.semilogy([i - min_f for i in [i for i in fs2 if i - min_f>1e-04]], label='1/L')
#     plt.semilogy([i - min_f for i in [i for i in fs2l if i - min_f>1e-04]], label='2/L')
#     plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Fixed')
#     plt.legend()
#     plt.show()




#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
#     plt.xlabel('Iteration Number')  # Replace with your actual label
#     plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     plt.semilogy([i - min_f for i in [i for i in fs2small if i - min_f>1e-04]], label='Large Step Size')
#     plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Fixed')
#     plt.semilogy([i - min_f for i in [i for i in fs2large if i - min_f>1e-04]], label='Small Step Size')
#     plt.legend()
#     plt.show()



#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title("Learned $\phi$ function vs Iteration Number")
#     plt.xlabel("Iteration Number")
#     plt.ylabel("$\phi$")
#     plt.scatter(range(len(taus_Usg)), taus_Usg, marker='.', color='red')
#     plt.grid(True)
#     plt.show()


#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title(r'semilogy plot of $f(x_k) - f(x^*)$')
#     plt.xlabel('Iteration Number')  # Replace with your actual label
#     plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Constant')
#     plt.semilogy([i - min_f for i in fs_tau_Usg], label='Learned Tau Function Safeguarded')
#     plt.legend()
#     plt.show()


#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title(r'semilogy plot of $f(x_k) - f(x^*)$')
#     plt.xlabel('Iteration Number')  # Replace with your actual label
#     plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     plt.semilogy([i - min_f for i in fs_tau], label='Learned Tau Function Safeguarded')
#     plt.legend()
#     plt.show()

#     fs_learn2u.insert(0, fs2[0])
#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title(r'semilogy plot of $f(x_k) - f(x^*)$')
#     plt.xlabel('Iteration Number')  # Replace with your actual label
#     plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     plt.semilogy([i - min_f for i in fs_learn2u[:100]], label='Learned Correction')
#     plt.semilogy([i - min_f for i in fs_tau_Usg[:100]], label='Learned Tau Function Safeguarded')
#     plt.legend()
#     plt.show()



#     plt.semilogy([i - min_f for i in fs2[:6]], label='1/L')
#     plt.semilogy([i - min_f for i in fs22[:6]], label='Learned Fixed')
#     plt.semilogy([i - min_f for i in fs_learn[:6]], label='Learned')
#     plt.semilogy([i - min_f for i in fs_learn2[:6]], label='Learned by me')
#     plt.semilogy([i - min_f for i in fs_bt[:6]], label='Backtracking')
#     plt.semilogy([i - min_f for i in fs_bt_tau[:6]], label='Backtracking Imitation')
#     plt.legend()
#     plt.show()



#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\tau$')
#     plt.xlabel('Iteration Number')  # Replace with your actual label
#     plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     plt.semilogy([i - min_f for i in fs_learn], label=r'Learned $\tau$, u', marker='s', markersize=5, linestyle='-')
#     plt.semilogy([i - min_f for i in fs_bt[:len(fs_bt)]], label=r'Backtracking Line Search', marker='o', markersize=5, linestyle='-')
#     plt.semilogy([i - min_f for i in fs22[:len(fs_bt)]], label=r'$\tau$ learned', marker='s', markersize=5, linestyle='-')
#     plt.legend()
#     plt.show()


#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\tau$')
#     plt.xlabel('Iteration Number')  # Replace with your actual label
#     plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
#     plt.semilogy([i - min_f for i in fs2], label=r'$\tau$ = 1/L', marker='o', markersize=5, linestyle='-')
#     plt.semilogy([i - min_f for i in fs22], label=r'$\tau$ learned', marker='s', markersize=5, linestyle='-')
#     plt.legend()
#     plt.show()

#     break

