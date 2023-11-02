

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


def huber(s, epsilon=0.01):
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


def huber_total_variation(u, eps=0.01):
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






def gradient_descent_fixed_iter(operator, reg_func, x0, y, tau_in, iters=1):
    xo = x0.clone().detach().double()  # Convert to double precision

    new_f = lambda x: 0.5 * torch.norm(operator(x) - y).double()**2 + reg_func(x).double()  # Convert intermediate calculations to double

    xo = xo.requires_grad_(True)  # Set requires_grad to True
    f_value = new_f(xo).double()  # Convert intermediate calculations to double
    grad_f_new = torch.autograd.grad(f_value, xo)[0].double()  # Convert gradients to double

    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients

    i=0
    while i < iters:
        i+=1
        torch.cuda.empty_cache()

        xn = xo.to(device) - tau_in * grad_f_new.to(device)

        xn.requires_grad_(True)
        f_value = new_f(xn)

        f_value = new_f(xn).double()  # Convert intermediate calculations to double
        grad_f_new = torch.autograd.grad(f_value, xn)[0].double()
        
    return xn  



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
    




import odl
import torch
import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from LGS_train_module import get_images, TauModelNoAdjoint
import matplotlib.pyplot as plt

### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def huber(s, epsilon=0.01):
    return torch.where(torch.abs(s) <= epsilon, (0.5 * s ** 2) / epsilon, s - 0.5 * epsilon)

def Du(u):
    """Compute the discrete gradient of u using PyTorch."""
    diff_x = torch.diff(u, dim=1, prepend=u[:, :1])
    diff_y = torch.diff(u, dim=0, prepend=u[:1, :])
    return diff_x, diff_y

def huber_total_variation(u, eps=0.01):
    diff_x, diff_y = Du(u)
    norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2+1e-08)))
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

alpha = 0.0002

reg_func = lambda x: reg(x, alpha)

def f(x, y, operator, alpha):
    return 0.5 * (torch.norm(operator(x) - y, p=2) ** 2) + reg(x, alpha)


# sourcery skip: for-index-underscore, remove-redundant-fstring
from LGS_train_module import get_images

n_images = 10001
#images = get_images(r'C:\Users\Patrick\XRayL2O\notumor', n_images, scale_number=2)
images = get_images(r'C:\Users\Patrick\XRayL2O\lhq_256', n_images, scale_number=2)
images = np.array(images, dtype='float32')
images = torch.from_numpy(images).float().to(device)



std = 0.05

from datasets import Blur2Dataset, ImageDataBlur2

blur_level = 10
blur_size = 7
# training_set = Blur2Dataset(images, alpha, std, blur_level, blur_size)

from torchvision.transforms import GaussianBlur
model = GaussianBlur(blur_size, blur_level).to(device)
op_norm = 1#power_iteration(model)

imgs = []
ys = []
for img in images:
    y = model(img.unsqueeze(0)).squeeze(0)
    noise = torch.tensor(np.random.normal(0, y.cpu().numpy().std(), y.shape) * std).to(device)
    y = torch.tensor(y+noise).to(device)
    imgs.append(img.unsqueeze(0))
    ys.append(y)


tau_network =  TauModelNoAdjoint(model, lambda x: reg(x, alpha), in_channels=4, out_channels=1).to(device)




print('FIXED LEARNED')
start_time = time.time()
xs22, fs22 = gradient_descent_fixed_odl_blur(model, 0*img.unsqueeze(0), y.unsqueeze(0), 1.0125, reg, alpha)
plt.imshow(torch.tensor(xs22[-1]).squeeze(0).cpu().numpy(), cmap='gray')
end_time = time.time()
execution_time = end_time - start_time
execution_times_nonlearn1.append(execution_time)
fs22 = correct_fs(fs22)



from LGS_train_module import AlgoTauModelNoAdjoint, AlgoTauModelNoAdjointSmaller, AlgoTauModelNoAdjointX
print('LEARNED TAU')
tau_net_checkpoint = AlgoTauModelNoAdjoint(model, reg_func, in_channels=4, out_channels=1).to(device)
path1 = 'models/BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_1_200_1000_DIFF.pth'
path2 = 'models/BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_1_138_1000_ALL.pth'
path3 = 'models/BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_100_4_1000_ALL.pth'
# tau_net_checkpoint.load_state_dict(torch.load(path2, map_location=device))
# tau_net = AlgoTauModelNoAdjoint(model, reg_func, in_channels=4, out_channels=1).to(device)
# tau_net.load_state_dict(torch.load('models/BLURRING_TAU_UNSUPERVISED.pth', map_location=device))

# f_list_tau_cp, taus_list_cp = tau_net_checkpoint(0*img.unsqueeze(0), y.unsqueeze(0), n_iter=500)
# fs_tau_cp = correct_fs(f_list_tau_cp)
# taus_U_cp = [float(t) for t in taus_list_cp]

# #fs_tau_cp.insert(0, fs22[0])

# f_list_tau, taus_list1 = tau_net(0*img.unsqueeze(0), y.unsqueeze(0), n_iter=500)
# fs_tau = correct_fs(f_list_tau)
# taus_U = [float(t) for t in taus_list1]
# #fs_tau.insert(0, fs22[0])


tau_net_checkpoint_x = AlgoTauModelNoAdjointX(model, reg_func, in_channels=4, out_channels=1).to(device)
tau_net_checkpoint_x.load_state_dict(torch.load(path2, map_location=device))
tau_net_x = AlgoTauModelNoAdjointX(model, reg_func, in_channels=4, out_channels=1).to(device)
tau_net_x.load_state_dict(torch.load('models/BLURRING_TAU_UNSUPERVISED.pth', map_location=device))

max_iter = 5000

fs_tau_cp = [fs22[0]]
taus_cp = []
xn = 0*img.unsqueeze(0)
for _ in range(max_iter):
    new_f, tau_new, tau_reco_sg = tau_net_checkpoint_x(xn, y.unsqueeze(0), n_iter=1)
    fs_tau_cp.append(new_f[-1])
    xn = tau_reco_sg
    taus_cp.append(float(tau_new[0]))
fs_tau_cp = correct_fs(fs_tau_cp)

fs_tau = [fs22[0]]
taus = []
xn = 0*img.unsqueeze(0)
for _ in range(max_iter):
    new_f, tau_new, tau_reco_sg = tau_net_x(xn, y.unsqueeze(0), n_iter=1)
    fs_tau.append(new_f[-1])
    xn = tau_reco_sg
    taus.append(float(tau_new[0]))
fs_tau = correct_fs(fs_tau)



print('SAFEGUARDED TAU MODEL')

### PROVABLY CONVERGENT IF THE F IS AT MOST WHAT GRADIENT DESCENT GIVES, BUT CHECKING AT EACH ITERATION IS COSTLY? MAYBE NOT
fs_tau_sg = [fs22[0]]
taus_sg = []
xn = 0*img.unsqueeze(0)
for _ in range(max_iter):
    new_f, tau_new, tau_reco_sg = tau_net_checkpoint_x(xn, y.unsqueeze(0), n_iter=1)
    new_f = new_f[-1]
    grad_des_f =  float(f(xn, y.unsqueeze(0), model, alpha))
    xn = gradient_descent_fixed_iter(model, reg_func,  xn, y.unsqueeze(0), 1, 1)
    if new_f < grad_des_f:
        fs_tau_sg.append(new_f)
        xn = tau_reco_sg
        taus_sg.append(float(tau_new[0]))
    else:
        fs_tau_sg.append(grad_des_f)
        taus_sg.append(1)
fs_tau_sg = correct_fs(fs_tau_sg)


n_iter = 10
n_iter2 = 200
max_iter = 5000
#fs_tau[:20] = fs_tau_cp[:20]
fig, ax = plt.subplots()
ax.semilogy([i-min(fs22) for i in fs_tau_cp[:max_iter]], label='Learned Function (Greedy)')
ax.semilogy([i-min(fs22) for i in fs_tau[:max_iter]], label='Learned Function')
ax.semilogy([i-min(fs22) for i in fs22[:max_iter]], label='Learned Constant', color='red')
ax.semilogy([i-min(fs22) for i in fs_tau_sg[:max_iter]], label='Learned Function (Safeguarded)')
ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
ax.axvspan(n_iter, n_iter2, facecolor='yellow', alpha=0.5)
ax.axvspan(n_iter2, max_iter, facecolor='orange', alpha=0.5)
ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$. learned function greedy vs learned function vs learned constant')
ax.set_xlabel('Iteration Number')  # Replace with your actual label
ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
ax.legend()
plt.show()

## sg is not necessarily lower than cp becasue 


dfgdfgdfg

#### BLUR


for lst in test_set:

    num += 1


    # num+=1
    if num<6:
        continue

    tau_opt = 1/(lst.operator_norm**2 + 8*alpha/0.01)
    print('OPTIMAL TAU', tau_opt)
    print('FIXED NOT LEARNED 1')
    start_time = time.time()
    #xs, fs = gradient_descent_fixed_iters_odl_blur(lst.operator, lst.x0, lst.y, 1.0125, reg, alpha, 10)
    xs, fs = gradient_descent_fixed_odl_blur(lst.operator, lst.x0, lst.y, tau_opt, reg, alpha)
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
    xs22, fs22 = gradient_descent_fixed_odl_blur(lst.operator, lst.x0, lst.y, 1.0125, reg, alpha)
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


    # print('EXACT')
    # xse, taus, fs_exact = gradient_descent_fixed_odl_blur(lst.operator, lst.x0, lst.y, 0.6309675766619465, reg, alpha)
    # torch.cuda.empty_cache()
    # plt.imshow(xs[-1].squeeze(0), cmap='gray')
    # plt.show()
    # plt.imshow(lst.img.squeeze(0), cmap='gray')

    # print(fs_exact[-1])

    # xs_b, fs_b, regs = gradient_descent_fixed_iters_odl(lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 0.198,1000)                                                                     


    # dfgdfg

    # print('IMAGE REG, DATA', 'REG', reg_func(lst.img), 'FIT', data_fidelity(lst.img.unsqueeze(0).to(device), lst.y, lst.A_func), 'TOTAL', lst.f(lst.img.unsqueeze(0).to(device), lst.y))
    # print('APPROXIMATION', 'REG', reg_func(xs[-1]), 'FIT', data_fidelity(xs[-1].unsqueeze(0).to(device), lst.y, lst.A_func), 'TOTAL', lst.f(xs[-1].unsqueeze(0).to(device), lst.y))    #sdfsdfsdf

    # #xs, fs, regs = gradient_descent_fixed_iters_odl(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), 0.0987868702216, 200)

    # plt.imshow(torch.tensor(xs[10]).squeeze(0).cpu().numpy(), cmap='gray')
    # # ## 0.105 is divergent


    # # plt.semilogy([i-min(fs_exact) for i in fs2], label='LEARNED')

    # n_iter = 1000
    # fig, ax = plt.subplots()
    # ax.semilogy([i-min(fs) for i in fs22 if i>=min(fs)], label='Learned Fixed')
    # ax.semilogy([i-min(fs) for i in fs], label='1/L', color='red')
    # ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
    # ax.axvspan(n_iter, len(fs), facecolor='yellow', alpha=0.5)
    # ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
    # ax.set_xlabel('Iteration Number')  # Replace with your actual label
    # ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    # ax.legend()
    # plt.show()

    # n_iter = 1000
    # fig, ax = plt.subplots()
    # ax.semilogy([i-min(fs_exact) for i in fs22 if i>=min(fs)], label='Learned Fixed')
    # ax.semilogy([i-min(fs_exact) for i in fs], label='1/L', color='red')
    # ax.semilogy([i-min(fs_exact) for i in fs2 if i>=min(fs)], label='2/L', color='black')
    # ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
    # ax.axvspan(n_iter, len(fs), facecolor='yellow', alpha=0.5)
    # ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs 2/L vs learned $\phi$')
    # ax.set_xlabel('Iteration Number')  # Replace with your actual label
    # ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    # ax.legend()
    # plt.show()


    # n_iter = 1000
    # fig, ax = plt.subplots()
    # ax.semilogy([i-min(fs_exact) for i in fs22 if i>min(fs)], label='Learned Fixed')
    # ax.semilogy([i-min(fs_exact) for i in fs_b], label='$\phi = 0.198$', color='red')
    # ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
    # ax.axvspan(n_iter, len([i for i in fs22 if i>min(fs)]), facecolor='yellow', alpha=0.5)
    # ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$.  learned $\phi$ vs $\phi = 0.198$')
    # ax.set_xlabel('Iteration Number')  # Replace with your actual label
    # ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    # ax.legend()
    # plt.show()


    # LEARNED TAU
    from LGS_train_module import AlgoTauModelNoAdjoint, AlgoTauModelNoAdjointSmaller
    print('LEARNED TAU')
    iteration_number = 1
    train_iters_one = 1000
    num_mult_fives = 100
    tau_net = AlgoTauModelNoAdjointSmaller(lst.operator, reg_func, in_channels=4, out_channels=1).to(device)
    #tau_net.load_state_dict(torch.load('TAU_UNSUPERVISED.pth', map_location=device))
    tau_net.load_state_dict(torch.load('models/BLURRING_TAU_UNSUPERVISED.pth', map_location=device))

    tau_net_checkpoint = AlgoTauModelNoAdjoint(lst.operator, reg_func, in_channels=4, out_channels=1).to(device)
    tau_net_checkpoint.load_state_dict(torch.load('BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_1_100_1000.pth', map_location=device))

    f_list_tau, taus_list1 = tau_net(lst.x0, lst.y, n_iter=200)
    # plt.imshow(tau_reco.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='gray')
    # plt.imshow(lst.img.squeeze(0).cpu().detach().numpy(), cmap='gray')
    #fs_tau = [f(torch.tensor(x).to(device), lst.y, lst.operator, alpha) for x in total_lst]
    fs_tau = correct_fs(f_list_tau)
    taus_U = [float(t) for t in taus_list1]
    fs_tau.insert(0, fs[0])

    f_list_tau_cp, taus_list_cp = tau_net_checkpoint(lst.x0, lst.y, n_iter=200)
    # plt.imshow(tau_reco_cp.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='gray')
    # plt.imshow(lst.img.squeeze(0).cpu().detach().numpy(), cmap='gray')
    #fs_tau_cp = [f(torch.tensor(x).to(device), lst.y, lst.operator, alpha) for x in total_lst_cp]
    fs_tau_cp = correct_fs(f_list_tau_cp)
    taus_U_cp = [float(t) for t in taus_list_cp]
    fs_tau_cp.insert(0, fs[0])




    n_iter = 10
    fig, ax = plt.subplots()
    ax.semilogy([i-min(fs) for i in fs_tau[:2000]], label='Learned Function')
    ax.semilogy([i-min(fs) for i in fs_tau_cp[:2000]], label='Learned Function (Greedy)')
    ax.semilogy([i-min(fs) for i in fs[:2000]], label='1/L', color='black')
    ax.semilogy([i-min(fs) for i in fs22[:2000]], label='Learned Constant', color='red')
    ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
    ax.axvspan(n_iter, 2000, facecolor='yellow', alpha=0.5)
    ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$.  learned function vs learned constant vs 1/L')
    ax.set_xlabel('Iteration Number')  # Replace with your actual label
    ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    ax.legend()
    plt.show()

    n_iter = 10
    n_iter2 = 100
    fig, ax = plt.subplots()
    ax.semilogy([i-min(fs) for i in fs_tau[:2000]], label='Learned Step Size Function (Unrolling)')
    ax.semilogy([i-min(fs) for i in fs_tau_cp[:2000]], label='Learned Step Size Function (Greedy)')
    ax.axvspan(0, n_iter, facecolor='green', alpha=0.5)
    ax.axvspan(n_iter, n_iter2, facecolor='orange', alpha=0.5)
    ax.axvspan(n_iter, 200, facecolor='yellow', alpha=0.5)
    ax.set_title(r'semilogy plot of $f(x_k) - f(x^*)$.  Greedy vs Full Unrolled')
    ax.set_xlabel('Iteration Number')  # Replace with your actual label
    ax.set_ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    ax.legend()
    plt.show()

    dfgdfg


    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title("Learned $\phi$ function vs Iteration Number")
    plt.xlabel("Iteration Number")
    plt.ylabel("$\phi$")
    plt.scatter(range(len(taus_U)), taus_U, marker='.', color='red')
    plt.axhline(y=0.0852, color='blue', linestyle='--', label='1/L')
    plt.grid(True)
    plt.legend()
    plt.show()



    fghfgh

    print('SAFEGUARDED TAU MODEL')
    ### PROVABLY CONVERGENT IF THE F IS AT MOST WHAT GRADIENT DESCENT GIVES, BUT CHECKING AT EACH ITERATION IS COSTLY? MAYBE NOT
    fs_tau_sg = [lst.f(lst.x0, lst.y)]
    taus_sg = []
    xn = x0_tensor
    for _ in range(400):
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
    for _ in range(400):
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

