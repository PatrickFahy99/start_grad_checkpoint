import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import torch
import torch.nn.functional as F
from torch.fft import fftn, ifftn
from algorithms import gradient_descent_fixed
from datasets import TestDataset, XRayDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

folder_path = r'C:\Users\Patrick\XRayL2O\notumor'

test_dataset = TestDataset(folder_path, num_imgs=30)

std = 0.05

def data_fidelity(x,y,A_func):
    return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2)

def huber_total_variation(u):
    if len(u.shape) == 2:
        diff_x, diff_y = Du(u)
        norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2+1e-08)))
    elif len(u.shape) == 3:
        diff_x, diff_y = Du(u.squeeze(0))
        norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2+1e-08)))
    elif len(u.shape) == 4:
        diff_x, diff_y = Du(u.squeeze(0).squeeze(0))
        norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2+1e-08)))
    else:
        print('UH OH, WRONG SHAPE')
        return 'sdkfjsd'
    return norm_2_1

def huber(s, epsilon=0.01):
    return torch.where(torch.abs(s) <= epsilon, (0.5 * s ** 2) / epsilon, s - 0.5 * epsilon)

def Du(u):
    """Compute the discrete gradient of u using PyTorch."""
    diff_x = torch.diff(u, dim=1, prepend=u[:, :1])
    diff_y = torch.diff(u, dim=0, prepend=u[:1, :])
    return diff_x, diff_y


def f(x,y,A_func, alpha):
    return data_fidelity(x,y,A_func) + reg(x, alpha)

def reg(x, alpha):
    if len(x.shape) == 2:
        return alpha * huber_total_variation(x)
    elif len(x.shape) == 3:
        return alpha * huber_total_variation(x.squeeze(0))
    elif len(x.shape) == 4:
        return alpha * huber_total_variation(x.squeeze(0).squeeze(0))
    else:
        print('UH OH, WRONG SHAPE')


def gradient_descent_fixed(f, operator, adjoint, reg_func, x0, y, tau_in, tol=1e-12):
    tau = tau_in
    xo = x0.clone().detach().double()  # Convert to double precision
    xs = [xo]
    fs = [f(xo, y).detach().cpu().numpy()]
    num = 0
    taus = []

    new_f = lambda x: 0.5 * torch.norm(operator(x) - y).double()**2 + reg_func(x).double()  # Convert intermediate calculations to double

    xo = xo.requires_grad_(True)  # Set requires_grad to True
    reg_value = reg_func(xo).double()  # Convert intermediate calculations to double
    grad_reg_new = torch.autograd.grad(reg_value, xo)[0].double()  # Convert gradients to double

    grad_fit = adjoint(operator(xo) - y).double()  # Convert intermediate calculations to double
    grad_f_new = grad_fit + grad_reg_new

    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients

    scale_tau = False


    while torch.norm(grad_f_new) > tol:

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

        if num > 1000:
            return xs, taus, fs
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
            xs.append(xn)
            fs.append(f(xn, y).detach().cpu().numpy())
            taus.append(tau)
            xo = xn.clone().detach()
            num+=1
    return xs, taus, fs


# def get_blurred_image(img, s=1):
#     device = img.device  # Get the device of the input
#     n = img.shape[0]
#     x = torch.hstack((torch.arange(0, n // 2), torch.arange(-n // 2, 0))).to(device)  # Move tensor to the correct device
#     [Y, X] = torch.meshgrid(x, x)
#     h = torch.exp((-X ** 2 - Y ** 2) / (2 * s ** 2))
#     h = h / torch.sum(h)
#     Fh = fftn(h)
#     Fu = fftn(img)
#     out = ifftn(Fh * Fu)
#     return out.real

# new_A_func = lambda im: get_blurred_image(im, 4)

# epsilon = torch.normal(mean=0, std=5/255, size=(img.shape[0], img.shape[1]))

# y = new_A_func(img) + epsilon


# Define regularization operator L (e.g., finite differences)

# Range of regularization strengths to evaluate
alphas = [0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.0075, 0.001, 0.002, 0.003]

# Lists to store logarithmic values
regularization_terms = []
data_fit_terms = []

# Evaluate L-curve
for alpha in alphas:

    reg_func = lambda x: reg(x, alpha)

    print(alpha)

    test_set = XRayDataset(test_dataset, alpha, f, std, 'limited')
    num=0
    for lst in test_set:

        if num<20:
            num+=1
            continue


        op_tau_new = 1 / (lst.operator_norm**2 + 8 * alpha / 0.01)

        xs, taus, fs_exact = gradient_descent_fixed(lambda x,y: f(x, y, lst.A_func, alpha), lst.A_func, lst.A_adj, reg_func, lst.x0.double(), lst.y.double(), op_tau_new)

        data_fit_term = data_fidelity(xs[-1].unsqueeze(0), lst.y.to(device), lst.A_func)
        regularization_term = huber_total_variation(xs[-1])

        regularization_terms.append(float(regularization_term))
        data_fit_terms.append(float(data_fit_term))

        break

# Create L-curve plot
plt.scatter(data_fit_terms, regularization_terms, marker='o')
plt.scatter([float(data_fidelity(lst.img.unsqueeze(0).to(device),lst.y.to(device), lst.A_func))], [float(huber_total_variation(lst.img))], marker='o', color='red')
plt.xlabel("Data Fit Term")
plt.ylabel("Regularisation Term")
plt.title("L-Curve")
plt.grid(True)

# Identify balance point (usually visually)
# For this example, you can manually identify the corner point on the plot

plt.show()
