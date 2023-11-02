from torch.utils.data import DataLoader
import numpy as np
import torch
from huber_TV import power_iteration
from datasets import NaturalDataset, TestDataset, ImageBlurDataset, my_collate
from optimisers import CNN_LSTM, CNN_LSTM_Full, CNN_LSTM_CORRECTION, AdagradModel, RMSPropModel, AdamModel, TauFuncNet
from algorithms import function_evals, gradient_descent_fixed,  gradient_descent_update, gradient_descent_modelfree, gradient_descent_correctionNN, adagrad, rmsprop, adam, gradient_descent_post_correction
from grad_x import grad, laplacian
import time
from torch.nn import HuberLoss
import torch.nn.functional as F

from functions import get_blurred_image
sig=2
alpha = 0.01
new_f = lambda x, y: f(x, y, lambda im: get_blurred_image(im, sig), alpha)
def get_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_x_star(f, x0, y):
    xs, taus = gradient_descent_fixed(f, x0.to(device), y.to(device), 0.5, tol=1e-03)
    x_star = xs[-1]
    return x_star

def get_recon_psnr(f, x0, y, img):
    x_star = get_x_star(f, x0, y)
    return get_psnr(x_star, img.to(device))

def get_f_from_alpha(alpha, A_func):
    return lambda x,y: data_fidelity(x,y,A_func) + alpha * huber_total_variation(x)

def get_recon_psnr_from_alpha(alpha, y,A_func, img):
    f = get_f_from_alpha(alpha, A_func)
    return get_recon_psnr(f, x0, y, img)

def get_recon_psnrs_from_alpha(alpha, images, x,y,A_func):
    f = get_f_from_alpha(alpha, x,y,A_func)
    return sum([get_recon_psnr(f, x0, y, img)] for img in images)


delta= 0.05
huber = HuberLoss(reduction='none',delta=delta)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_models = False
alpha = 0.005
NUM_IMAGES = 8
NUM_TEST = 10

noise_list = [0.]
sigma_list = list(np.linspace(2, 8, 6))
noise_list_test = [0.]
sigma_list_test = list(np.linspace(3, 10, 7))

num_batch = 10
wk_list = [1]
wk_list2 = [-1]

n_iters=50
num_iters=50

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


folder_path = 'Images_128/'
dataset = NaturalDataset(folder_path, num_imgs=NUM_IMAGES)
test_dataset = TestDataset(folder_path, num_imgs=NUM_TEST)

blurred_list = ImageBlurDataset(dataset, alpha, noise_list, sigma_list, f, grad_f)

train_loader = DataLoader(dataset=blurred_list, batch_size=num_batch, shuffle=True, num_workers=0,
                          collate_fn=my_collate)


for i, batch in enumerate(train_loader):
    img, y, x0, std, sig, epsilon, A_func, A_adj, f, grad_f = batch


import matplotlib.pyplot as plt

F.interpolate(img[1], scale_factor=4, mode='bilinear', align_corners=False)


img1 = F.interpolate(img[1].unsqueeze(0).unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
plt.imshow(img1, cmap='gray')


from PIL import Image
import torch
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image

# Define a transformation to convert the PIL image to a tensor
transform = transforms.ToTensor()
i1 = transform(Image.open(r'C:\Users\Patrick\PycharmProjects\TFR\notumor\Tr-no_0011.jpg').convert("L"))
i2 = transform(Image.open(r'C:\Users\Patrick\PycharmProjects\TFR\notumor\Tr-no_0021.jpg').convert("L"))
i3 = transform(Image.open(r'C:\Users\Patrick\PycharmProjects\TFR\notumor\Tr-no_0022.jpg').convert("L"))
i4 = transform(Image.open(r'C:\Users\Patrick\PycharmProjects\TFR\notumor\Tr-no_0050.jpg').convert("L"))
img1 = F.interpolate(i1.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
img2 = F.interpolate(i2.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
img3 = F.interpolate(i3.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
img4 = F.interpolate(i4.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.imshow(img3, cmap='gray')#
plt.axis('off')
plt.tight_layout()
plt.show()

plt.imshow(img4, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()









transform = transforms.ToTensor()
i1 = transform(Image.open(r'C:\Users\Patrick\PycharmProjects\TFR\lhq_256\0000168.png').convert("L"))
i2 = transform(Image.open(r'C:\Users\Patrick\PycharmProjects\TFR\lhq_256\0000169.png').convert("L"))
i3= transform(Image.open(r'C:\Users\Patrick\PycharmProjects\TFR\lhq_256\0000170.png').convert("L"))
i4 = transform(Image.open(r'C:\Users\Patrick\PycharmProjects\TFR\lhq_256\0000171.png').convert("L"))
img1 = F.interpolate(i1.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
img2 = F.interpolate(i2.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
img3 = F.interpolate(i3.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
img4 = F.interpolate(i4.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.imshow(img3, cmap='gray')#
plt.axis('off')
plt.tight_layout()
plt.show()

plt.imshow(img4, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()



im = img2
sig=1
new_im = get_blurred_image(im, sig)
plt.imshow(new_im, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()


std=0.1
epsilon = torch.normal(mean=0, std=std, size=(im.shape[0], im.shape[1]))
new_im = im + epsilon
plt.imshow(new_im, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()


