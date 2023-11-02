
from PIL import Image
import os
import random
import numpy as np
from torch.utils.data import Dataset
import torch
from functions import get_blurred_image, get_adj_image
import torch.nn.functional as F
from algorithms import gradient_descent_fixed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from scipy.ndimage import rotate

import cv2 as cv
import torch
import odl
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from odl.contrib.torch import OperatorModule

#
# def radon(image):
#     image_size = 256
#
#     # Number of projection angles
#     num_angles = 256
#     image = image.detach().cpu().numpy()
#     # Initialize the sinogram
#     sinogram = np.zeros((num_angles, image_size))
#
#     # Generate projection angles
#     angles = np.linspace(0, num_angles, num_angles, endpoint=False)
#
#     # Perform Radon transform
#     for i, angle in enumerate(angles):
#         rotated_image = rotate(image, angle, reshape=False)
#         sinogram[i, :] = np.sum(rotated_image, axis=0)
#     return torch.tensor(sinogram)
# def backprojection(sinogram):
#     # Get sinogram dimensions
#     num_angles, image_size = sinogram.shape
#
#     # Initialize the reconstructed image
#     reconstructed_image = np.zeros((image_size, image_size))
#
#     # Generate projection angles
#     angles = np.linspace(0, 256, num_angles, endpoint=False)
#
#     sinogram = sinogram.cpu().numpy()
#
#     # Perform backprojection
#     for i, angle in enumerate(angles):
#         rotated_projection = rotate(sinogram[i, np.newaxis], -angle, reshape=False)
#         reconstructed_image += np.tile(rotated_projection, (image_size, 1))
#
#     return torch.tensor(reconstructed_image)

from LGS_train_module import geometry_and_ray_trafo








class NaturalDataset(Dataset):
    def __init__(self, folder_path, num_imgs=False, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_list = []
        self.num_imgs = num_imgs

        num = 0
        for filename in os.listdir(self.folder_path):
            num += 1
            if num_imgs:
                if num > num_imgs:
                    break

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.folder_path, filename)
                image = Image.open(image_path).convert("L")
                image_tensor = torch.tensor(np.array(image)) / np.max(np.array(image))  # normalize the pixel values
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
                image_tensor = F.interpolate(image_tensor, size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                self.image_list.append(image_tensor)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.image_list[idx]

        if self.transform:
            img = self.transform(img)

        return img



class ImageData:
    def __init__(self, img, y, x0, std, sig, epsilon, new_A_func, new_A_adj, new_f, new_grad_f):#, f_x_star):
        self.img = img
        self.y = y
        self.x0 = x0
        self.std = std
        self.sig = sig
        self.epsilon = epsilon
        self.A_func = new_A_func
        self.A_adj = new_A_adj
        self.f = new_f
        self.grad_f = new_grad_f
        #self.f_x_star = f_x_star

class ImageDataNonsmooth:
    def __init__(self, img, y, x0, std, sig, epsilon, new_A_func, new_A_adj, new_f, new_data_fit, new_grad_data_fit, new_reg):
        self.img = img
        self.y = y
        self.x0 = x0
        self.std = std
        self.sig = sig
        self.epsilon = epsilon
        self.A_func = new_A_func
        self.A_adj = new_A_adj
        self.f = new_f
        self.data_fit = new_data_fit
        self.grad_data_fit = new_grad_data_fit
        self.reg = new_reg


class ImageBlurDataset(Dataset):
    def __init__(self, image_dataset, alpha, noise_list, sigma_list, f, grad_f):
        self.image_dataset = image_dataset
        self.alpha = alpha
        self.noise_list = noise_list
        self.sigma_list = sigma_list
        self.f = f
        self.grad_f = grad_f

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.image_dataset[idx]

        std = torch.tensor(random.choice(self.noise_list)).float()
        sig = torch.tensor(random.choice(self.sigma_list)).float()

        epsilon = torch.normal(mean=0, std=std, size=(img.shape[0], img.shape[1]))

        new_A_func = lambda im: get_blurred_image(im, sig)
        new_A_adj = lambda im: get_adj_image(im, sig)

        new_A_func = lambda im: get_blurred_image(im, sig)
        new_A_adj = lambda im: get_adj_image(im, sig)

        #print(img.shape)
        #print(new_A_func(img).shape)
        y = new_A_func(img) + epsilon
        x0 = new_A_adj(y)

        new_f = lambda x, y: self.f(x, y, new_A_func, self.alpha)
        new_grad_f = lambda x, y: self.grad_f(x, y, new_A_func, new_A_adj, self.alpha)

        # xs, taus = gradient_descent_fixed(new_f, x0.to(device), y.to(device), 0.5, tol=1e-06)
        # x_star = xs[-1]
        # f_x_star = torch.tensor(new_f(x_star.to(device), y.to(device))).float()
        # print('calcd min')

        return ImageData(img, y, x0, std, sig, epsilon, new_A_func, new_A_adj, new_f, new_grad_f)#, f_x_star)



class ImageDataXRay:
    def __init__(self, img, y, x0, std, domain, geometry, shape, sinograms, new_A_func, new_A_adj, new_f, pseudo_inv, operator_norm):#, f_x_star):
        self.img = img
        self.y = y
        self.x0 = x0
        self.std = std
        self.domain = domain
        self.geometry = geometry
        self.shape = shape
        self.sinograms = sinograms
        self.A_func = new_A_func
        self.A_adj = new_A_adj
        self.f = new_f
        self.pseudo_inv = pseudo_inv
        self.operator_norm = operator_norm

class XRayDataset(Dataset):
    def __init__(self, image_dataset, alpha, f, std, sampling = 'full'):
        self.img = image_dataset
        self.alpha = alpha
        self.std = [std for _ in range(len(image_dataset))]

        shape = (256,256)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup=sampling, shape=shape, device=device, factor_lines = 1)
        fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

        operator_norm = odl.power_method_opnorm(ray_transform, atol=0, rtol=0, maxiter=10000)
        print(operator_norm)

        ### Using odl functions to make odl operators into PyTorch modules
        ray_transform_module = OperatorModule(ray_transform).to(device)
        adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
        fbp_operator_module = OperatorModule(fbp_operator).to(device)

        lst = [im for  im in self.img]
        images = np.array(lst, dtype='float32')
        images = torch.from_numpy(images).float().to(device)
        self.lst = lst

        ### Making sinograms from the images using Radon transform module
        sinograms = ray_transform_module(images) #.cpu().detach().numpy()

        ### Allocating used tensors
        noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape).cpu().detach().numpy()
        rec_images = torch.zeros((sinograms.shape[0], ) + shape)

        ### Adding Gaussian noise to the sinograms.
        for k in range(len(image_dataset)):
            sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
            noise = np.random.normal(0, sinogram_k.std(), sinogram_k.shape) * self.std[k]
            noisy_sinograms[k,:,:] = sinogram_k + noise

        noisy_sinograms = np.array(noisy_sinograms, dtype='float32')
        noisy_sinograms = torch.from_numpy(noisy_sinograms).float().to(device)

        rec_images = 0*images#fbp_operator_module(noisy_sinograms)
        #rec_images = rec_images[:,None,:,:]

        self.y = noisy_sinograms
        self.x0 = rec_images
        self.domain = domain
        self.geometry = geometry
        self.shape = output_shape ## 360, 1028
        self.sinograms = sinograms
        self.new_A_func = ray_transform_module
        self.new_A_adj = adjoint_operator_module
        self.new_f = lambda x, y: f(x, y, ray_transform_module, self.alpha)
        self.pseudo_inv = fbp_operator_module
        self.operator_norm = operator_norm

        print(self.alpha)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return ImageDataXRay(self.lst[idx], self.y[idx].unsqueeze(0), self.x0[idx].unsqueeze(0), self.std[idx], self.domain, self.geometry, self.shape, self.sinograms[idx], self.new_A_func, self.new_A_adj, self.new_f, self.pseudo_inv, self.operator_norm)



def power_iteration(func, num_iterations=1000, N=256):
    u = torch.rand(1,N,N)
    for _ in range(num_iterations):
        v = func(u)
        norm_v = torch.linalg.norm(v.view(-1))
        u = v / norm_v
    operator_norm = torch.linalg.norm(func(u).view(-1)) / torch.linalg.norm(u.view(-1))
    return operator_norm.item()


from torchvision.transforms import GaussianBlur
class ImageDataBlur2:
    def __init__(self, img, y, x0, operator, op_norm):#, f_x_star):
        self.img = img
        self.y = y
        self.x0 = x0
        self.operator = operator
        self.operator_norm = op_norm

class Blur2Dataset(Dataset):
    def __init__(self, image_dataset, alpha, std, blur_level, blur_size):

        self.std = std
        self.blur_level = blur_level
        self.image_dataset = image_dataset
        self.alpha = alpha
        self.blur_size = blur_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        img = self.image_dataset[idx].unsqueeze(0)
        operator = GaussianBlur(self.blur_size, self.blur_level).to(device)
        op_norm = power_iteration(operator)
        y = operator(img).to(device)
        noise = torch.tensor(np.random.normal(0, y.cpu().numpy().std(), y.shape) * self.std).to(device)
        y = torch.tensor(y+noise).to(device)
        x0 = 0*img


        return ImageDataBlur2(img, y, x0, operator, op_norm)





def gradient_descent_fixed(f, x0, y, tau_in, tol=01e-06):
    tau = tau_in
    xo = x0.clone().detach()
    xs = [xo]
    fs = [f(xo, y).detach().cpu().numpy()]
    taus = []
    grad_fs = []
    num = 1
    scale_num = 0
    res = 0
    new_f = lambda x: f(x,y)
    scale_nums = []
    scale_tau=False

    xo = xo.requires_grad_(True)  # Set requires_grad to True
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]
    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients

    while torch.norm(grad_f_new) > tol:
        #print(torch.norm(grad_f_new))

        go = torch.norm(grad_f_new)
        if not scale_tau:
            tau = tau_in#.to(device)
        xn = xo.to(device) - tau * grad_f_new.to(device)

        xn.requires_grad_(True)
        f_value = new_f(xn)
        grad_f_new = torch.autograd.grad(f_value, xn)[0]  # Compute gradients explicitly
        xn = xn.clone().detach()

        gn = torch.norm(grad_f_new)
        if go == gn:
            print('ITERS:', num)
            print(scale_num)
            return xn, taus, fs, num, scale_nums
        if f(xn,y)>f(xo,y):
            res += 1
            tau *= 0.9
            if not scale_tau:
                scale_num += 1
                scale_nums.append(num)
            scale_tau = True
            if res>1000:
                print('too many reductions')
                print(scale_num)
                return xn, taus, fs, num, scale_nums
        else:
            scale_tau = False
            res = 0
            xs.append(xn)
            xs = xs[-2:]
            fs.append(f(xn, y).detach().cpu().numpy())
            taus.append(tau)
            xo = xn.clone().detach()
            num += 1
    return xn, taus, fs, num, scale_nums



class ImageDataXRay:
    def __init__(self, img, y, x0, std, domain, geometry, shape, sinograms, new_A_func, new_A_adj, new_f, pseudo_inv, operator_norm):#, x_star, fx_star):#, f_x_star):
        self.img = img
        self.y = y
        self.x0 = x0
        self.std = std
        self.domain = domain
        self.geometry = geometry
        self.shape = shape
        self.sinograms = sinograms
        self.A_func = new_A_func
        self.A_adj = new_A_adj
        self.f = new_f
        self.pseudo_inv = pseudo_inv
        self.operator_norm = operator_norm
        #self.x_star = x_star
        #self.fx_star = fx_star

class XRayDatasetSupervised(Dataset):
    def __init__(self, image_dataset, alpha, f):
        self.img = image_dataset
        self.alpha = alpha
        self.std = [0.01 for _ in range(len(image_dataset))]

        shape = (256,256)

        domain, geometry, ray_transform, output_shape = get_ray_transform(shape=shape)
        fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

        operator_norm = odl.power_method_opnorm(ray_transform)

        ### Using odl functions to make odl operators into PyTorch modules
        ray_transform_module = OperatorModule(ray_transform).to(device)
        adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
        fbp_operator_module = OperatorModule(fbp_operator).to(device)

        lst = [im for  im in self.img]
        images = np.array(lst, dtype='float32')
        images = torch.from_numpy(images).float().to(device)
        self.lst = lst

        ### Making sinograms from the images using Radon transform module
        sinograms = ray_transform_module(images) #.cpu().detach().numpy()

        ### Allocating used tensors
        noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape).cpu().detach().numpy()
        rec_images = torch.zeros((sinograms.shape[0], ) + shape)

        ### Adding Gaussian noise to the sinograms.
        for k in range(len(image_dataset)):
            sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
            noise = np.random.normal(0, sinogram_k.std(), sinogram_k.shape) * self.std[k]
            noisy_sinograms[k,:,:] = sinogram_k + noise

        noisy_sinograms = np.array(noisy_sinograms, dtype='float32')
        noisy_sinograms = torch.from_numpy(noisy_sinograms).float().to(device)

        rec_images = fbp_operator_module(noisy_sinograms)
        rec_images = rec_images[:,None,:,:]

        self.y = noisy_sinograms
        self.x0 = rec_images
        self.domain = domain
        self.geometry = geometry
        self.shape = output_shape ## 360, 1028
        self.sinograms = sinograms
        self.new_A_func = ray_transform_module
        self.new_A_adj = adjoint_operator_module
        self.new_f = lambda x, y: f(x, y, ray_transform_module, self.alpha)
        self.pseudo_inv = fbp_operator_module
        self.operator_norm = operator_norm


        print('get_mins')
        self.x_star = []
        for i in range(self.x0.shape[0]):
            print(i)
            self.x_star.append(gradient_descent_fixed(f, self.x0[i,:,:,:], self.y[i,:,:,:], tau_in=0.05))
        print('get_f_mins')
        self.fx_star = [self.new_f(self.x_star[k], self.y[k]) for k in range(len(image_dataset))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return ImageDataXRay(self.lst[idx], self.y[idx], self.x0[idx], self.std[idx], self.domain, self.geometry, self.shape, self.sinograms[idx], self.new_A_func, self.new_A_adj, self.new_f, self.pseudo_inv, self.operator_norm, self.x_star[idx], self.fx_star[idx])



class ImageBlurDatasetNonsmooth(Dataset):
    def __init__(self, image_dataset, alpha, noise_list, sigma_list, f, data_fit, grad_data_fit, reg):
        self.image_dataset = image_dataset
        self.alpha = alpha
        self.noise_list = noise_list
        self.sigma_list = sigma_list
        self.f = f
        self.data_fit = data_fit
        self.grad_data_fit = grad_data_fit
        self.reg = reg

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.image_dataset[idx]

        std = torch.tensor(random.choice(self.noise_list)).float()
        sig = torch.tensor(random.choice(self.sigma_list)).float()

        epsilon = torch.normal(mean=0, std=std, size=(img.shape[0], img.shape[1]))

        new_A_func = lambda im: get_blurred_image(im, sig)
        new_A_adj = lambda im: get_adj_image(im, sig)

        y = new_A_func(img) + epsilon
        x0 = new_A_adj(y)

        new_f = lambda x, y: self.f(x, y, new_A_func, self.alpha)
        new_data_fit = lambda x, y: self.data_fit(x, y, new_A_func)
        new_grad_data_fit = lambda x, y: self.grad_data_fit(x, y, new_A_func, new_A_adj)
        new_reg = lambda x: self.reg(x, self.alpha)

        return ImageDataNonsmooth(img, y, x0, std, sig, epsilon, new_A_func, new_A_adj, new_f, new_data_fit, new_grad_data_fit, new_reg)



class TestDataset(Dataset):
    def __init__(self, folder_path, num_imgs=False, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_list = []
        self.num_imgs = num_imgs

        num = 0
        for filename in os.listdir(self.folder_path):
            num += 1
            if num_imgs:
                if num > num_imgs:
                    break

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.folder_path, filename)
                image = Image.open(image_path).convert("L")
                image_tensor = torch.tensor(np.array(image)) / np.max(np.array(image))  # normalize the pixel values
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
                image_tensor = F.interpolate(image_tensor, size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                self.image_list.append(image_tensor)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.image_list[idx]

        if self.transform:
            img = self.transform(img)

        return img


def my_collate(batch):
    img = torch.stack([item.img for item in batch])
    y = torch.stack([item.y for item in batch])
    x0 = torch.stack([item.x0 for item in batch])
    std = torch.stack([torch.tensor(item.std) for item in batch])
    sig = torch.stack([torch.tensor(item.sig) for item in batch])
    epsilon = torch.stack([item.epsilon for item in batch])
    A_func = [item.A_func for item in batch]
    A_adj = [item.A_adj for item in batch]
    f = [item.f for item in batch]
    grad_f = [item.grad_f for item in batch]
    #f_x_star = torch.stack([torch.tensor(item.f_x_star) for item in batch])
    return img, y, x0, std, sig, epsilon, A_func, A_adj, f, grad_f#, f_x_star

def my_collate_nonsmooth(batch):
    img = torch.stack([item.img for item in batch])
    y = torch.stack([item.y for item in batch])
    x0 = torch.stack([item.x0 for item in batch])
    std = torch.stack([torch.tensor(item.std) for item in batch])
    sig = torch.stack([torch.tensor(item.sig) for item in batch])
    epsilon = torch.stack([item.epsilon for item in batch])
    A_func = [item.A_func for item in batch]
    A_adj = [item.A_adj for item in batch]
    f = [item.f for item in batch]
    data_fit = [item.data_fit for item in batch]
    grad_data_fit = [item.grad_data_fit for item in batch]
    reg = [item.reg for item in batch]
    return img, y, x0, std, sig, epsilon, A_func, A_adj, f, data_fit, grad_data_fit, reg

def xray_collate(batch):
    img = torch.stack([item.img for item in batch])
    y = torch.stack([item.y for item in batch])
    x0 = torch.stack([item.x0 for item in batch])
    std = torch.stack([torch.tensor(item.std) for item in batch])
    sig = torch.stack([torch.tensor(item.sig) for item in batch])
    epsilon = torch.stack([item.epsilon for item in batch])
    A_func = [item.A_func for item in batch]
    A_adj = [item.A_adj for item in batch]
    f = [item.f for item in batch]
    grad_f = [item.grad_f for item in batch]
    #f_x_star = torch.stack([torch.tensor(item.f_x_star) for item in batch])
    return img, y, x0, std, domain, geometry, shape, sinograms, new_A_func, new_A_adj, new_f, pseudo_inv
