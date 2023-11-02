### Importing packages and modules
import odl
import torch
import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from LGS_train_module import get_images, geometry_and_ray_trafo, LGD, LGD2, ConstTauModel
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

def huber_total_variation(u, eps=0.05):
    diff_x, diff_y = Du(u)
    norm_2_1 = torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2+1e-08)))
    return norm_2_1

def reg(x, alpha):
    return alpha * huber_total_variation(x)

reg_func = lambda x: reg(x, 0.0001)

def f(x, y, operator, alpha=0.0001):
    return 0.5 * (torch.norm(operator(x) - y, p=2) ** 2) + reg(x, alpha)


# Load the list of tensors from the specified file
x_stars = torch.load("x_stars.pt")
x_list = torch.load("x_list.pt")
x_stars_test = torch.load("x_stars_test.pt")
x_list_test = torch.load("x_list_test.pt")
f_images = torch.load("f_images.pt")
g_sinograms = torch.load("g_sinograms.pt")
f_rec_images = torch.load("f_rec_images.pt")
f_test_rec_images = torch.load("f_test_rec_images.pt")
f_test_images = torch.load("f_test_images.pt")
g_test_sinograms = torch.load("g_test_sinograms.pt")

# print(x_stars[0].shape)
# print(x_stars_test[0].shape)
# print(f_images.shape)
# print(g_sinograms.shape)
# print(f_rec_images.shape)
# print(f_test_rec_images.shape)
# print(f_test_images.shape)
# print(g_test_sinograms.shape)



### Defining variables which define the amount of training and testing data
### being used. The training_scale is between 0 and 1 and controls how much
### training data is taken from whole data
training_scale = 1
amount_of_data = g_sinograms.shape[0]
n_train = int(np.floor(training_scale * amount_of_data))
n_test = int(np.floor(amount_of_data - n_train))


### Using functions from "LGS_train_module". Taking shape from images to produce
### odl parameters and getting Ray transform operator and its adjoint.
shape = (np.shape(f_images)[2], np.shape(f_images)[3])
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 1)

### Defining FBP operator
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

partial_x = odl.PartialDerivative(domain, axis=0)
partial_y = odl.PartialDerivative(domain, axis=1)
regularizer = OperatorModule(partial_x.adjoint * partial_x + partial_y.adjoint * partial_y).to(device)

### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)





#print('EVAL',f(x_stars_test[0], g_test_sinograms[0,:,:,:], ray_transform_module))







print(f(x_stars[0], g_sinograms[0,:,:,:], ray_transform_module))
print(f(f_rec_images[0,:,:,:], g_sinograms[0,:,:,:], ray_transform_module))

# x_stars = [(x_star-torch.min(x_star))/(torch.max(x_star)-torch.min(x_star)) for x_star in x_stars]
# x_stars_test = [(x_star-torch.min(x_star))/(torch.max(x_star)-torch.min(x_star)) for x_star in x_stars_test]

# plt.imshow(x_stars[0].squeeze(0).cpu().detach().numpy(), cmap='gray')
# plt.show()

fx_star = [f(x_stars[k].unsqueeze(0), g_sinograms[k,:,:,:].unsqueeze(0), ray_transform_module) for k in range(f_rec_images.shape[0])]
fx_star_test = [f(x_stars_test[k], g_test_sinograms[k,:,:,:], ray_transform_module) for k in range(f_test_rec_images.shape[0])]


x_stars_tensor = torch.cat(x_stars, dim=0)
x_stars_test_tensor = torch.cat(x_stars_test, dim=0)

fx_stars_array = [float(i) for i in fx_star] 
fx_stars_test_array = [float(i) for i in fx_star_test]

print(f(x_stars[0], g_sinograms[0,:,:,:], ray_transform_module))
print(f(f_rec_images[0,:,:,:], g_sinograms[0,:,:,:], ray_transform_module))
print(f(x_stars_tensor[0,:,:].unsqueeze(0), g_sinograms[0,:,:,:], ray_transform_module))

### Setting UNet as model and passing it to the used device
#LGS_network = LGD(adjoint_operator_module, ray_transform_module, lambda x: reg(x, 0.0001), in_channels=2, out_channels=1, step_length=0.005, n_iter=5).to(device)
#LGS_network = LGD2(adjoint_operator_module, ray_transform_module, lambda x: reg(x, 0.0001), in_channels=6, out_channels=1).to(device)
tau_network = ConstTauModel(adjoint_operator_module, ray_transform_module, lambda x: reg(x, 0.0001), in_channels=6, out_channels=1).to(device)

### Getting model parameters
#LGS_parameters = list(LGS_network.parameters())
tau_parameters = list(tau_network.parameters())

### Defining PSNR function
def psnr(loss):
    psnr = 10 * np.log10(1.0 / loss+1e-10)
    return psnr

loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

### Setting up some lists used later
running_loss = []
running_test_loss = []

import random


### Defining training scheme
def train_network(net, x_list, x_stars, x_list_test, x_stars_test, n_train=500000, batch_size=4):

    n_iter = 5
    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(tau_parameters, lr=0.01) #betas = (0.9, 0.99)

    ### Here starts the itarting in training
    for i in range(n_train):

        ## first choose which image to use
        n_index = random.choice(range(len(x_stars)))
        
        x_star = x_stars[n_index]
        x_list_new = x_list[n_index]
        y = g_sinograms[n_index,:,:,:]

        ## now choose which element of the list to use
        new_list = range(len(x_list_new))
        ## decreasing weights, (len(new_list)-i)/const
        probs = [np.exp(-0.1*k) for k in range(len(new_list))]
        probs = [i/sum(probs) for i in probs]
        k_iter = random.choices(new_list, weights=probs, k=1)[0]
        xk = x_list_new[k_iter].to(torch.float32).unsqueeze(0).detach()
        net.train()
        
        optimizer.zero_grad()

        outs, _, outs_list = net(xk, y, n_iter=1)

        #for outs_el in outs_list:
        #    loss += loss_train(outs_el, x_star)/len(outs_list)
        loss = f(outs, y, ray_transform_module)


        loss.backward()
        
        ### Here gradient clipping could be used
        torch.nn.utils.clip_grad_norm_(tau_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()


        if i % 100==0:

            train_loss = loss.item()
            running_loss.append(train_loss)
            print(train_loss, k_iter)
            print(_)
            #print(float(new_tau), tauk)
   
    ### After iterating taking one reconstructed image and its ground truth
    ### and showing them
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(outs[0,0,:,:].cpu().detach().numpy())
    plt.subplot(1,2,2)
    plt.imshow(f_batch[0,0,:,:].cpu().detach().numpy())
    plt.show()

    ### Plotting running loss and running test loss
    plt.figure()
    plt.semilogy(running_loss)
    plt.semilogy(running_test_loss)
    plt.show()

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, net = train_network(tau_network, x_list, x_stars, x_list_test, x_stars_test, n_train=500001, \
                                                                  batch_size=1)

### Evaluating the network
net.eval()
### Taking images and plotting them to show how the neural network does succeed
image_number = int(np.random.randint(g_test_sinograms.shape[0], size=1))
LGD_reconstruction, _ = net(f_test_rec_images[None,image_number,:,:,:], g_test_sinograms[None,image_number,:,:,:])
LGD_reconstruction = LGD_reconstruction[0,0,:,:].cpu().detach().numpy()
ground_truth = f_test_images[image_number,0,:,:].cpu().detach().numpy()
noisy_reconstruction = f_test_rec_images[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(noisy_reconstruction)
plt.subplot(1,3,2)
plt.imshow(LGD_reconstruction)
plt.subplot(1,3,3)
plt.imshow(ground_truth)
plt.show()

