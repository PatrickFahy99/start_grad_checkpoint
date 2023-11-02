### Importing packages and modules
import odl
import torch
import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from LGS_train_module import get_images, geometry_and_ray_trafo
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
x_stars = torch.load("x_list_bt.pt")
taus = torch.load("tau_list_bt.pt")
x_stars_test = torch.load("x_list_test_bt.pt")
taus_test = torch.load("tau_list_test_bt.pt")
f_images = torch.load("f_images_bt.pt")
g_sinograms = torch.load("g_sinograms_bt.pt")
f_rec_images = torch.load("f_rec_images_bt.pt")
f_test_rec_images = torch.load("f_test_rec_images_bt.pt")
f_test_images = torch.load("f_test_images_bt.pt")
g_test_sinograms = torch.load("g_test_sinograms_bt.pt")

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


print(f(x_stars[0][-1], g_sinograms[0,:,:,:], ray_transform_module))
print(f(f_rec_images[0,:,:,:], g_sinograms[0,:,:,:], ray_transform_module))

from LGS_train_module import BackTrackingTauTrain#, BackTrackingX

LGS_network = BackTrackingTauTrain(adjoint_operator_module, ray_transform_module, lambda x: reg(x, 0.0001), in_channels=4, out_channels=1).to(device)

### Getting model parameters
LGS_parameters = list(LGS_network.parameters())

def loss_train(pred, actual):
    diff = (pred-actual)
    if float(pred)>float(actual):
        return (pred-actual)/actual
    else:
        return (actual-pred)/actual

### Setting up some lists used later
running_loss = []
running_test_loss = []

def psnr(loss):
    psnr = 10 * np.log10(1.0 / loss+1e-10)
    return psnr
import random
### Defining training scheme
def train_network(net, x_stars, taus, x_stars_test, taus_test, n_train=50000, batch_size=4):

    n_iter = 0
    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(LGS_parameters, lr=0.0005) #betas = (0.9, 0.99)

    ### Here starts the itarting in training
    for i in range(n_train):

        ## first choose which image to use
        n_index = random.choice(range(len(x_stars)))
    
        x_list = x_stars[n_index]
        tau_list = taus[n_index]
        y = g_sinograms[n_index,:,:,:]

        ## now choose which element of the list to use
        k_iter = random.choice(range(len(x_list)))
        xk = x_list[k_iter].to(torch.float32)
        if k_iter > 0:
            xkm1 = x_list[k_iter-1].to(torch.float32)
        else:
            xkm1 = x_list[k_iter].to(torch.float32)
        tauk = tau_list[k_iter]
        
        net.train()
        
        optimizer.zero_grad()

        new_tau = net(xk, xkm1, y)

        #print(new_tau, tauk)

        loss = loss_train(new_tau, torch.tensor([tauk]).to(torch.float32).to(device))

        loss.backward()
        
        ### Here gradient clipping could be used
        torch.nn.utils.clip_grad_norm_(LGS_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()

        if i % 1000==0:

            train_loss = loss.item()
            running_loss.append(train_loss)
            print(i, 'PSNR:', psnr(train_loss))
            print(float(new_tau), tauk)

        # ### Here starts the running tests
        # if i % 10000 == 0:

        #     print(i)
            
        #     new_tau = net(xk, xkm1, y)


        #     print(outs2.shape)
        #     print(f_test_images.shape)

        #     test_loss = 0
        #     for k in range(outs2.shape[0]):
        #         test_loss += loss_test(x_stars_test_tensor[k,:,:].unsqueeze(0) , outs2[k,:,:,:])/outs2.shape[0]

            
        #     test_loss = 0
        #     for outs_el in outs2_list:
        #         for k in range(outs2.shape[0]):
        #             test_loss += loss_test(x_stars_test_tensor[k,:,:].unsqueeze(0) , outs_el[k,:,:,:])/(len(outs2_list)*outs2.shape[0])
            

        #     #test_loss = loss_test(torch.cat([x_stars_test_tensor[k,:,:].unsqueeze(0) for k in range(x_stars_test_tensor.shape[0])], dim=0), outs2)
        #     train_loss = loss.item()
        #     test_loss = test_loss.item()
        #     running_loss.append(train_loss)
        #     running_test_loss.append(test_loss)

        #     print(f(x_star_batch, g_batch, ray_transform_module))
        #     print(f(x_stars_test_tensor[0].unsqueeze(0), g_test_sinograms[0], ray_transform_module))

        #     print(train_loss, test_loss)

        #     print('PSNRs:', psnr(train_loss), psnr(test_loss))
            
        if i % 1000 == 0:
            torch.save(net.state_dict(), 'BACKTRACKING_IMITATION_TAU.pth')
            print('saved')      
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
running_loss, running_test_loss, net = train_network(LGS_network, x_stars, taus, x_stars_test, taus_test, n_train=500001, \
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

