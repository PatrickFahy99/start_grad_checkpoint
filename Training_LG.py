### This is a training script for the LGS network.

### Needed packages: -odl
###                  -PyTorch
###                  -NumPy
###                  -matplotlib
###                  -LGS_train_module.py (NEEDS ITS OWN PACKAGES EG. OpenCV)
###

### Importing packages and modules
import odl
import torch
import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from training_LG_module import get_images, geometry_and_ray_trafo, LGD
import matplotlib.pyplot as plt

### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Using function "get_images" to import images from the path.
n_images = 10
images = get_images(r'C:\Users\Patrick\XRayL2O\lhq_256', amount_of_images=n_images, scale_number=2)
### Converting images such that they can be used in calculations
images = np.array(images, dtype='float32')
images = torch.from_numpy(images).float().to(device)

### Using functions from "LGS_train_module". Taking shape from images to produce
### odl parameters and getting Ray transform operator and its adjoint.
shape = (np.shape(images)[1], np.shape(images)[2])
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

print('FIRST SINOGRAMS')
### Making sinograms from the images using Radon transform module
sinograms = ray_transform_module(images)

### Allocating used tensors
noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape)
rec_images = torch.zeros((sinograms.shape[0], ) + shape)

### Defining variables which define the amount of training and testing data
### being used. The training_scale is between 0 and 1 and controls how much
### training data is taken from whole data
training_scale = 1
amount_of_data = sinograms.shape[0]
n_train = int(np.floor(training_scale * amount_of_data))
n_test = int(np.floor(amount_of_data - n_train))

mean = 0
percentage = 0.01

print('GET SINOGRAMS')
### Adding Gaussian noise to the sinograms. Here some problem solving is
### needed to make this smoother.
for k in range(np.shape(sinograms)[0]):
    if k%100==0:
        print(k)
    sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, sinogram_k.std(), sinogram_k.shape) * percentage
    noisy_sinogram = sinogram_k + noise
    noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)
print('REC IMAGES')
### Using FBP to get reconstructed images from noisy sinograms
rec_images = fbp_operator_module(noisy_sinograms)

print('TO DEVICE')
### All the data into same device
sinograms = sinograms[:,None,:,:].to(device)
noisy_sinograms = noisy_sinograms[:,None,:,:].to(device)
rec_images = rec_images[:,None,:,:].to(device)
images = images[:,None,:,:].to(device)

f_images = images[0:n_train]
g_sinograms = noisy_sinograms[0:n_train]
f_rec_images = rec_images[0:n_train]

print('TEST IMAGES')
#test_images = get_images(r'C:\Users\Patrick\XRayL2O\lhq_256', amount_of_images='all', scale_number=2)
n_test_images = 10
test_images = get_images(r'C:\Users\Patrick\XRayL2O\lhq_256', amount_of_images = n_test_images, scale_number=2)

test_images = np.array(test_images, dtype='float32')
test_images = torch.from_numpy(test_images).float().to(device)

list_of_test_images = list(range(0,363,5))

test_sinograms = ray_transform_module(test_images)

test_noisy_sinograms = torch.zeros((test_sinograms.shape[0], ) + output_shape)
test_rec_images = torch.zeros((test_sinograms.shape[0], ) + shape)

print('TEST SINOGRAMS')
for k in range(np.shape(test_sinograms)[0]):
    if k%100==0:
        print(k)
    test_sinogram_k = test_sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, test_sinogram_k.std(), test_sinogram_k.shape) * percentage
    test_noisy_sinogram = test_sinogram_k + noise
    test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)
                                                  

test_rec_images = fbp_operator_module(test_noisy_sinograms).to(device)
    
test_sinograms = test_sinograms[:,None,:,:].to(device)
test_noisy_sinograms = test_noisy_sinograms[:,None,:,:].to(device)
test_rec_images = test_rec_images[:,None,:,:].to(device)
test_images = test_images[:,None,:,:].to(device)

# indices = np.random.permutation(test_rec_images.shape[0])[:75]
#f_test_images = test_images[list_of_test_images]
#g_test_sinograms = test_noisy_sinograms[list_of_test_images]
#f_test_rec_images = test_rec_images[list_of_test_images]
f_test_images = test_images.to(device)
g_test_sinograms = test_noisy_sinograms.to(device)
f_test_rec_images = test_rec_images.to(device)

### Plotting one image from all and its sinogram and noisy sinogram
image_number = 1
noisy_sino = noisy_sinograms[image_number,0,:,:].cpu().detach().numpy()
orig_sino = sinograms[image_number,0,:,:].cpu().detach().numpy()
orig = rec_images[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(orig, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(noisy_sino, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(orig_sino, cmap='gray')
plt.show()

image_number = 1
noisy_sino = test_noisy_sinograms[image_number,0,:,:].cpu().detach().numpy()
orig_sino = test_sinograms[image_number,0,:,:].cpu().detach().numpy()
orig = test_rec_images[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(orig, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(noisy_sino, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(orig_sino, cmap='gray')
plt.show()


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
    return 0.5 * (torch.norm(operator(x) - y) ** 2) + reg(x, alpha)

print(torch.norm(ray_transform_module(f_images[0]) - g_sinograms[0]))
print(torch.norm(ray_transform_module(f_images) - g_sinograms)/f_images.shape[0])
print(f(f_images[0], g_sinograms[0], ray_transform_module))
### Setting UNet as model and passing it to the used device
LGS_network = LGD(adjoint_operator_module, ray_transform_module, reg_func, in_channels=3, out_channels=1, step_length=0.005, n_iter=5).to(device)

### Getting model parameters
LGS_parameters = list(LGS_network.parameters())

# ### Defining PSNR function
# def psnr(loss):
    
#     psnr = 10 * np.log10(1.0 / loss+1e-10)
    
#     return psnr

# loss_train = nn.MSELoss()
# loss_test = nn.MSELoss()


### Setting up some lists used later
running_loss = []
running_test_loss = []

def mean(lst):
    return sum(lst)/len(lst)

### Defining training scheme
def train_network(net, f_images, g_sinograms, f_rec_images, f_test_rec_images, \
                  f_test_images, g_test_sinograms, n_train=50000, batch_size=4):

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(LGS_parameters, lr=0.001) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_train)

    ### Here starts the itarting in training
    for i in range(n_train):
        print(i)
        
        # scheduler.step()
        
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data
        #print(g_sinograms.shape)
        #print(batch_size)
        n_index = np.random.permutation(g_sinograms.shape[0])[:batch_size]
        g_batch = g_sinograms[n_index,:,:,:].to(device)
        f_batch = f_images[n_index].to(device)
        f_batch2 = f_rec_images[n_index].to(device)
        
        #print(n_index)


        net.train()
        
        optimizer.zero_grad()

        ### Evaluating the network which produces reconstructed images.
        outs, _ = net(f_batch2, g_batch)
        
        ### Calculating loss of the outputs
        #loss = loss_train(f_batch, outs)
        #print(outs[0].shape)
        #print(f_batch2.shape)
        #lst = [f(x, g_batch, ray_transform_module) for x in outs]
        lst = [[f(outs[j][i,:,:,:], g_batch[i,:,:,:], ray_transform_module)/n_test_images for i in range(outs[j].shape[0])] for j in range(len(outs))][-1]
        #print(mean([mean(i) for i in lst2]))
        loss = sum([sum([i]) for i in lst])/(len(lst))

        #print(lst)
        #loss = mean(lst)

        if i%100==0:
            print('TRAIN LOSS:',loss)
        
        ### Calculating gradient
        loss.backward()
        
        ### Here gradient clipping could be used
        torch.nn.utils.clip_grad_norm_(LGS_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()
        #scheduler.step()

        if i%100==0:
            print('STEP LENGTH:', net.step_length.float().item())

        #print('TEST')
        ### Here starts the running tests
        if i % 100 == 0:
            
            ### Using predetermined test data to see how the outputs are
            ### in our neural network
            # net.eval()
            #with torch.no_grad():
            outs2, step_len = net(f_test_rec_images, g_test_sinograms)
            ### Calculating test loss with test data outputs
            #test_loss = loss_test(f_test_images, outs2).item()
            lst2 = [[f(outs2[j][i,:,:,:], g_test_sinograms[i,:,:,:], ray_transform_module)/n_test_images for i in range(outs2[j].shape[0])] for j in range(len(outs2))][-1]
            #print(mean([mean(i) for i in lst2]))
            test_loss = sum([sum([i]) for i in lst2])/(len(lst2))
            if i%100==0:
                print('TEST LOSS:', test_loss)
            train_loss = loss.item()
            running_loss.append(train_loss)
            running_test_loss.append(test_loss)

            
            # ### Printing some data out
            # if i % 500 == 0:
            #     print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss: {test_loss:.2e}') #, end='\r'
            #     print(f'Step lenght: {step_len[0]}')
            #     plt.figure()
            #     plt.subplot(1,2,1)
            #     plt.imshow(outs2[54,0,:,:].cpu().detach().numpy())
            #     plt.subplot(1,2,2)
            #     plt.imshow(f_test_images[54,0,:,:].cpu().detach().numpy())
            #     plt.show()
                
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
running_loss, running_test_loss, net = train_network(LGS_network, f_images, \
                                                                  g_sinograms, f_rec_images, \
                                                                  f_test_rec_images, f_test_images, \
                                                                  g_test_sinograms, n_train=50001, \
                                                                  batch_size=4)

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

torch.save(net.state_dict(), 'LGD_TRAINED_VAR')

