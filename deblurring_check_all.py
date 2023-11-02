### Importing packages and modules
#import odl
import torch
import torch.nn as nn
import torch.optim as optim
#from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from LGS_train_module import get_images, TauModelNoAdjoint
#import matplotlib.pyplot as plt
#import time
from torchvision.transforms import GaussianBlur
#from datasets import Blur2Dataset, ImageDataBlur2

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
    return torch.sum(huber(torch.sqrt(diff_x**2 + diff_y**2+1e-08)))

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

def reg_func(x):
    return reg(x, alpha)

def f(x, y, operator, alpha):
    return 0.5 * (torch.norm(operator(x) - y, p=2) ** 2) + reg(x, alpha)

n_images = 10000
#images = get_images(r'C:\Users\Patrick\XRayL2O\notumor', n_images, scale_number=2)
images = get_images(r'C:\Users\Patrick\XRayL2O\lhq_256', n_images, scale_number=2)
images = np.array(images, dtype='float32')
images = torch.from_numpy(images).float().to(device)




std = 0.05

blur_level = 10
blur_size = 7
# training_set = Blur2Dataset(images, alpha, std, blur_level, blur_size)
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

### Getting model parameters
tau_parameters = list(tau_network.parameters())

print(sum(p.numel() for p in tau_network.parameters() if p.requires_grad))


loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

### Setting up some lists used later
running_loss = []
running_test_loss = []

### Defining training scheme
def train_network(net, ys, n_train=50000, batch_size=4):

    n_iter = 0
    num_mult_fives = 0

    print('TRAINING STARTED')

    optimizer = optim.Adam(tau_parameters, lr=0.0005) #betas = (0.9, 0.99)
    ## reduced lr to stop nan
    iteration_number = 1
    train_iters_one = 1000
    
    
    # If you have adjacent point-wise operations you can use PyTorch JIT
    # to combine them into one FusionGroup which can then be launched on a single kernel rather than multiple kernels as would have been done per default. You'll also save some memory reads and writes. 

    
    # Gradient Accumulation: Instead of updating model parameters after processing each batch, you can accumulate gradients over several batches before performing a parameter update. This reduces memory usage but requires more compute cycles.
    
    for i in range(n_train): 

        if n_iter <= iteration_number-1:
            if (i % train_iters_one == 0):
                print('NUMBER OF ITERS:', n_iter)
        if (i%(iteration_number*train_iters_one) == 0):
            if num_mult_fives < 1000:
                num_mult_fives+=1
            optimizer = optim.Adam(tau_parameters, lr=0.0001) ## maybe scale by num_mult_fives
            optimizer.zero_grad()
            n_iter=1
            print(f'NUMBER OF MULTIPLES OF {iteration_number}:', num_mult_fives)
            print('NUMBER OF ITERS:', n_iter)

        n_index = np.random.permutation(len(ys))[:batch_size]
        n_index = int(n_index)
        g_batch = ys[n_index].unsqueeze(0).float()
        f_batch2 = 0*g_batch.float()
        
        net.train()
        
        optimizer.zero_grad()


        inpt = f_batch2
        inpt_lst = [inpt]
        tau_lst = []
        if num_mult_fives != 0:
            for k in range(num_mult_fives):
                outs, _, outs_list = net(inpt, g_batch, n_iter=iteration_number)
                loss = f(outs, g_batch, model, alpha)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tau_parameters, max_norm=1.0, norm_type=2)
                optimizer.step()
                inpt = outs
                inpt = inpt.detach()
                inpt_lst.append(inpt)
                tau_lst.append(_)
                optimizer.zero_grad()

        ### Here starts the running tests
        if i % int(train_iters_one/10) == 0:

            print(i)

            print('f INPUTS:', [f(inpt, g_batch, model, alpha) for inpt in inpt_lst])
            print('f OUTPUT', f(outs, g_batch, model, alpha))
            print('TAUS', tau_lst)
            
            train_loss = loss.item()

            running_loss.append(train_loss)


            print('LOSS:', train_loss)#, test_loss)

        if i % train_iters_one == 0 and num_mult_fives == 0:
                if torch.isnan(loss):
                    print('NAN')
                    break
                torch.save(net.state_dict(), 'models/BLURRING_TAU_UNSUPERVISED.pth')
                print('saved')      
                print('DEFAULT DIRECTORY')
        elif i % train_iters_one == 0:
            if torch.isnan(loss):
                print('NAN')
                break
            torch.save(net.state_dict(), f'models/BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_{iteration_number}_{num_mult_fives}_{train_iters_one}_ALL.pth')
            print('saved')
            print(f'models/BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_{iteration_number}_{num_mult_fives}_{train_iters_one}_ALL.pth')

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, net = train_network(tau_network, ys, 
                                                     n_train=1000000000, \
                                                                  batch_size=1)

