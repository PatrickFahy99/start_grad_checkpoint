### Importing packages and modules
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


from LGS_train_module import get_images

n_images = 10000
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
    num_mult_fives = 100


    for i in range(n_train): 
        


        n_index = np.random.permutation(len(ys))[:batch_size]
        n_index = int(n_index)
        g_batch = ys[n_index].unsqueeze(0).float()
        f_batch2 = 0*g_batch.float()
        
        net.train()
        
        optimizer.zero_grad()


        inpt = f_batch2
        inpt_lst = [inpt]
        tau_lst = []
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
            print('TAUS', tau_lst)
            
            train_loss = loss.item()

            running_loss.append(train_loss)


            print('LOSS:', train_loss)#, test_loss)

        if i % train_iters_one == 0:
            if num_mult_fives == 0:
                if torch.isnan(loss):
                    print('NAN')
                    break
                torch.save(net.state_dict(), 'BLURRING_TAU_UNSUPERVISED.pth')
                print('saved')      
                print('DEFAULT DIRECTORY')
            else:
                if torch.isnan(loss):
                    print('NAN')
                    break
                torch.save(net.state_dict(), f'BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_{iteration_number}_{num_mult_fives}_{train_iters_one}.pth')
                print('saved')
                print(f'BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_{iteration_number}_{num_mult_fives}_{train_iters_one}.pth')

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, net = train_network(tau_network, ys, 
                                                     n_train=1000000000, \
                                                                  batch_size=1)

