import torch
import torch.nn as nn
import torch.nn.functional as F


### Function needed when defining the UNet encoding and decoding parts
def double_conv_and_ReLU(in_channels, out_channels):
    list_of_operations = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU()
    ]

    return nn.Sequential(*list_of_operations)

### Class for the Learned Gradient Descent (LGD) algorithm.
class LGS(nn.Module):
    def __init__(self, adjoint_operator_module, operator_module, \
                  g_sinograms, f_rec_images, in_channels, out_channels, step_length, n_iter):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.step_length = step_length
        self.step_length = nn.Parameter(torch.zeros(1,1,1,1))
        self.n_iter = n_iter
        
        self.operator = operator_module
        self.gradient_of_f = adjoint_operator_module

        
        LGS_layers = [
            nn.Conv2d(in_channels=self.in_channels, \
                                    out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels, \
                                    kernel_size=(3,3), padding=1),
        ]
        
        self.layers = nn.Sequential(*LGS_layers)
        
        self.layers2 = [self.layers for i in range(n_iter)]
 

    def forward(self, f_rec_images, g_sinograms):
        total_lst = [f_rec_images]
        for i in range(self.n_iter):
        
            f_sinogram = self.operator(f_rec_images)
            
            grad_f = self.gradient_of_f(f_sinogram - g_sinograms) # (output of dual - g_sinograms)
            
            u = torch.cat([f_rec_images, grad_f], dim=1)
            
            u = self.layers2[i](u)

            df = -self.step_length * u[:,0:1,:,:]
            
            f_rec_images = f_rec_images + df

            total_lst.append(f_rec_images)
        
        return f_rec_images, self.step_length, total_lst
    

class LGS2(nn.Module):
    def __init__(self, adjoint_operator_module, operator_module, reg, in_channels, out_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.step_length = step_length
        self.step_length = nn.Parameter(torch.zeros(1,1,1,1))
        self.reg = reg
        
        self.operator = operator_module
        self.adjoint_operator = adjoint_operator_module
        
        self.LGD_layers = [
            nn.Conv2d(in_channels=self.in_channels, \
                                    out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels, \
                                    kernel_size=(3,3), padding=1),
        ]

        ###Weight-sharing, each iterate has the same weight

        ### Every iterate has a different weights
        ### Initializing the parameters for every unrolled iteration
        #self.layers2 = [nn.Sequential(*LGD_layers) for i in range(n_iter)]

        self.layers = nn.Sequential(*self.LGD_layers)
        
    def forward(self, x, y, n_iter):
        total_lst = [x]

        layers2 = [self.layers for i in range(n_iter)]
        
        for i in range(n_iter):
        
            f_sinogram = self.operator(x)

            adjoint_eval = self.adjoint_operator(f_sinogram - y)

            x = x.requires_grad_(True)
            new_reg_value = self.reg(x)
            grad_reg_new = torch.autograd.grad(new_reg_value, x)[0]

            if i == 0:
                x_old = x
                adjoint_eval_old = adjoint_eval
                grad_reg_old = grad_reg_new

            u = torch.cat([x, adjoint_eval + grad_reg_new, x_old, adjoint_eval_old+grad_reg_old], dim=1)
            
            u = layers2[i](u)
            #df = -self.step_length *u[:,0:1,:,:]
            df = -self.step_length *(adjoint_eval + grad_reg_new + u[:,0:1,:,:])
            
            x = x + df

            x_old = x
            adjoint_eval_old = adjoint_eval
            grad_reg_old = grad_reg_new

            total_lst.append(x)
        
        return x, self.step_length, total_lst




class BackTrackingTau(nn.Module):
    def __init__(self, adjoint_operator_module, operator_module, reg, in_channels, out_channels):
        super().__init__()

        self.reg = reg
        
        self.operator = operator_module
        self.adjoint_operator = adjoint_operator_module

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*64, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x, y, n_iter=1):

        x_list = []
        tau_list = []
        
        for i in range(n_iter):
            #print(grad_fit_new.shape, grad_fit_old.shape, grad_reg_new.shape, grad_reg_old.shape)

        
            f_sinogram = self.operator(x)

            adjoint_eval = self.adjoint_operator(f_sinogram - y)

            x = x.requires_grad_(True)
            new_reg_value = self.reg(x)
            grad_reg_new = torch.autograd.grad(new_reg_value, x)[0]

            if i == 0:
                x_old = x
                adjoint_eval_old = adjoint_eval
                grad_reg_old = grad_reg_new

            
            u = torch.cat([x, adjoint_eval+grad_reg_new, x_old, adjoint_eval_old+grad_reg_old], dim=0)

            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.flatten()  # Flatten the tensor
            u = torch.relu(self.fc1(u))
            new_tau = F.softplus(self.fc2(u))
            #u = layers2[i].to(device)(u).to(device)

            #df = -self.step_length *u[:,0:1,:,:]
            df = -new_tau * (adjoint_eval + grad_reg_new)
            
            x = x + df

            x_old = x
            adjoint_eval_old = adjoint_eval
            grad_reg_old = grad_reg_new

            x_list.append(x)
            tau_list.append(new_tau)
        
        return x, tau_list, x_list
    



