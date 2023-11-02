### Importing packages
import os
import cv2 as cv
import numpy as np
import odl
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch.jit as jit

device = 'cuda' if torch.cuda.is_available() else 'cpu'
### Function that takes all the images from the path directory and crops them.
### Cropping part is hardcoded to match certain type of pictures. One probably
### needs to change the values.
### Inputs: -'path': path to directory where the images are
###         -'amount_of_images': how many images one wants to get from the
###                              given directory
###         -'scale_number': number of how many pixels does the function skip.
###                          Eg. scale_number = 4 -> every 4th pixel is taken
###                          from the original image
### Outputs: -'all_images': list of all images taken from directory
def get_images(path, amount_of_images='all', scale_number=1):

    all_images = []
    all_image_names = os.listdir(path)
    print(len(all_image_names))
    if amount_of_images == 'all':
        i=0
        for name in all_image_names:
            i+=1
            try:
                image = Image.open(path + '/' + name).convert("L")
                image = torch.tensor(np.array(image)) / np.max(np.array(image))  # normalize the pixel values
                image = image.unsqueeze(0).unsqueeze(0)
                image = F.interpolate(image, size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                all_images.append(image)
                print(i)
            except:
                print('didnt work')
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
        images_to_take = [all_image_names[i] for i in temp_indexing]
        for name in images_to_take:
            #temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            # # Convert the image to grayscale
            # grayscale_image = cv.cvtColor(temp_image, cv.COLOR_BGR2GRAY)
            # image = grayscale_image[90:410, 90:410]
            # image = image[0:320:scale_number, 0:320:scale_number]
            # image = image / 0.07584485627272729
            # all_images.append(image)
            image = Image.open(path + '/' + name).convert("L")
            image = torch.tensor(np.array(image)) / np.max(np.array(image))  # normalize the pixel values
            image = image.unsqueeze(0).unsqueeze(0)
            image = F.interpolate(image, size=(256,256), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            all_images.append(image)
    
    return all_images


### Function that defines mathematical background for the script.
### More precise function defines geometry for the Radon transform.
### Inputs: -'setup': determines what kind of geometry one wants. Possible
###                   choices are 'full', 'sparse', 'limited'. Default: 'full'
###         -'min_domain_corner': Determines where the bottom left corner is
###                               is in the geometry. Default: [-1,-1]
###         -'max_domain_corner': Determines where the upper right corner is
###                               is in the geometry. Default: [1,1]
###         -'shape': how many points there is in x- and y-axis between the
###                   corners of the geometry. Default: (100,100)
###         -'source_radius': radius of the 'object' when taken measurements.
###                           Default: 2
###         -'detector_radius': radius of the ??? when taken measurements.
###                             Default: 1
###         -'dtype': Python data type. Default: 'float32'
###         -'device': Device which is used in calculations. Default: 'cpu'
###         -'factor_lines': Parameter which controls the line-measurements
###                          in 'sparse' and 'limited' geometries.
###                          Default: 1
### Outputs: -'domain': odl domain, not really used, could be deleted from
###                     the outputs
###          -'geometry': odl geometry, could be deleted from the outputs
###          -'ray_transform': Radon transform operator defined by
###                            given geometry
###          -'output_shape': Shape defined by angles and lines in geometry.
###                           Needed in the allocations.
def geometry_and_ray_trafo(setup='full', min_domain_corner=[-1,-1], max_domain_corner=[1,1], \
                           shape=(100,100), source_radius=2, detector_radius=1, \
                           dtype='float32', device='cpu', factor_lines = 1):

    device = 'astra_' + device
    print(device)
    domain = odl.uniform_discr(min_domain_corner, max_domain_corner, shape, dtype=dtype)

    if setup == 'full':
        angs = 360
        factor_lines = 1
        angles = odl.uniform_partition(0, 2*np.pi, angs)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(1028/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (angs,int(1028/factor_lines))
    elif setup == 'sparse':
        angle_measurements = 100
        line_measurements = int(512/factor_lines)
        angles = odl.uniform_partition(0, 2*np.pi, angle_measurements)
        lines = odl.uniform_partition(-1*np.pi, np.pi, line_measurements)
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (angle_measurements, line_measurements)
    elif setup == 'limited':
        starting_angle = 0
        final_angle = np.pi * 1.0
        angles = odl.uniform_partition(starting_angle, final_angle, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(1028/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (int(360), int(1028/factor_lines))
        
    ray_transform = odl.tomo.RayTransform(domain, geometry, impl=device)

    return domain, geometry, ray_transform, output_shape

def double_conv_and_ReLU(in_channels, out_channels):
    list_of_operations = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU()
    ]

    return nn.Sequential(*list_of_operations)

### Class for the Learned Gradient Scheme (LGS) algorithm.
class LGD(nn.Module):
    def __init__(self, adjoint_operator_module, operator_module, reg, in_channels, out_channels, step_length, n_iter):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.step_length = step_length
        self.step_length = nn.Parameter(torch.zeros(1,1,1,1))
        self.n_iter = n_iter
        self.reg = reg
        
        self.operator = operator_module
        self.adjoint_operator = adjoint_operator_module
        
        LGD_layers = [
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
        self.layers = nn.Sequential(*LGD_layers)
        self.layers2 = [self.layers for i in range(n_iter)]

        ### Every iterate has a different weights
        ### Initializing the parameters for every unrolled iteration
        #self.layers2 = [nn.Sequential(*LGD_layers) for i in range(n_iter)]
        
    
    def forward(self, f_rec_images, g_sinograms):
        
        for i in range(self.n_iter):
        
            f_sinogram = self.operator(f_rec_images)

            adjoint_eval = self.adjoint_operator(f_sinogram - g_sinograms)

            f_rec_images = f_rec_images.requires_grad_(True)
            new_reg_value = self.reg(f_rec_images)
            grad_reg_new = torch.autograd.grad(new_reg_value, f_rec_images)[0]

            u = torch.cat([f_rec_images, adjoint_eval], dim=1)
            
            u = self.layers2[i].to(device)(u).to(device)

            #df = -self.step_length *u[:,0:1,:,:]
            df = -self.step_length *(adjoint_eval + grad_reg_new + u[:,0:1,:,:])
            
            f_rec_images = f_rec_images + df
        
        return f_rec_images, self.step_length
    

class LGD2(nn.Module):
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

        #layers2 = [self.layers for i in range(n_iter)]

        x_list = []
        
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

            u = torch.cat([x, adjoint_eval, x_old, adjoint_eval_old, grad_reg_new, grad_reg_old], dim=1)
            u = self.layers.to(device)(u).to(device)
            #u = layers2[i].to(device)(u).to(device)

            #df = -self.step_length *u[:,0:1,:,:]
            df = -self.step_length *(adjoint_eval + grad_reg_new + u[:,0:1,:,:])
            
            print(self.step_length)
            x = x + df

            x_old = x
            adjoint_eval_old = adjoint_eval
            grad_reg_old = grad_reg_new

            x_list.append(x)
        
        return x, self.step_length, x_list
    



class LGD2_normgrad(nn.Module):
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

        print(self.layers)
        
    
    def forward(self, x, y, n_iter):

        #layers2 = [self.layers for i in range(n_iter)]

        x_list = []
        
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

            u = torch.cat([x, adjoint_eval, x_old, adjoint_eval_old, grad_reg_new, grad_reg_old], dim=1)
            u = self.layers.to(device)(u).to(device)
            #u = layers2[i].to(device)(u).to(device)

            #df = -self.step_length *u[:,0:1,:,:]
            df = -self.step_length *(adjoint_eval + grad_reg_new + u[:,0:1,:,:])
            
            x = x + df

            x_old = x
            adjoint_eval_old = adjoint_eval
            grad_reg_old = grad_reg_new

            x_list.append(x)

            grad_norm = torch.norm(grad_reg_new+adjoint_eval, p=2)
        
        return x, self.step_length, x_list, grad_norm


class BackTrackingTauTrain(nn.Module):
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

        ###Weight-sharing, each iterate has the same weight

        ### Every iterate has a different weights
        ### Initializing the parameters for every unrolled iteration
        #self.layers2 = [nn.Sequential(*LGD_layers) for i in range(n_iter)]

        #self.layers = nn.Sequential(*self.tau_layers)
        
    
    def forward(self, x, x_old, y):

        x = x.requires_grad_(True)
        x_old = x_old.requires_grad_(True)

        f_sinogram = self.operator(x)
        f_sinogram_old = self.operator(x_old)

        grad_fit_new = self.adjoint_operator(f_sinogram - y)
        grad_fit_old = self.adjoint_operator(f_sinogram_old - y)

        new_reg_value = self.reg(x)
        grad_reg_new = torch.autograd.grad(new_reg_value, x)[0]
        old_reg_value = self.reg(x_old)
        grad_reg_old = torch.autograd.grad(old_reg_value, x_old)[0]

        #print(grad_fit_new.shape, grad_fit_old.shape, grad_reg_new.shape, grad_reg_old.shape)

        u = torch.cat([x, grad_fit_new+grad_reg_new, x_old, grad_fit_old+grad_reg_old], dim=0)

        u = self.pool(torch.relu(self.conv1(u)))
        u = self.pool(torch.relu(self.conv2(u)))
        u = u.view(u.size(0), -1)  # Flatten the tensor
        u = torch.relu(self.fc1(u))
        new_tau = F.softplus(self.fc2(u))

        # new_tau = self.layers.to(device)(u).to(device)
        
        return new_tau 



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
        self.fc2 = nn.Linear(64*64, 1)

    
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
            #u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)  # Flatten the tensor
            u = torch.relu(self.fc1(u))
            new_tau = F.softplus(self.fc2(u))

            #u = layers2[i].to(device)(u).to(device)

            #df = -self.step_length *u[:,0:1,:,:]
            df = - new_tau * (adjoint_eval + grad_reg_new)
            
            x = x + df

            x_old = x
            adjoint_eval_old = adjoint_eval
            grad_reg_old = grad_reg_new

            x_list.append(x)
            tau_list.append(new_tau)
        
        return x, tau_list, x_list
    


# class TauModelNoAdjoint(nn.Module):
#     def __init__(self, model, reg, in_channels, out_channels):
#         super().__init__()

#         ### Defining instance variables
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.reg = reg
        
#         self.model = model

#         self.f = lambda x,y : 0.5*(torch.norm(model(x)- y))**2 + reg(x)
        
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1)#, dtype=torch.double
#         self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64*64, 64)
#         self.fc2 = nn.Linear(64, 1)

#    
#    def forward(self, x, y, n_iter=1):

#         x_list = []
#         tau_list = []

#         new_f = lambda x : self.f(x,y)
        
#         for i in range(n_iter):

#             x = torch.tensor(x.clone().detach(), requires_grad=True).to(device)
#             f_val = new_f(x) 
#             grad_new = torch.autograd.grad(f_val, x)[0]

#             if i == 0:
#                 x_old = x
#                 grad_old = grad_new

#             #print(x.shape, adjoint_eval.shape, x_old.shape, adjoint_eval_old.shape, grad_reg_new.shape, grad_reg_old.shape)
#             u = torch.cat([x, x_old, grad_new, grad_old], dim=0)
    
#             #u = u.squeeze(1).unsqueeze(0)
#             u = self.pool(torch.relu(self.conv1(u)))
#             u = self.pool(torch.relu(self.conv2(u)))
#             u = u.view(u.size(0), -1)  # Flatten the tensor
#             u = torch.relu(self.fc1(u))
#             new_tau = F.softplus(self.fc2(u))

#             #u = layers2[i].to(device)(u).to(device)

#             #df = -self.step_length *u[:,0:1,:,:]
#             df = - new_tau * grad_new
            
#             x = x + df

#             x_old = x
#             grad_old = grad_new

#             x_list.append(x)
#             tau_list.append(new_tau)
        
#         return x, tau_list, x_list
    




class TauModelNoAdjoint(nn.Module):
    def __init__(self, model, reg, in_channels, out_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg = reg
        
        self.model = model

        self.f = lambda x, y: 0.5 * (torch.norm(model(x) - y, p=2).pow(2)) + reg(x)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    #@jit.script_method
    def forward(self, x, y, n_iter=1):

        x_list = []
        tau_list = []

        new_f = lambda x: self.f(x, y)
        
        for i in range(n_iter):

            x = torch.tensor(x.clone().detach(), requires_grad=True).to(device)
            f_val = new_f(x)
            grad_new = torch.autograd.grad(f_val, x, create_graph=True)[0]

            if i == 0:
                x_old = x
                grad_old = grad_new

            u = torch.cat([x, x_old, grad_new, grad_old], dim=0)
            ##u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)
            u = torch.relu(self.fc1(u))
            u = torch.relu(self.fc2(u))
            new_tau = F.softplus(self.fc3(u))

            df = -new_tau * grad_new
            x = x + df

            x_old = x
            grad_old = grad_new

            x_list.append(x)

            tau_list.append(new_tau)
        
        return x, tau_list, x_list
    
    


class TauModelNoAdjointBatchNorm(nn.Sequential):
    def __init__(self, model, reg, in_channels, out_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg = reg
        
        self.model = model

        self.f = lambda x, y: 0.5 * (torch.norm(model(x) - y, p=2).pow(2)) + reg(x)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(3)  # BatchNorm after conv1
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)  # BatchNorm after conv2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 64, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)  # BatchNorm for fc1
        self.fc2 = nn.Linear(64, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)  # BatchNorm for fc2
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
        # -> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC

    #@jit.script_method
    def forward(self, x, y, n_iter=1):

        x_list = []
        tau_list = []

        def new_f(x):
            return self.f(x, y)
        
        for i in range(n_iter):

            x = torch.tensor(x.clone().detach(), requires_grad=True).to(device)
            f_val = new_f(x)
            grad_new = torch.autograd.grad(f_val, x, create_graph=True)[0]
            x = x.detach()
            if i == 0:
                x_old = x ## x_old and x have the same memory space, so if x changes, x_old changes too
                grad_old = grad_new

            u = torch.cat([x, x_old, grad_new, grad_old], dim=0)
            #u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)
            u = torch.relu(self.fc1(u))
            u = torch.relu(self.fc2(u))
            new_tau = F.softplus(self.fc3(u))
            
            x_old = x.clone() ## x_old and x have different memory space
            grad_old = grad_new.clone()
            
            x -= new_tau * grad_new

            x_list.append(x)

            tau_list.append(new_tau)
            
            #print(torch.cuda.memory_allocated())
        
        return x, tau_list, x_list
    
    def forward_checkpoint(self, x, y, n_iter=1, checkpoints=[]):
        
        '''
        what we want to do
        In an RNN model. Before we start:
        save first input to memory
        compute forward pass iteration by iteration, if previous iteration value (which input to this iter) is not a checkpoint, then remove it from memory
        if it is a checkpoint, then keep it in memory, except if you're in the second last value, then keep to do backprop

        compute backward pass:
        in the backward passes, the gradients are computed iteratively between each input/output pair using .backward(), but the parameters of the NN aren't updated until we call .step() at the end.
        the backward passes are computed as following: 
        using the last two iterations, do .backward()
        using the last checkpoint, compute up to the third last checkpoint, then do .backward() between the third last and the .backward() just computed.
        continue, and if the last checkpoint is used in a .backward(), then remove it from memory.
        continue until the input value is used in backprop.
        ## then do .step()
        '''
        if checkpoints == []:
            checkpoint_freq = np.floor(np.sqrt(n_iter))
            checkpoints = np.arange(checkpoint_freq, n_iter, checkpoint_freq)
        def new_f(x):
            return self.f(x, y)
        
        checkpoint_xs = []
        
        for i in range(n_iter):

            x = torch.tensor(x.clone().detach(), requires_grad=True).to(device)
            f_val = new_f(x)
            grad_new = torch.autograd.grad(f_val, x)[0]
            x = x.detach()
            if i == 0:
                x_old = x ## x_old and x have the same memory space, so if x changes, x_old changes too
                grad_old = grad_new

            u = torch.cat([x, x_old, grad_new, grad_old], dim=0)
            #u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)
            u = torch.relu(self.fc1(u))
            u = torch.relu(self.fc2(u))
            new_tau = F.softplus(self.fc3(u))
            
            x_old = x.clone() ## x_old and x have different memory space
            grad_old = grad_new.clone()
            
            
            x -= new_tau * grad_new
            
            if (i-1) % checkpoint_freq == 0 or i == n_iter-1: ## get final output too. i=0 -> x_1
                checkpoint_xs.append(x)
            #print(torch.cuda.memory_allocated())
        #print(checkpoints, checkpoint_xs)
        return checkpoint_xs, checkpoints
    
    
    def backward_checkpoint(self, checkpoint_xs, checkpoints, inpt, tau_model, n_iter, y):
        def new_f(x):
            return self.f(x, y)
        
        def get_new_tau(x, x_old, grad_new, grad_old):
            u = torch.cat([x, x_old, grad_new, grad_old], dim=0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)
            u = torch.relu(self.fc1(u))
            u = torch.relu(self.fc2(u))
            new_tau = F.softplus(self.fc3(u))
            return new_tau
        
        def get_new_x_old_x(x, x_old, grad_new, grad_old):
            new_tau = get_new_tau(x, x_old, grad_new, grad_old)
            return x - new_tau * grad_new, x.clone()
        
        #### just considering the loss as a function of x_out, the general case is similar, but just a sum of interemdiates
        # output = checkpoint_xs[-1]
        # output = torch.tensor(output.clone().detach(), requires_grad=True).to(device)
        # f_val = new_f(output)
        # grad_matrix_continue = torch.autograd.grad(f_val, output)[0]
        
        # ### now get the intermediate grads, dx_i/dx_{i-1}
        # for i in range(n_iter-1, 0, -1):
        #     checkpt = checkpoint_xs[-1]
            
        #     x = torch.tensor(x.clone().detach(), requires_grad=True).to(device)
        #     f_val = new_f(x)
        #     grad_new = torch.autograd.grad(f_val, x)[0]
        #     x = x.detach()
        #     if i == 0:
        #         x_old = x ## x_old and x have the same memory space, so if x changes, x_old changes too
        #         grad_old = grad_new

        #     x, x_old = get_new_x_old_x(x, x_old, grad_new, grad_old)

        #     grad_loss_output = torch.autograd.grad(x, x_old)[0]
            
        #     grad_matrix_continue = grad_matrix_continue@grad_loss_output
        
        ## now get final grad, dx_1/d\theta
        inpt = torch.tensor(inpt.clone().detach(), requires_grad=True).to(device)
        f_val_x0 = new_f(inpt)
        grad_input = torch.autograd.grad(f_val_x0, inpt)[0]
        x1 = get_new_x_old_x(inpt, inpt, grad_input, grad_input)
        #grad_x1_theta = torch.autograd.grad(x1, tau_model.parameters())[0]
        grad_x1_theta = []
        print('start')
        i=0
        for x1_element in x1:
            i+=1
            print(i)
            grad_element = torch.autograd.grad(x1_element, tau_model.parameters(), retain_graph=True)
            grad_x1_theta.append(grad_element)
        print(len(grad_x1_theta))
        print(grad_x1_theta.shape)
        
        return
    
       
    
    


class TauModelNoAdjointSmaller(nn.Module):
    def __init__(self, model, reg, in_channels, out_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg = reg
        
        self.model = model

        self.f = lambda x, y: 0.5 * (torch.norm(model(x) - y, p=2).pow(2)) + reg(x)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*64, 64)
        self.fc2 = nn.Linear(64, 1)

    
    def forward(self, x, y, n_iter=1):

        x_list = []
        tau_list = []

        def new_f(x):
            return self.f(x, y)
        
        for i in range(n_iter):

            x = torch.tensor(x.clone().detach(), requires_grad=True).to(device)
            f_val = new_f(x)
            grad_new = torch.autograd.grad(f_val, x)[0]

            if i == 0:
                x_old = x
                grad_old = grad_new

            u = torch.cat([x, x_old, grad_new, grad_old], dim=0)
            #u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)
            u = torch.relu(self.fc1(u))
            new_tau = F.softplus(self.fc2(u))

            df = -new_tau * grad_new
            x = x + df

            x_old = x
            grad_old = grad_new

            x_list.append(x)

            tau_list.append(new_tau)

        return x, tau_list, x_list
    
    
     

    












class AlgoTauModelNoAdjoint(nn.Module):
    def __init__(self, model, reg, in_channels, out_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg = reg
        
        self.model = model

        self.f = lambda x, y: 0.5 * (torch.norm(model(x) - y, p=2).pow(2)) + reg(x)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1, dtype=torch.double)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1, dtype=torch.double)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*64, 64, dtype=torch.double)
        self.fc2 = nn.Linear(64, 64, dtype=torch.double)
        self.fc3 = nn.Linear(64, 1, dtype=torch.double)

    
    def forward(self, x, y, n_iter=1):

        tau_list = []

        new_f = lambda x: self.f(x, y)
        
        f_list = []
        
        for i in range(n_iter):

            x = torch.tensor(x.clone().detach(), requires_grad=True, dtype=torch.double).to(device)
            f_val = new_f(x)
            f_list.append(f_val.detach().cpu().numpy())
            grad_new = torch.autograd.grad(f_val, x, create_graph=True)[0]

            if i == 0:
                x_old = x
                grad_old = grad_new

            u = torch.cat([x, x_old, grad_new, grad_old], dim=0)
            #u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)
            u = torch.relu(self.fc1(u))
            u = torch.relu(self.fc2(u))
            new_tau = F.softplus(self.fc3(u))

            df = -new_tau * grad_new
            x = x + df

            x_old = x
            grad_old = grad_new

            tau_list.append(new_tau.item())
        f_list.append(new_f(x).detach().cpu().numpy())
        return f_list, tau_list
    



class AlgoTauModelNoAdjointX(nn.Module):
    def __init__(self, model, reg, in_channels, out_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg = reg
        
        self.model = model

        self.f = lambda x, y: 0.5 * (torch.norm(model(x) - y, p=2).pow(2)) + reg(x)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1, dtype=torch.double)
        ## This is a very simple one: turn off the bias of layers before BatchNormalization layers. For a 2-D convolutional layer, this can be done by setting the bias keyword to False: torch.nn.Conv2d(..., bias=False, ...)
        
        ## Layer Normalization: Replace batch normalization with layer normalization, which can reduce memory consumption during training.
        
        ## Memory-Efficient Layers: Use memory-efficient layer implementations, such as depthwise separable convolutions in convolutional neural networks (CNNs), which reduce the number of parameters and memory consumption.
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1, dtype=torch.double)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*64, 64, dtype=torch.double)
        self.fc2 = nn.Linear(64, 64, dtype=torch.double)
        self.fc3 = nn.Linear(64, 1, dtype=torch.double)


    
    def forward(self, x, y, n_iter=1):

        tau_list = []

        new_f = lambda x: self.f(x, y)
        
        f_list = [new_f(x).detach().cpu().numpy()]
        
        for i in range(n_iter):

            x = torch.tensor(x.clone().detach(), requires_grad=True, dtype=torch.double).to(device)
            f_val = new_f(x)
            grad_new = torch.autograd.grad(f_val, x, create_graph=True)[0]

            if i == 0:
                x_old = x
                grad_old = grad_new

            u = torch.cat([x, x_old, grad_new, grad_old], dim=0)
            #u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)
            u = torch.relu(self.fc1(u))
            u = torch.relu(self.fc2(u))
            new_tau = F.softplus(self.fc3(u))

            df = -new_tau * grad_new
            x = x + df

            x_old = x
            grad_old = grad_new

            tau_list.append(new_tau.item())
            f_list.append(new_f(x).detach().cpu().numpy())
        f_list.append(new_f(x).detach().cpu().numpy())
        return f_list, tau_list, x
    



class AlgoTauModelNoAdjointSmaller(nn.Module):
    def __init__(self, model, reg, in_channels, out_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg = reg
        
        self.model = model

        self.f = lambda x, y: 0.5 * (torch.norm(model(x) - y, p=2).pow(2)) + reg(x)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1, dtype=torch.double)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1, dtype=torch.double)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*64, 64, dtype=torch.double)
        self.fc2 = nn.Linear(64, 1, dtype=torch.double)

    
    def forward(self, x, y, n_iter=1):

        f_list = []
        tau_list = []

        new_f = lambda x: self.f(x, y)
        
        for i in range(n_iter):

            x = torch.tensor(x.detach(), requires_grad=True, dtype=torch.double).to(device)
            f_val = new_f(x)
            f_list.append(f_val)
            grad_new = torch.autograd.grad(f_val, x, create_graph=True)[0]

            if i == 0:
                x_old = x
                grad_old = grad_new

            u = torch.cat([x, x_old, grad_new, grad_old], dim=0)
            #u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)
            u = torch.relu(self.fc1(u))
            new_tau = F.softplus(self.fc2(u))

            df = -new_tau * grad_new
            x = x + df

            x_old = x
            grad_old = grad_new

            tau_list.append(new_tau)
        f_list.append(new_f(x).detach().cpu().numpy())
        return f_list, tau_list
    










    



class ConstTauModel(nn.Module):
    def __init__(self, adjoint_operator_module, operator_module, reg, in_channels, out_channels, tau=1):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg = reg
        if tau == 1:
            self.tau = nn.Parameter(0.12*torch.ones(1,1,1,1))
        else:
            self.tau = tau
        
        self.operator = operator_module
        self.adjoint_operator = adjoint_operator_module

    
    def forward(self, f, x, y, n_iter=1):

        x_list = []
        tau_list = []
        
        for i in range(n_iter):
        
            #f_sinogram = self.operator(x)
            #adjoint_eval = self.adjoint_operator(f_sinogram - y)
            x = x.requires_grad_(True)
            #new_reg_value = self.reg(x)
            #grad_reg_new = torch.autograd.grad(new_reg_value, x)[0]
            #df = - self.tau * (adjoint_eval + grad_reg_new)
            f_value = f(x, y)
            grad_f_new = torch.autograd.grad(f_value, x)[0]
            df = - self.tau * grad_f_new
            
            x = x + df

            x_list.append(x)
            tau_list.append(self.tau)
        
        return x, tau_list, x_list



class TauModelSample(nn.Module):
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

    
    def forward(self, x, xm1, y, n_iter=1):

        x_list = []
        tau_list = []
        
        for i in range(n_iter):
        
            f_sinogram = self.operator(x)

            adjoint_eval = self.adjoint_operator(f_sinogram - y)

            x = x.requires_grad_(True)
            new_reg_value = self.reg(x)
            grad_reg_new = torch.autograd.grad(new_reg_value, x)[0]

            if i == 0:
                x_old = xm1
                f_sinogram_old = self.operator(x_old)

                adjoint_eval_old = self.adjoint_operator(f_sinogram_old - y)

                xm1 = xm1.requires_grad_(True)
                old_reg_value = self.reg(x)
                grad_reg_old = torch.autograd.grad(old_reg_value, x)[0]

            #print(x.shape, adjoint_eval.shape, x_old.shape, adjoint_eval_old.shape, grad_reg_new.shape, grad_reg_old.shape)
            u = torch.cat([x, adjoint_eval, x_old, adjoint_eval_old, grad_reg_new, grad_reg_old], dim=0)
            #u = u.squeeze(1).unsqueeze(0)
            u = self.pool(torch.relu(self.conv1(u)))
            u = self.pool(torch.relu(self.conv2(u)))
            u = u.view(u.size(0), -1)  # Flatten the tensor
            u = torch.relu(self.fc1(u))
            new_tau = F.softplus(self.fc2(u))

            #u = layers2[i].to(device)(u).to(device)

            #df = -self.step_length *u[:,0:1,:,:]
            df = - new_tau * (adjoint_eval + grad_reg_new)
            
            x = x + df

            x_old = x
            adjoint_eval_old = adjoint_eval
            grad_reg_old = grad_reg_new

            x_list.append(x)
            tau_list.append(new_tau)
        
        return x, tau_list, x_list



    
class BackTrackingX(nn.Module):
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

        print(self.layers)
        
    
    def forward(self, x, y, n_iter):

        #layers2 = [self.layers for i in range(n_iter)]

        x_list = []
        
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

            u = torch.cat([x, adjoint_eval, x_old, adjoint_eval_old, grad_reg_new, grad_reg_old], dim=1)
            u = self.layers.to(device)(u).to(device)
            #u = layers2[i].to(device)(u).to(device)

            #df = -self.step_length *u[:,0:1,:,:]
            df = -self.step_length *(adjoint_eval + grad_reg_new + u[:,0:1,:,:])
            
            x = x + df

            x_old = x
            adjoint_eval_old = adjoint_eval
            grad_reg_old = grad_reg_new

            x_list.append(x)
        
        return x, self.step_length, x_list
    































def double_conv_and_ReLU(in_channels, out_channels):
    list_of_operations = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU()
    ]

    return nn.Sequential(*list_of_operations)

### Class for encoding part of the UNet. In other words, this is the part of
### the UNet which goes down with maxpooling.
class encoding(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels

        self.convs_and_relus1 = double_conv_and_ReLU(self.in_channels, out_channels=32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus2 = double_conv_and_ReLU(in_channels=32, out_channels=64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus3 = double_conv_and_ReLU(in_channels=64, out_channels=128)

    ### Must have forward function. Follows skip connecting UNet architechture
    
    def forward(self, g):
        g_start = g
        encoding_features = []
        g = self.convs_and_relus1(g)
        encoding_features.append(g)
        g = self.maxpool1(g)
        g = self.convs_and_relus2(g)
        encoding_features.append(g)
        g = self.maxpool2(g)
        g = self.convs_and_relus3(g)

        return g, encoding_features, g_start

### Class for decoding part of the UNet. This is the part of the UNet which
### goes back up with transpose of the convolution
class decoding(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        ### Defining instance variables
        self.out_channels = out_channels

        self.transpose1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus1 = double_conv_and_ReLU(in_channels=128, out_channels=64)
        self.transpose2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus2 = double_conv_and_ReLU(in_channels=64, out_channels=32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=(3,3), padding=1)

    ### Must have forward function. Follows skip connecting UNet architechture
    
    def forward(self, g, encoding_features, g_start):
        g = self.transpose1(g)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        encoding_features.pop()
        g = self.convs_and_relus1(g)
        g = self.transpose2(g)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        encoding_features.pop()
        g = self.convs_and_relus2(g)
        g = self.final_conv(g)

        g = g_start + g

        return g

### Class for the UNet model itself
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = encoding(self.in_channels)
        self.decoder = decoding(self.out_channels)

    ### Must have forward function. Calling encoder and deoder classes here
    ### and making the whole UNet model
    
    def forward(self, g):
        g, encoding_features, g_start = self.encoder(g)
        g = self.decoder(g, encoding_features, g_start)

        return g
