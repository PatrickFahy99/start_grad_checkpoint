### Necessary functions needed for "LGS_train.py" to work.

### Needed packages: -odl
###                  -PyTorch
###                  -NumPy
###                  -os
###                  -OpenCv
###


### Importing packages
import os
import cv2 as cv
import numpy as np
import odl
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

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
        for name in all_image_names:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            # Convert the image to grayscale
            grayscale_image = cv.cvtColor(temp_image, cv.COLOR_BGR2GRAY)
            image = grayscale_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
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
        angles = odl.uniform_partition(0, 2*np.pi, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(1028/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (360,int(1028/factor_lines))
    elif setup == 'sparse':
        angle_measurements = 100
        line_measurements = int(512/factor_lines)
        angles = odl.uniform_partition(0, 2*np.pi, angle_measurements)
        lines = odl.uniform_partition(-1*np.pi, np.pi, line_measurements)
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (angle_measurements, line_measurements)
    elif setup == 'limited':
        starting_angle = 0
        final_angle = np.pi * 3/4
        angles = odl.uniform_partition(starting_angle, final_angle, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(512/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (int(360), int(512/factor_lines))
        
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
    def __init__(self, adjoint_operator_module, operator_module, reg_func, in_channels, out_channels, step_length, n_iter):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.step_length = step_length
        self.step_length = nn.Parameter(0.0 * torch.ones(1, 1, 1, 1))
        self.n_iter = n_iter
        self.reg = reg_func
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
        
        self.layers = nn.Sequential(*LGD_layers)
        self.layers2 = [self.layers for i in range(n_iter)]

        ### Initializing the parameters for every unrolled iteration
        #self.layers2 = [nn.Sequential(*LGD_layers) for i in range(n_iter)]
        
    def forward(self, f_rec_images, g_sinograms): ## self, x0, y
        all_lst = []

        for i in range(self.n_iter):

            #print('shape:', f_rec_images.shape)
        
            f_sinogram = self.operator(f_rec_images)

            adjoint_eval = self.adjoint_operator(f_sinogram - g_sinograms) ## is this just adjoint_operator(noise)?

            f_rec_images = f_rec_images.requires_grad_(True)
            new_reg_value = self.reg(f_rec_images)
            grad_reg_new = torch.autograd.grad(new_reg_value, f_rec_images)[0]

            #u = torch.cat([f_rec_images, adjoint_eval, grad_reg_new], dim=1)


            second_input = adjoint_eval + grad_reg_new
            u = torch.cat([f_rec_images.to(device), adjoint_eval.to(device), second_input.to(device)], dim=1)

            
            u = self.layers2[i].to(device)(u.to(device))

            df = -(F.relu(self.step_length)) * u[:,0:1,:,:]
            #df = -self.step_length * second_input.to(device)
            
            f_rec_images = f_rec_images + df

            all_lst.append(f_rec_images)
        
        return all_lst, self.step_length#f_rec_images, self.step_length
    




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
    

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=a, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * (b // 2) * (c // 2), 128)  # Adjust the input size based on pooling
        self.fc2 = nn.Linear(128, 1)  # Output a 1D scalar

    def forward(self, x):
        # Convolutional layers with ReLU activation and max-pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape for fully connected layers
        x = x.view(-1, 32 * (b // 2) * (c // 2))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Output a 1D scalar
        return x.squeeze()

