### Importing packages and modules
import odl
import torch
import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from LGS_train_module import get_images, geometry_and_ray_trafo, LGD
import matplotlib.pyplot as plt

### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Using function "get_images" to import images from the path.
images = get_images(r'C:\Users\Patrick\XRayL2O\lhq_256', amount_of_images=100, scale_number=2)
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
percentage = 0.05

### Adding Gaussian noise to the sinograms. Here some problem solving is
### needed to make this smoother.
for k in range(np.shape(sinograms)[0]):
    sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, sinogram_k.std(), sinogram_k.shape) * percentage
    noisy_sinogram = sinogram_k + noise
    noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)

### Using FBP to get reconstructed images from noisy sinograms
rec_images = fbp_operator_module(noisy_sinograms)

### All the data into same device
sinograms = sinograms[:,None,:,:].to(device)
noisy_sinograms = noisy_sinograms[:,None,:,:].to(device)
rec_images = rec_images[:,None,:,:].to(device)
images = images[:,None,:,:].to(device)

f_images = images[0:n_train]
g_sinograms = noisy_sinograms[0:n_train]
f_rec_images = rec_images[0:n_train]

test_images = get_images(r'C:\Users\Patrick\XRayL2O\lhq_256', amount_of_images=10, scale_number=2)

test_images = np.array(test_images, dtype='float32')
test_images = torch.from_numpy(test_images).float().to(device)

#list_of_test_images = list(range(0,363,5))

test_sinograms = ray_transform_module(test_images)

test_noisy_sinograms = torch.zeros((test_sinograms.shape[0], ) + output_shape)
test_rec_images = torch.zeros((test_sinograms.shape[0], ) + shape)

for k in range(np.shape(test_sinograms)[0]):
    test_sinogram_k = test_sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, test_sinogram_k.std(), test_sinogram_k.shape) * percentage
    test_noisy_sinogram = test_sinogram_k + noise
    test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)
                                                  

test_rec_images = fbp_operator_module(test_noisy_sinograms)
    
test_sinograms = test_sinograms[:,None,:,:].to(device)
test_noisy_sinograms = test_noisy_sinograms[:,None,:,:].to(device)
test_rec_images = test_rec_images[:,None,:,:].to(device)
test_images = test_images[:,None,:,:].to(device)

# indices = np.random.permutation(test_rec_images.shape[0])[:75]
f_test_images = test_images#[list_of_test_images]
g_test_sinograms = test_noisy_sinograms#[list_of_test_images]
f_test_rec_images = test_rec_images#[list_of_test_images]


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

def data_fidelity(x,y,operator):
    return 0.5 * (torch.norm(operator(x) - y, p=2) ** 2)


### Setting UNet as model and passing it to the used device
#LGS_network = LGD(adjoint_operator_module, ray_transform_module, lambda x: reg(x, 0.25), in_channels=2, out_channels=1, step_length=0.005, n_iter=5).to(device)

### Getting model parameters
#LGS_parameters = list(LGS_network.parameters())

### Defining PSNR function
def psnr(loss):
    psnr = 10 * np.log10(1.0 / loss+1e-10)
    return psnr

new_f = lambda x,y : f(x,y, ray_transform_module)

#test_rec_images = [(test_rec_images[k,:,:,:]-torch.min(test_rec_images[k,:,:,:]))/(torch.max(test_rec_images[k,:,:,:])-torch.min(test_rec_images[k,:,:,:])) for k in range(test_rec_images.shape[0])]



# # Range of regularization strengths to evaluate
# alphas = [0.0001,0.0002,0.0003,.0004,0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]

# # Lists to store logarithmic values
# regularization_terms = []
# data_fit_terms = []
# from algorithms import gradient_descent_fixed
# # Evaluate L-curve
# for alpha in alphas:
#     print(alpha)

#     reg_func = lambda x: reg(x, alpha)
#     new_f = lambda x,y : f(x,y, ray_transform_module, alpha=alpha)

#     [float(data_fidelity(f_images[0,:,:,:],noisy_sinograms[0,:,:,:],ray_transform_module))], [float(reg_func(f_images[0,:,:,:]))]

#     x_star, taus, fs, iters, scale_nums = gradient_descent_fixed(new_f,rec_images[0,:,:,:], noisy_sinograms[0,:,:,:], 0.1, tol=0.0001)


#     data_fit_term = data_fidelity(x_star, noisy_sinograms[0,:,:,:],ray_transform_module)
#     regularization_term = reg_func(x_star)

#     regularization_terms.append(float(regularization_term))
#     data_fit_terms.append(float(data_fit_term))

# # Create L-curve plot
# plt.scatter(data_fit_terms, regularization_terms, marker='o')
# plt.scatter([float(data_fidelity(f_images[0,:,:,:],noisy_sinograms[0,:,:,:],ray_transform_module))], [float(reg_func(f_images[0,:,:,:]))], marker='o', color='red')
# plt.xlabel("Data Fit Term")
# plt.ylabel("Regularisation Term")
# plt.title("L-Curve")
# plt.grid(True)

# # Identify balance point (usually visually)
# # For this example, you can manually identify the corner point on the plot

# plt.show()


from tqdm import tqdm
from algorithms import gradient_descent_backtracking_all
print('get_mins')
x_list_all_test = []
tau_list_all_test = []
for i in tqdm(range(test_rec_images.shape[0])):
    print(i)
    xs, taus_test, fs, num, scale_nums, all_xs_test = gradient_descent_backtracking_all(new_f, test_rec_images[i,:,:,:], test_noisy_sinograms[i,:,:,:], tol=1e-06)
    x_list_all_test.append(all_xs_test)
    tau_list_all_test.append(taus_test)
    #print(0.5 * (torch.norm(ray_transform_module(output[0]) - test_noisy_sinograms[i,:,:,:], p=2) ** 2))
print('get_f_mins')
#x_stars_test = [i[0] for i in x_stars_test]
#fx_star_test = [f(x_stars_test[k], test_noisy_sinograms[k,:,:,:], ray_transform_module) for k in range(test_rec_images.shape[0])]


print('get_mins')
x_list_all = []
tau_list_all = []
for i in tqdm(range(f_rec_images.shape[0])):
    print(i)
    xs, taus, fs, num, scale_nums, all_xs = gradient_descent_backtracking_all(new_f, f_rec_images[i,:,:,:], g_sinograms[i,:,:,:], tol=1e-06)
    x_list_all.append(all_xs)
    tau_list_all.append(taus)
    #print(0.5 * (torch.norm(ray_transform_module(output[0]) - test_noisy_sinograms[i,:,:,:], p=2) ** 2))


# Save the list of tensors to the specified file
torch.save(x_list_all, "x_list_bt.pt")
torch.save(tau_list_all, "tau_list_bt.pt")
torch.save(x_list_all_test, "x_list_test_bt.pt")
torch.save(tau_list_all_test, "tau_list_test_bt.pt")
torch.save(f_images, "f_images_bt.pt")
torch.save(g_sinograms, "g_sinograms_bt.pt")
torch.save(f_rec_images, "f_rec_images_bt.pt")
torch.save(f_test_rec_images, "f_test_rec_images_bt.pt")
torch.save(f_test_images, "f_test_images_bt.pt")
torch.save(g_test_sinograms, "g_test_sinograms_bt.pt")

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

