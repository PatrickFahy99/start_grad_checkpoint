
import odl
import torch
from odl.contrib.torch import OperatorModule
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from functions import estimate_operator_norm
import time
from huber_TV import power_iteration
from datasets import NaturalDataset, TestDataset, XRayDataset, my_collate
from optimisers import TauFuncNet, TauFunc10Net, TauFuncUnboundedNet, TauFuncUnboundedAboveNet, UpdateModel, \
    UnrollingFunc, GeneralUpdateModel, FixedModel, FixedMomentumModel, AdagradModel, RMSPropModel, AdamModel, CNN_LSTM
from algorithms import gradient_descent_fixed, gradient_descent_backtracking, gradient_descent_const
from torch.utils.data import ConcatDataset
from grad_x import grad, laplacian
import time
import pandas as pd
from torch.nn import HuberLoss
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from LGS_module import LGS, LGS2, BackTrackingTau
from LGS_train_module import LGD2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

alpha = 0.0001

NUM_TEST = 100


def huber(s, epsilon=0.01):
    return torch.where(torch.abs(s) <= epsilon, (0.5 * s ** 2) / epsilon, s - 0.5 * epsilon)

def Du(u):
    """Compute the discrete gradient of u using PyTorch."""
    diff_x = torch.diff(u, dim=1, prepend=u[:, :1])
    diff_y = torch.diff(u, dim=0, prepend=u[:1, :])
    return diff_x, diff_y

def f(x, y, A_func, alpha=0.0001):
    return data_fidelity(x, y, A_func) + alpha * huber_total_variation(x)


def correct_fs(fs):
    return [float(f) for f in fs]


def data_fidelity(x, y, A_func):
    return 0.5 * (torch.linalg.norm(A_func(x) - y) ** 2)


def huber_total_variation(u, eps=0.01):
    diff_x, diff_y = Du(u)
    norm_2_1 = torch.sum(huber(torch.sqrt(diff_x ** 2 + diff_y ** 2 + 1e-08), eps))
    return norm_2_1


def reg(x, alpha):
    return alpha * huber_total_variation(x)

reg_func = lambda x: reg(x, 0.0001)


# Path to the folder containing the images
#folder_path = r'C:\Users\Patrick\XRayL2O\notumor'
folder_path = r'C:\Users\Patrick\XRayL2O\lhq_256'

test_dataset = TestDataset(folder_path, num_imgs=NUM_TEST)

test_set = XRayDataset(test_dataset, alpha, f)

# new_input = 'small_var_noise_4sig'
# fixed_model = torch.load(f'grad_fixed_model{new_input}.pth')
# tau_model = torch.load(f'grad_tau_model_ubd_above{new_input}.pth')
# fixed_heavy_model = torch.load(f'grad_fixed_heavy_model{new_input}.pth')
# hb_func_model = torch.load(f'grad_hb_func_model{new_input}.pth')

# correction_model = torch.load(f'grad_correction_model{new_input}.pth')


# post_correction_model = torch.load(f'grad_post_correction_model{new_input}.pth')

num = 0

num_iters_nonlearn1 = []
execution_times_nonlearn1 = []
fs_nonlearn1 = []


x_stars = torch.load("x_stars.pt")
x_stars_test = torch.load("x_stars_test.pt")
f_images = torch.load("f_images.pt")
g_sinograms = torch.load("g_sinograms.pt")
f_rec_images = torch.load("f_rec_images.pt")
f_test_rec_images = torch.load("f_test_rec_images.pt")
f_test_images = torch.load("f_test_images.pt")
g_test_sinograms = torch.load("g_test_sinograms.pt")

from LGS_train_module import geometry_and_ray_trafo

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


operator_norm = odl.power_method_opnorm(ray_transform)
op_tau_new = 1 / (operator_norm**2 + 8 * 0.0001 / 0.01)

for i in range(len(x_stars)): 
    

    n_index = np.random.permutation(g_sinograms.shape[0])[:1]
    g_batch = g_sinograms[n_index,:,:,:]
    f_batch = f_images[n_index]
    f_batch2 = f_rec_images[n_index]

    num += 1

    print(op_tau_new)

    # print('FIXED NOT LEARNED 1')
    # start_time = time.time()
    # x_star2, taus2, fs2, iters2, scale_nums2 = gradient_descent_fixed(new_f, f_batch2.double().to(device), g_batch.double().to(device),
    #                                                                   op_tau_new, tol=0.005)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # execution_times_nonlearn1.append(execution_time)
    # fs2 = correct_fs(fs2)
    # taus2 = correct_fs(taus2)
    # fs_nonlearn1.append(fs2)
    # num_iters_nonlearn1.append(iters2)

    # print('FIXED NOT LEARNED 1')
    # start_time = time.time()
    # x_star2, taus2, fs2 = gradient_descent_const(new_f, f_batch2.double().to(device), g_batch.double().to(device),
    #                                                                   op_tau_new)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # execution_times_nonlearn1.append(execution_time)
    # fs2 = correct_fs(fs2)
    # taus2 = correct_fs(taus2)
    # fs_nonlearn1.append(fs2)

    print('FIXED LEARNED')
    start_time = time.time()
    new_f = lambda x, y: f(x, y, ray_transform_module)
    # x_star2, taus2 = gradient_descent_fixed(new_f, f_batch2.double().to(device), g_batch.double().to(device),
    #                                                                   0.1639, max_iter=1000)
    x_star2, taus2, fs22 = gradient_descent_const(new_f, f_batch2.double().to(device), g_batch.double().to(device),
                                                                    0.1639)
    end_time = time.time()
    fs22 = [new_f(x, g_batch) for x in x_star2]
    execution_time = end_time - start_time
    execution_times_nonlearn1.append(execution_time)
    taus2 = correct_fs(taus2)
    fs22 = correct_fs(fs22)
    torch.cuda.empty_cache()


    print('FIXED 2/L')
    start_time = time.time()
    x_star2, taus2, fs2l = gradient_descent_const(new_f, f_batch2.double().to(device), g_batch.double().to(device),
                                                                      2*op_tau_new)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_nonlearn1.append(execution_time)
    fs2l = correct_fs(fs2l)
    taus2 = correct_fs(taus2)

    print(2*op_tau_new)

    torch.cuda.empty_cache()

    print('Learned', len(fs22))
    print('2/L', len(fs2l))
    print(fs2l[-1],fs22[-1])
    if min(fs2l)>min(fs22):
        print('bad')
        break
    else:
        print('next one')
        continue

    min_f = np.min([min(fs22), min(fs2l)])
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
    plt.xlabel('Iteration Number')  # Replace with your actual label
    plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    plt.semilogy([i - min_f for i in [i for i in fs2l if i - min_f>1e-04]], label='2/L')
    plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Fixed')
    plt.legend()
    plt.show()

    # print('FIXED SMALL')
    # start_time = time.time()
    # x_star2, taus2, fs2small, iters2, scale_nums2 = gradient_descent_fixed(new_f, f_batch2.double().to(device), g_batch.double().to(device),
    #                                                                   0.01, tol=0.005)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # execution_times_nonlearn1.append(execution_time)
    # fs2small = correct_fs(fs2small)
    # taus2 = correct_fs(taus2)
    # fs_nonlearn1.append(fs2)
    # num_iters_nonlearn1.append(iters2)

    # print('FIXED LARGE')
    # start_time = time.time()
    # x_star2, taus2, fs2large, iters2, scale_nums2 = gradient_descent_fixed(new_f, f_batch2.double().to(device), g_batch.double().to(device),
    #                                                                   0.15, tol=0.005)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # execution_times_nonlearn1.append(execution_time)
    # fs2large = correct_fs(fs2large)
    # taus2 = correct_fs(taus2)
    # fs_nonlearn1.append(fs2)
    # num_iters_nonlearn1.append(iters2)

    print('FIXED LEARNED')
    start_time = time.time()
    x_star2, taus2, fs22, iters2, scale_nums2 = gradient_descent_fixed(new_f, f_batch2.double().to(device), g_batch.double().to(device),
                                                                      0.0626, tol=0.005)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_nonlearn1.append(execution_time)
    fs22 = correct_fs(fs22)
    taus2 = correct_fs(taus2)
    fs_nonlearn1.append(fs2)
    num_iters_nonlearn1.append(iters2)

    torch.cuda.empty_cache()
    


    # #BACKTRACKING

    print('BACKTRACKING')
    start_time = time.time()
    x_star_bt, taus_bt, fs_bt, iters_bt, scale_nums_bt = gradient_descent_backtracking(new_f, f_batch2.double().to(device),
                                                                                       g_batch.double().to(device), tol=1e-08)
    end_time = time.time()
    execution_time = end_time - start_time
    #execution_times_backtracking.append(execution_time)
    fs_bt = correct_fs(fs_bt)
    taus_bt = correct_fs(taus_bt)
    #fs_bt_list.append(fs_bt)
    #num_iters_fixed.append(iters_bt)
    #scale_nums_list_fixed.append(scale_nums_bt)
    #taus_bt_list.append(taus_bt)

    ## LEARNED
    print('LEARNED')
    LGS_net = LGS(lst.A_adj, lst.A_func, g_batch, f_batch2, in_channels=2, out_channels=1, step_length=0.1, n_iter=100).to(device)
    LGS_net.load_state_dict(torch.load('Deep Learning Reconstructions\LGS1_005.pth', map_location=device))

    LGS_net.eval()

    x0_tensor = f_batch2[:, None, :, :]
    y_tensor = g_batch[None, None, :, :]

    # Now, you can use these 4D tensors as inputs to your network
    LGS_reco, _, total_lst = LGS_net(x0_tensor, y_tensor)
    fs_learn = [new_f(x, g_batch) for x in total_lst]
    fs_learn = correct_fs(fs_learn)


    ## LEARNED BY ME
    print('LEARNED BY ME')
    from LGS_train_module import LGD2
    LGS_net = LGD2(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
    LGS_net.load_state_dict(torch.load('CORR_LG_SUPERVISED_GT.pth', map_location=device))

    LGS_net.eval()

    x0_tensor = f_batch2[:, None, :, :]
    y_tensor = g_batch[None, None, :, :]

    # Now, you can use these 4D tensors as inputs to your network
    LGS_reco, _, total_lst = LGS_net(x0_tensor, y_tensor, n_iter=100)
    fs_learn2 = [new_f(x, g_batch) for x in total_lst]
    fs_learn2 = correct_fs(fs_learn2)


    ## LEARNED BY ME - UNSUPERVISED
    print('LEARNED BY ME - UNSUPERVISED')
    LGS_netu = LGD2(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
    LGS_netu.load_state_dict(torch.load('CORR_LG_UNSUPERVISED_GT.pth', map_location=device))

    LGS_netu.eval()

    x0_tensor = f_batch2[:, None, :, :]
    y_tensor = g_batch[None, None, :, :]

    # Now, you can use these 4D tensors as inputs to your network
    LGS_recou, _, total_lstu = LGS_netu(x0_tensor, y_tensor, n_iter=100)
    fs_learn2u = [new_f(x, g_batch) for x in total_lstu]
    fs_learn2u = correct_fs(fs_learn2u)

    plt.semilogy([i for i in fs_learn2], label='SUPERVISED')
    plt.semilogy([i for i in fs_learn2u], label='UNSUPERVISED')
    plt.legend()

    # LEARNED TAU
    print('LEARNED TAU')
    from LGS_train_module import TauModel
    tau_net = TauModel(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
    tau_net.load_state_dict(torch.load('TAU_SUPERVISED.pth', map_location=device))

    # tau_net.eval() 

    x0_tensor = f_batch2[:, None, :, :]
    y_tensor = g_batch[None, None, :, :]

    # Now, you can use these 4D tensors as inputs to your network


    tau_reco, _, total_lst = tau_net(x0_tensor, y_tensor, n_iter=100)
    fs_tau = [new_f(x, g_batch) for x in total_lst]
    fs_tau = correct_fs(fs_tau)

    # SAFEGUARDED TAU MODEL

    print('SAFEGUARDED TAU MODEL')
    ### PROVABLY CONVERGENT IF THE F IS AT MOST WHAT GRADIENT DESCENT GIVES, BUT CHECKING AT EACH ITERATION IS COSTLY? MAYBE NOT
    fs_tau_sg = [new_f(f_batch2, g_batch)]
    taus_sg = []
    xn = x0_tensor
    for i in range(400):
        tau_reco_sg, _, total_lst = tau_net(xn, y_tensor, n_iter=1)
        new_f = new_f(tau_reco_sg, g_batch)
        grad_des_f = new_f(xn, g_batch)
        xn = gradient_descent_fixed(new_f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
        if new_f < grad_des_f:
            fs_tau_sg.append(new_f)
            xn = tau_reco_sg
            taus_sg.append(float(_[0]))
        else:
            fs_tau_sg.append(grad_des_f)
            taus_sg.append(op_tau_new)
    fs_tau_sg = correct_fs(fs_tau_sg)




    print('SAFEGUARDED UNSUPERVISED TAU MODEL')
    from LGS_train_module import TauModel
    tau_net = TauModel(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
    tau_net.load_state_dict(torch.load('TAU_UNSUPERVISED.pth', map_location=device))
    x0_tensor = f_batch2[:, None, :, :]
    y_tensor = g_batch[None, None, :, :]
    ### PROVABLY CONVERGENT IF THE F IS AT MOST WHAT GRADIENT DESCENT GIVES, BUT CHECKING AT EACH ITERATION IS COSTLY? MAYBE NOT
    fs_tau_Usg = [new_f(f_batch2, g_batch)]
    taus_Usg = []
    xn = x0_tensor
    for i in range(400):
        tau_reco_Usg, _, total_lst = tau_net(xn, y_tensor, n_iter=1)
        new_f = new_f(tau_reco_Usg, g_batch)
        grad_des_f = new_f(xn, g_batch)
        xn = gradient_descent_fixed(new_f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
        if new_f < grad_des_f:
            fs_tau_Usg.append(new_f)
            xn = tau_reco_Usg
            taus_Usg.append(float(_[0]))
        else:
            fs_tau_Usg.append(grad_des_f)
            taus_Usg.append(op_tau_new)
    fs_tau_Usg = correct_fs(fs_tau_Usg)


        ## SAFEGUARDED ZONE TAU MODEL

    # print('SAFEGUARDED ZONE TAU MODEL')

    # tau_net_zone = TauModel(lst.A_adj, lst.A_func, reg_func, in_channels=6, out_channels=1).to(device)
    # tau_net_zone.load_state_dict(torch.load('ZONE_CORR_TAU_MODEL.pth', map_location=device))

    # tau_net_zone.eval() 

    # x0_tensor = f_batch2[:, None, :, :]
    # y_tensor = g_batch[None, None, :, :]

    # # Now, you can use these 4D tensors as inputs to your network


    # #tau_reco_zone, _, total_lst_zone = tau_net_zone(x0_tensor, y_tensor, n_iter=100)
    # #fs_tau_zone = [new_f(x, g_batch) for x in total_lst_zone]
    # #fs_tau_zone = correct_fs(fs_tau_zone)

    # fs_tau_sg_zone = [new_f(f_batch2, g_batch)]
    # taus_sg_zone = []
    # xn = x0_tensor
    # for i in range(400):
    #     tau_reco_sg_zone, _, total_lst = tau_net_zone(xn, y_tensor, n_iter=1)
    #     new_f = new_f(tau_reco_sg_zone, g_batch)
    #     if new_f < fs_tau_sg_zone[-1]:
    #         fs_tau_sg_zone.append(new_f)
    #         xn = tau_reco_sg_zone
    #         taus_sg_zone.append(float(_[0]))
    #     else:
    #         xn = gradient_descent_fixed(new_f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
    #         fs_tau_sg_zone.append(new_f(xn, g_batch))
    #         taus_sg_zone.append(op_tau_new)
    # fs_tau_sg_zone = correct_fs(fs_tau_sg_zone)

    # for i in range(400):
    #     tau_reco_sg_zone, _, total_lst = tau_net_zone(xn, y_tensor, n_iter=1)
    #     xn = gradient_descent_fixed(new_f, xn, y_tensor, op_tau_new, max_iter=1)[0][-1]
    #     new_f = new_f(tau_reco_sg_zone, g_batch)
    #     grad_des_f = new_f(xn, g_batch)
    #     if new_f < grad_des_f:
    #         fs_tau_sg_zone.append(new_f)
    #         xn = tau_reco_sg_zone
    #         taus_sg_zone.append(float(_[0]))
    #     else:
    #         fs_tau_sg_zone.append(grad_des_f)
    #         taus_sg_zone.append(op_tau_new)
    # fs_tau_sg_zone = correct_fs(fs_tau_sg_zone)


    # ## BACKTRACKING TAU
    # print('BACKTRACKING TAU')
    # BT_net = BackTrackingTau(lst.A_adj, lst.A_func, reg_func, in_channels=4, out_channels=1).to(device)
    # BT_net.load_state_dict(torch.load('BACKTRACKING_IMITATION_TAU.pth', map_location=device))

    # from LGS_train_module import BackTrackingTauTrain
    # net = BackTrackingTauTrain(lst.A_adj, lst.A_func, reg_func, in_channels=4, out_channels=1).to(device)
    # net.load_state_dict(torch.load('BACKTRACKING_IMITATION_TAU.pth', map_location=device))
    # new_tau = net(f_batch2, f_batch2, g_batch)

    # BT_net.eval()

    # x0_tensor = f_batch2[:, None, :, :]
    # y_tensor = g_batch[None, None, :, :]

    # # Now, you can use these 4D tensors as inputs to your network
    # BT_reco, _, total_lst_BT = BT_net(f_batch2, g_batch, n_iter=100)
    # fs_bt_tau = [new_f(x, g_batch) for x in total_lst_BT]
    # fs_bt_tau = correct_fs(fs_bt_tau)

    # plt.semilogy([i for i in fs_tau_sg], label='SUPERVISED')
    # plt.semilogy([i for i in fs_tau_Usg], label='UNSUPERVISED')
    # plt.legend()
    # plt.show()

    ###### UNSUPERVISED PERFORMS BETTER!!!!!!!!!!!!!!!!!!
    fs_learn2u

        #plt.imshow(g_batch.squeeze(0).cpu().detach().numpy())
    min_f = np.min([np.min(fs2), np.min(fs22), np.min(fs_bt)])
    plt.semilogy([i - min_f for i in fs2], label='1/L')
    plt.semilogy([i - min_f for i in fs22], label='Learned Fixed')
    plt.semilogy([i - min_f for i in fs_learn2], label='Learned Correction Supervised')
    plt.semilogy([i - min_f for i in fs_learn2u], label='Learned Correction Unsupervised')
    plt.semilogy([i - min_f for i in fs_bt], label='Backtracking')
    plt.semilogy([i - min_f for i in fs_tau], label='Learned Tau')
    plt.semilogy([i - min_f for i in fs_tau_sg], label='Learned Tau Supervised Safeguarded')
    plt.semilogy([i - min_f for i in fs_tau_Usg], label='Learned Tau Unsupervsied Safeguarded')
    plt.legend()
    plt.show()


    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
    plt.xlabel('Iteration Number')  # Replace with your actual label
    plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    plt.semilogy([i - min_f for i in [i for i in fs2 if i - min_f>1e-04]], label='1/L')
    plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Fixed')
    plt.legend()
    plt.show()


    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
    plt.xlabel('Iteration Number')  # Replace with your actual label
    plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    plt.semilogy([i - min_f for i in [i for i in fs2 if i - min_f>1e-04]], label='1/L')
    plt.semilogy([i - min_f for i in [i for i in fs2l if i - min_f>1e-04]], label='2/L')
    plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Fixed')
    plt.legend()
    plt.show()




    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\phi$')
    plt.xlabel('Iteration Number')  # Replace with your actual label
    plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    plt.semilogy([i - min_f for i in [i for i in fs2small if i - min_f>1e-04]], label='Large Step Size')
    plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Fixed')
    plt.semilogy([i - min_f for i in [i for i in fs2large if i - min_f>1e-04]], label='Small Step Size')
    plt.legend()
    plt.show()



    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title("Learned $\phi$ function vs Iteration Number")
    plt.xlabel("Iteration Number")
    plt.ylabel("$\phi$")
    plt.scatter(range(len(taus_Usg)), taus_Usg, marker='.', color='red')
    plt.grid(True)
    plt.show()


    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(r'semilogy plot of $f(x_k) - f(x^*)$')
    plt.xlabel('Iteration Number')  # Replace with your actual label
    plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    plt.semilogy([i - min_f for i in [i for i in fs22 if i - min_f>1e-04]], label='Learned Constant')
    plt.semilogy([i - min_f for i in fs_tau_Usg], label='Learned Tau Function Safeguarded')
    plt.legend()
    plt.show()


    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(r'semilogy plot of $f(x_k) - f(x^*)$')
    plt.xlabel('Iteration Number')  # Replace with your actual label
    plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    plt.semilogy([i - min_f for i in fs_tau], label='Learned Tau Function Safeguarded')
    plt.legend()
    plt.show()

    fs_learn2u.insert(0, fs2[0])
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(r'semilogy plot of $f(x_k) - f(x^*)$')
    plt.xlabel('Iteration Number')  # Replace with your actual label
    plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    plt.semilogy([i - min_f for i in fs_learn2u[:100]], label='Learned Correction Unsupervised')
    plt.semilogy([i - min_f for i in fs_tau_Usg[:100]], label='Learned Tau Function Safeguarded')
    plt.legend()
    plt.show()



    plt.semilogy([i - min_f for i in fs2[:6]], label='1/L')
    plt.semilogy([i - min_f for i in fs22[:6]], label='Learned Fixed')
    plt.semilogy([i - min_f for i in fs_learn[:6]], label='Learned')
    plt.semilogy([i - min_f for i in fs_learn2[:6]], label='Learned by me')
    plt.semilogy([i - min_f for i in fs_bt[:6]], label='Backtracking')
    plt.semilogy([i - min_f for i in fs_bt_tau[:6]], label='Backtracking Imitation')
    plt.legend()
    plt.show()



    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\tau$')
    plt.xlabel('Iteration Number')  # Replace with your actual label
    plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    plt.semilogy([i - min_f for i in fs_learn], label=r'Learned $\tau$, u', marker='s', markersize=5, linestyle='-')
    plt.semilogy([i - min_f for i in fs_bt[:len(fs_bt)]], label=r'Backtracking Line Search', marker='o', markersize=5, linestyle='-')
    plt.semilogy([i - min_f for i in fs22[:len(fs_bt)]], label=r'$\tau$ learned', marker='s', markersize=5, linestyle='-')
    plt.legend()
    plt.show()


    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title(r'semilogy plot of $f(x_k) - f(x^*)$. 1/L vs learned $\tau$')
    plt.xlabel('Iteration Number')  # Replace with your actual label
    plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
    plt.semilogy([i - min_f for i in fs2], label=r'$\tau$ = 1/L', marker='o', markersize=5, linestyle='-')
    plt.semilogy([i - min_f for i in fs22], label=r'$\tau$ learned', marker='s', markersize=5, linestyle='-')
    plt.legend()
    plt.show()

    break

