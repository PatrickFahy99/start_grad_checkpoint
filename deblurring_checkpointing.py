### Importing packages and modules
#import odl
import torch
import torch.nn as nn
import torch.optim as optim
#from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from LGS_train_module import get_images, TauModelNoAdjointBatchNorm
#import matplotlib.pyplot as plt
#import time
from torchvision.transforms import GaussianBlur
#from datasets import Blur2Dataset, ImageDataBlur2
from torch.utils.checkpoint import checkpoint_sequential

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


tau_network =  TauModelNoAdjointBatchNorm(model, lambda x: reg(x, alpha), in_channels=4, out_channels=1).to(device)

### Getting model parameters
tau_parameters = list(tau_network.parameters())

print(sum(p.numel() for p in tau_network.parameters() if p.requires_grad))


loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

### Setting up some lists used later
running_loss = []
running_test_loss = []


'''
stop exploding or vanishing gradients:

1. activation functions such as ReLU
2. batch normalization
3. gradient clipping
4. use LSTM or GRU as the recurrent units instead of simple RNNs
dropout?
'''

### Defining training scheme
def train_network(net, ys, n_train=int(1e10), batch_size=1):

    number_of_starting_iter_batches = 0
    print('TRAINING STARTED')

    optimizer = optim.AdamW(tau_parameters, lr=0.001) #betas = (0.9, 0.99)
    ## reduced lr to stop nan
    num_iterations_in_backprop = 10
    train_iters_each_iteration_batch = 1000

    grad_accumuluation_num = 1
    
    # Automatic Mixed Precision training to PyTorch. The main idea here is that certain operations can be run faster and without a loss of accuracy at semi-precision (FP16) rather than in the single-precision (FP32) used elsewhere. AMP, then, automatically decide which operation should be executed in which format. This allows both for faster training and a smaller memory footprint.
    scaler = torch.cuda.amp.GradScaler()
    #  This enables the cudNN autotuner which will benchmark a number of different ways of computing convolutions in cudNN and then use the fastest method from then on.
    torch.backends.cudnn.benchmark = True
    
    
    ## Note .backward() removes computational graph
    ## why do we need optimizer.zero_grad(set_to_none=True)? what it does is 

    
    
    ## can use gradient accumulation when the only part we are training is the last new group, otherwise we train over all groups!
    # note that gradient accumulation also increases memory requirement

    for i in range(n_train): 


        if i%train_iters_each_iteration_batch == 0:
            #optimizer = optim.AdamW(tau_parameters, lr=0.001) 
            optimizer.zero_grad(set_to_none=True)
            number_of_starting_iter_batches += 1
            print(f'NUMBER OF MULTIPLES OF {num_iterations_in_backprop}:', number_of_starting_iter_batches)

        n_index = int(np.random.permutation(len(ys))[:batch_size])
        g_batch = ys[n_index].unsqueeze(0).float()
        
        net.train()
        optimizer.zero_grad(set_to_none=True)
        

        inpt = 0*g_batch.float()
        fs_lst = [float(f(inpt, g_batch, model, alpha))]
        tau_lst = []
        for _ in range(number_of_starting_iter_batches):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                checkpoint_xs, checkpoints = net.forward_checkpoint(inpt, g_batch, n_iter= num_iterations_in_backprop)
                out = net.backward_checkpoint(checkpoint_xs, checkpoints, inpt, net, n_iter= num_iterations_in_backprop, y=g_batch)
                loss = sum([f(out, g_batch, model, alpha) for out in outs_list])/grad_accumuluation_num
                scaler.scale(loss).backward()

            if i % grad_accumuluation_num == 0: ## gradient accumulation
                torch.nn.utils.clip_grad_norm_(tau_parameters, max_norm=1.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            inpt = outs
            fs_lst.append(float(f(inpt, g_batch, model, alpha)))
            tau_lst.append([float(i) for i in tau])
            
            outs_list = None
            outs = None
            tau = None
            
            ## how much memory is being used by tensors:
            #print(torch.cuda.memory_allocated())

        ### Here starts the running tests
        if i % (train_iters_each_iteration_batch//10) == 0 and i>0:

            print(i)

            print('f INPUTS:', fs_lst)
            print('TAUS', tau_lst)
            
            train_loss = loss.item()

            running_loss.append(train_loss)


            print('LOSS:', train_loss)#, test_loss)

        if torch.isnan(loss):
            print('NAN')
            break

        if i % (train_iters_each_iteration_batch//10) == 0:
            if number_of_starting_iter_batches == 0:
                torch.save(net.state_dict(), 'models/BLURRING_TAU_UNSUPERVISED.pth')
                print('saved')      
            else:
                torch.save(net.state_dict(), f'models/BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_{num_iterations_in_backprop}_{number_of_starting_iter_batches}_{train_iters_each_iteration_batch}_ALL.pth')
                print('saved')
                print(f'models/BLURRING_TAU_UNSUPERVISED_MULTIPLE_FIVES_{num_iterations_in_backprop}_{number_of_starting_iter_batches}_{train_iters_each_iteration_batch}_ALL.pth')

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, net = train_network(tau_network, ys)



### https://aman.ai/primers/ai/grad-accum-checkpoint/#:~:text=Gradient%20checkpointing%20is%20a%20technique%20used%20to%20trade,activations%20for%20computing%20gradients%20during%20the%20backward%20pass.

### https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9

## checkpointing works by trading compute for memory. 
# Instead of storing all intermediate activations of the entire network, we recomputed them during the backward pass. 
# This allows us to reduce the memory consumption of the network by trading it for compute.




# what mine does is computes the forward pass for a fixed number of iterations, keeping all in memory, then does the backwards pass. Then continues to the next iteration.
# Let m be the fixed num of iterations, and T total number of iterations: mn, then in total it is O(m) memory and O(T) compute.

## Consider a number T of iterations.
# memory-poor strategy computes the forward pass of all iterations, forgetting all intermediate steps, then computed backward pass at the end, by recomputing all intermediate steps (which are then forgotten). 
# This is O(1) memory and O(T^2) compute.

# gradient checkpointing strategy: save some intermediate results, e.g. if the network consists of T iterations, 
# place checkpoints every sqrt(T) iterations (most memory-efficient strategy if you require that any node is computed at most twice). 
# This is O(sqrt(T)) memory and O(T) compute.

## The issue with my greedy one is that it is computing the gradient and updateting based on partial information, e.g. updating with respect to just the first iteration's information,
## when we could/should be considering all iterations' information. This may mean it performs worse.

## Can we do a trade-off, choosing a good m, and then doing e.g. gradient checkpointing in the backwards pass? Then repeat this T/m times, so this would be O() memory and O() compute.

## also, how to choose the best number of steps within gradient checkpointing? 
## memory poor is lowest memory, but highest compute time.  sqrt(T) may be below memory budget, can we choose higher than sqrt(T),
# which will reduce compute time while remaining below memory budget?  

## Building block of DP solution is the algorithm which keeps the first activation in memory and computes the target backprop as fast as possible with memory budget M.




## To see how it breaks into smaller parts, suppose the set of nodes checkpointed by this algorithm contained node i. Then this algorithm could be decomposed into parts Left/Right as follows:
# Left:
# 1. Given A0, compute Ai with budget M.
# Right:
# 2. Given Ai, Bn, compute Bi with budget M-M0. M0 is memory cost of first activation which needs to be subtracted since “Left” is keeping the first activation in memory.
# Left:
# 3. Given A0, Bi, Compute B0 with budget M
# If the memory budget is too small to save any nodes, then there’s only choice of strategy — the O(n²) memory poor strategy. This is the base case for divide and conquer.



## instead of recomputing, can we save tensors to SDD, and load in, or does the I/O take too long so we're better off recomputing?

## checkpointing:
'''
if k % n_checkpoint == 0:
    ## store the current state
TORCH.OPTIM.OPTIMIZER.ZERO_GRAD()
'''







# suppose we have a computation graph between two tensors, labelled one and two, and compute .backward(). Suppose now we have another tensor, labelled zero, which lies right before the computational graph mentioned above, and the tensor two is no longer available. How can we compute .backward() using zero and one, but get .backward() for zero, one and two in total? Suppose we have an RNN R, then one = R(zero) and two = R(one)


## what we want to do
## In an RNN model. Before we start:
## save first input to memory
## compute forward pass iteration by iteration, if last iteration value (which input to this iter) is not a checkpoint, then remove it from memory
## if it is a checkpoint, then keep it in memory, except if you're in the second last value, then keep to do backprop

## compute backward pass:
## in the backward passes, the gradients are computed iteratively between each input/output pair using .backward(), but the parameters of the NN aren't updated until we call .step() at the end.
## the backward passes are computed as following: 
## using the last two iterations, do .backward()
## using the last checkpoint, compute up to the third last checkpoint, then do .backward() between the third last and the .backward() just computed.
## continue, and if the last checkpoint is used in a .backward(), then remove it from memory.
## continue until the input value is used in backprop.
## then do .step()





# loss_history = []
#         for chunk_start in range(0, len(inputs), self.checkpoint_steps):
#             steps = min(self.checkpoint_steps, len(inputs) - chunk_start)
#             inner_losses, *flat_maml_state = checkpoint(_maml_internal, torch.as_tensor(steps), *flat_maml_state)
#             loss_history.extend(inner_losses.split(1))

#         step_index, final_trainable_parameters, final_optimizer_state = \
#             nested_pack(flat_maml_state, structure=initial_maml_state)
#         final_model = copy_and_replace(
#             self.model, dict(zip(parameters_to_copy, final_trainable_parameters)), parameters_not_to_copy)
#         return self.Result(final_model, loss_history=loss_history, optimizer_state=final_optimizer_state)
    
    
    
    

# efficient_maml = torch_maml.GradientCheckpointMAML(
#     model, compute_loss, optimizer=optimizer, checkpoint_steps=5)

# updated_model, loss_history, _ = efficient_maml(inputs, loss_kwargs={'device':device},
#                                                 max_grad_grad_norm=max_grad_grad_norm)
# final_loss = compute_loss(updated_model, (x_batch, y_batch), device=device)
# final_loss.backward()

