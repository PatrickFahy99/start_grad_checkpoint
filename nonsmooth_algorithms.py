import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def function_evals(f, xs, y,  wk_list, f_x_star=0):
    f_x0 = f(xs[0],y)
    fs = [((f(x,y)-f_x_star)/(f_x0 - f_x_star)) for x in xs]
    if wk_list == [-1]:
        wk_list = [0] * len(xs)
        wk_list[-1] = 1
    if len(wk_list) == 1:
        wk_list = [wk_list[0]] * len(xs)
        for k in range(len(wk_list)):
            wk_list[k] = wk_list[k]**(k)
    objective = 0
    for k in range(len(xs)):
        objective += wk_list[k]*((f(xs[k],y)-f_x_star)/(f_x0 - f_x_star))
    obj = objective/len(xs)
    return obj, fs





def gradient_descent_update(grad_data_fit, reg, x0, y, update_model, sig, std, iters):
    xo = x0
    xs = [xo]
    num = 1
    while (num <= iters):
        if num > 2:

            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device),
                                  torch.full((128,128), float(reg(xs[-3]))).to(device)))

        elif num == 1:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device)))
        else:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device)))


        input = input.view(1, 7, 1, x0.shape[0], x0.shape[1])
        update = update_model(input)#update_model(input, sig, std)
        xn = xo - update
        xs.append(xn)
        xo = xn
        num += 1

    return xs

def gradient_descent_modelfree(grad_data_fit, reg, x0, y, model, sig, std, iters):
    xo = x0
    xs = [xo]
    num = 1
    while (num <= iters):
        if num > 2:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device),
                                  torch.full((128,128), float(reg(xs[-3]))).to(device)))

        elif num == 1:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device)))
        else:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device)))


        input = input.view(1, 7, 1, x0.shape[0], x0.shape[1])
        #input = input.view(1, 4, x0.shape[0], x0.shape[1])
        xn = model(input)#model(input, sig, std)
        xs.append(xn)
        num += 1
    return xs


def gradient_descent_correctionNN(grad_data_fit, reg, x0, y, model, sig, std, iters):
    xo = x0
    xs = [xo]
    taus = []
    num = 1
    while (num <= iters):
        if num > 2:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device),
                                  torch.full((128,128), float(reg(xs[-3]))).to(device)))

        elif num == 1:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device)))
        else:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device)))


        input = input.view(1, 7, 1, x0.shape[0], x0.shape[1])
        correction, tau = model(input)
        correction = correction.squeeze(0).squeeze(0)
        xn = xo - tau * grad_data_fit(xo, y).to(device) + correction
        xs.append(xn)
        taus.append(tau)
        xo = xn
        num += 1
    return xs, taus



def PGD(grad_data_fit, reg, x0, y, model, tau_model, sig, std, iters):
    xs = [x0]
    taus = []
    num = 1
    while (num <= iters):

        if num > 2:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device),
                                  torch.full((128,128), float(reg(xs[-3]))).to(device)))

        elif num == 1:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device)))
        else:
            input = torch.concat((grad_data_fit(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  grad_data_fit(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                  xs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                  torch.full((128,128), float(reg(xs[-1]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device),
                                  torch.full((128,128), float(reg(xs[-2]))).to(device)))


        tau_input = input.view(1, 7, 1, x0.shape[0], x0.shape[1])

        tau = tau_model(tau_input).to(device)

        tau_full = torch.full((128, 128), tau.item()).view(1, 1, x0.shape[0], x0.shape[1]).to(device)
        inpt1 = (xs[-1] - tau*grad_data_fit(xs[-1],y)).view(1, 1, x0.shape[0], x0.shape[1]).to(device)
        input = torch.concat((inpt1,  tau_full)).to(device)
        correction = model(input.view(1, 2, 1, x0.shape[0], x0.shape[1]))
        xn = correction
        xs.append(xn)
        taus.append(tau)
        num += 1
    return xs, taus


