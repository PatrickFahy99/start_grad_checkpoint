import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def function_evals(f, xs, y,  wk_list, f_x_star=0):
    f_x0 = f(xs[0],y)
    xs = xs[1:]
    if len(xs) == 0:
        return f_x0/f_x0, []
    fs = [f(x,y) for x in xs]#[((f(x,y)-f_x_star)/(f_x0 - f_x_star)) for x in xs]
    if wk_list == [-1]:
        wk_list = [0] * len(xs)
        wk_list[-1] = len(xs)
    if len(wk_list) == 1:
        wk_list = [wk_list[0]] * len(xs)
        for k in range(len(wk_list)):
            wk_list[k] = wk_list[k]**(k)
    objective = 0
    for k in range(len(xs)):
        objective += wk_list[k]*((f(xs[k],y)-f_x_star)/(f_x0 - f_x_star))
    obj = objective/len(xs)
    return obj, fs


def gradient_descent_fixed(f, x0, y, tau_in, iters=10, tol=1, max_iter=10000):
    tau = tau_in
    xo = x0.clone().detach()
    xs = [xo]
    fs = [f(xo, y).detach().cpu().numpy()]
    taus = []
    grad_fs = []
    num = 1
    scale_num = 0
    res = 0
    new_f = lambda x: f(x,y)
    scale_nums = []
    scale_tau=False

    xo = xo.requires_grad_(True)  # Set requires_grad to True
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]
    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients

    if tol == 1:
        while (num <= max_iter):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
            grad_fs.append(grad_f_new)
            xn = xo - tau * grad_f_new.to(device)
            xs.append(xn)
            taus.append(tau)
            xo = xn
            num += 1
        return xs, taus
    else:
        while torch.norm(grad_f_new) > tol:
            #print(torch.norm(grad_f_new))
            if max_iter:
                if num >= max_iter:
                    #print('Max it reached')
                    #print('Norm', torch.norm(grad_f_new))
                    #print('Iters:', num)
                    #print(scale_num)
                    return xn, taus, fs, num, scale_nums
            go = torch.norm(grad_f_new)
            if not scale_tau:
                tau = tau_in#.to(device)
            xn = xo.to(device) - tau * grad_f_new.to(device)
            #print(tau)

            #new_maxi = torch.max(tau * grad_f_new.to(device))
            #if new_maxi<1e-05:
            #    print(new_maxi)

            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]  # Compute gradients explicitly
            xn = xn.clone().detach()

            gn = torch.norm(grad_f_new)
            if go == gn:
                #print(f"Iteration {num}, Initial Norm: {go:.12f}")
            #    print('Equal grads')
                print('ITERS:', num)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn,y)>f(xo,y):
                # print('REDUCE', num)
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                scale_tau = True
                #print("Scaling Tau")
                if res>1000:
                    print('too many reductions')
                    #print('No change for a while')
                    #print('ITERS:', num)
                    #print('Norm', torch.norm(grad_f_new))
                    print(scale_num)
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                #print(f_value)
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                xo = xn.clone().detach()
                num += 1
    #print('ITERS:', num)
    #print('Norm', torch.norm(grad_f_new))
    print('ITERS:', num)
    print(scale_num)
    return xn, taus, fs, num, scale_nums




def gradient_descent_const(f, x0, y, tau_in):
    tau = tau_in
    xo = x0.clone().detach()
    xs = [xo]
    fs = [f(xo, y).detach().cpu().numpy()]
    taus = []
    grad_fs = []
    num = 1
    new_f = lambda x: f(x,y)

    xo = xo.requires_grad_(True)  # Set requires_grad to True
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]
    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients
    f_new = 0
    f_old = 1
    while f_new < f_old:
        f_old = new_f(xo)
        xo.requires_grad_(True)
        f_value = new_f(xo)
        grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
        grad_fs.append(grad_f_new)
        xn = xo - tau * grad_f_new.to(device)
        xs.append(xn)
        taus.append(tau)
        fs.append(new_f(xn).detach().cpu().numpy())
        xo = xn
        num += 1
        f_new = new_f(xn)
    return xs, taus, fs



def gradient_descent_fixed_all(f, x0, y, tau_in, iters=10, tol=1, max_iter=10000):
    tau = tau_in
    xo = x0.clone().detach()
    xs = [xo]
    fs = [f(xo, y).detach().cpu().numpy()]
    taus = []
    grad_fs = []
    num = 1
    scale_num = 0
    res = 0
    new_f = lambda x: f(x,y)
    scale_nums = []
    scale_tau=False

    xo = xo.requires_grad_(True)  # Set requires_grad to True
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]
    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients

    if tol == 1:
        while (num <= max_iter):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
            grad_fs.append(grad_f_new)
            xn = xo - tau * grad_f_new.to(device)
            xs.append(xn)
            taus.append(tau)
            xo = xn
            num += 1
        return xs, taus
    else:
        while torch.norm(grad_f_new) > tol:
            #print(torch.norm(grad_f_new))
            if max_iter:
                if num >= max_iter:
                    #print('Max it reached')
                    #print('Norm', torch.norm(grad_f_new))
                    #print('Iters:', num)
                    #print(scale_num)
                    return xn, taus, fs, num, scale_nums
            go = torch.norm(grad_f_new)
            if not scale_tau:
                tau = tau_in#.to(device)
            xn = xo.to(device) - tau * grad_f_new.to(device)
            #print(tau)

            #new_maxi = torch.max(tau * grad_f_new.to(device))
            #if new_maxi<1e-05:
            #    print(new_maxi)

            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]  # Compute gradients explicitly
            xn = xn.clone().detach()

            gn = torch.norm(grad_f_new)
            if go == gn:
                #print(f"Iteration {num}, Initial Norm: {go:.12f}")
            #    print('Equal grads')
                print('ITERS:', num)
                print(scale_num)
                return xs, taus, fs, num, scale_nums
            if f(xn,y)>f(xo,y):
                # print('REDUCE', num)
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                scale_tau = True
                #print("Scaling Tau")
                if res>1000:
                    print('too many reductions')
                    #print('No change for a while')
                    #print('ITERS:', num)
                    #print('Norm', torch.norm(grad_f_new))
                    print(scale_num)
                    return xs, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                #print(f_value)
                xs.append(xn)
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                xo = xn.clone().detach()
                num += 1
    #print('ITERS:', num)
    #print('Norm', torch.norm(grad_f_new))
    print('ITERS:', num)
    print(scale_num)
    return xs, taus, fs, num, scale_nums


def backtracking_tau(new_f, grad_f, x_k, y, tau=np.float64(10), rho=np.float64(0.9)):
    num_times=0
    while new_f(x_k.double() - np.float64(tau) * grad_f.double(), y.double()) > new_f(x_k.double(), y.double()) - np.float64(tau / 2) * torch.norm(grad_f.double()) ** 2:
        tau *= rho
        num_times+=1
    return np.float64(tau), num_times


def gradient_descent_backtracking(f, x0, y, iters=False, tol=1):
    xo = x0.double().clone().detach()
    xs = [xo.double()]
    fs = [f(xo, y).detach().cpu().numpy()]
    taus = []
    grad_fs = []
    num = 1
    scale_num = 0
    res = 0
    new_f = lambda x: f(x,y)
    scale_nums = []
    scale_tau=False

    xo = xo.requires_grad_(True)  # Set requires_grad to True
    f_value = new_f(xo.double()).double()
    grad_f_new = torch.autograd.grad(f_value, xo.double())[0]
    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients
    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo.double()).double()
            grad_f_new = torch.autograd.grad(f_value, xo.double())[0]
            grad_fs.append(grad_f_new)


            tau, new_num = backtracking_tau(f, grad_f_new.to(device), xo.double().to(device), y.double().to(device))
            scale_num += new_num
            xn = (xo.double() - tau * grad_f_new.double().to(device)).double()
            xs.append(xn)
            taus.append(tau)
            xo = xn.clone().detach()
            num += 1

        return xs, taus
    else:
        while torch.norm(grad_f_new.double()).double() > np.float64(tol):
            # print(torch.norm(grad_f_new))
            if num%100==0:
                print(num)
            if iters:
                if num >= iters:
                    # print('Max it reached')
                    # print('Norm', torch.norm(grad_f_new))
                    # print('Iters:', num)
                    print(scale_num)
                    return xs, taus
            go = torch.norm(grad_f_new.double()).double()
            xo.requires_grad_(True)
            f_value = new_f(xo.double()).double()
            grad_f_new = (torch.autograd.grad(f_value, xo.double())[0]).double()
            grad_fs.append(grad_f_new.double())

            tau, new_num = backtracking_tau(f, grad_f_new.double().to(device), xo.double(), y.double())
            scale_num += new_num
            xn = (xo.double() - np.float64(tau) * grad_f_new.double().to(device)).double()
            # print(tau)

            # new_maxi = torch.max(tau * grad_f_new.to(device))
            # if new_maxi<1e-05:
            #    print(new_maxi)

            xn.requires_grad_(True)
            f_value = new_f(xn.double()).double()
            grad_f_new = (torch.autograd.grad(f_value, xn.double())[0]).double()  # Compute gradients explicitly
            xn = xn.double().clone().detach()

            gn = torch.norm(grad_f_new.double()).double()
            if go == gn:
                # print(f"Iteration {num}, Initial Norm: {go:.12f}")
                #    print('Equal grads')
                print('ITERS:', num)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if np.float64(f(xn.double(), y)) > np.float64(f(xo.double(), y)):
                # print('REDUCE', num)
                res += 1
                tau *= np.float64(0.9)
                scale_num += 1
                if not scale_tau:
                    scale_nums.append(num)
                scale_tau = True
                # print("Scaling Tau")
                if tau  < 1e-10:
                    print('too many reductions')
                    # print('No change for a while')
                    # print('ITERS:', num)
                    # print('Norm', torch.norm(grad_f_new))
                    print(scale_num)
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn.double())
                xs = xs[-2:]
                fs.append(f(xn.double(), y.double()).detach().cpu().numpy())
                taus.append(np.float64(tau))
                xo = xn.double().clone().detach()
                num += 1
        # print('ITERS:', num)
        # print('Norm', torch.norm(grad_f_new))
    print('ITERS:', num)
    print(scale_num)
    return xn, taus, fs, num, scale_nums



def gradient_descent_backtracking_all(f, x0, y, iters=False, tol=1):
    xo = x0.double().clone().detach()
    xs = [xo.double()]
    fs = [f(xo, y).detach().cpu().numpy()]
    taus = []
    grad_fs = []
    num = 1
    scale_num = 0
    res = 0
    new_f = lambda x: f(x,y)
    scale_nums = []
    scale_tau=False

    all_xs = []

    xo = xo.requires_grad_(True)  # Set requires_grad to True
    f_value = new_f(xo.double()).double()
    grad_f_new = torch.autograd.grad(f_value, xo.double())[0]
    xo = xo.clone().detach()  # Detach the tensor AFTER computing gradients
    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo.double()).double()
            grad_f_new = torch.autograd.grad(f_value, xo.double())[0]
            grad_fs.append(grad_f_new)


            tau, new_num = backtracking_tau(f, grad_f_new.to(device), xo.double().to(device), y.double().to(device))
            scale_num += new_num
            xn = (xo.double() - tau * grad_f_new.double().to(device)).double()
            xs.append(xn)
            taus.append(tau)
            xo = xn.clone().detach()
            num += 1

        return xs, taus
    else:
        while torch.norm(grad_f_new.double()).double() > np.float64(tol):
            # print(torch.norm(grad_f_new))
            if iters:
                if num >= iters:
                    # print('Max it reached')
                    # print('Norm', torch.norm(grad_f_new))
                    # print('Iters:', num)
                    print(scale_num)
                    return xs, taus, fs, num, scale_nums, all_xs
            go = torch.norm(grad_f_new.double()).double()
            xo.requires_grad_(True)
            f_value = new_f(xo.double()).double()
            grad_f_new = (torch.autograd.grad(f_value, xo.double())[0]).double()
            grad_fs.append(grad_f_new.double())

            tau, new_num = backtracking_tau(f, grad_f_new.double().to(device), xo.double(), y.double())
            scale_num += new_num
            xn = (xo.double() - np.float64(tau) * grad_f_new.double().to(device)).double()
            # print(tau)

            # new_maxi = torch.max(tau * grad_f_new.to(device))
            # if new_maxi<1e-05:
            #    print(new_maxi)

            xn.requires_grad_(True)
            f_value = new_f(xn.double()).double()
            grad_f_new = (torch.autograd.grad(f_value, xn.double())[0]).double()  # Compute gradients explicitly
            xn = xn.double().clone().detach()

            gn = torch.norm(grad_f_new.double()).double()
            if go == gn:
                # print(f"Iteration {num}, Initial Norm: {go:.12f}")
                #    print('Equal grads')
                print('ITERS:', num)
                print(scale_num)
                return xn, taus, fs, num, scale_nums, all_xs
            if np.float64(f(xn.double(), y)) > np.float64(f(xo.double(), y)):
                # print('REDUCE', num)
                res += 1
                tau *= np.float64(0.9)
                scale_num += 1
                if not scale_tau:
                    scale_nums.append(num)
                scale_tau = True
                # print("Scaling Tau")
                if res > 1000:
                    print('too many reductions')
                    # print('No change for a while')
                    # print('ITERS:', num)
                    # print('Norm', torch.norm(grad_f_new))
                    print(scale_num)
                    return xn, taus, fs, num, scale_nums, all_xs
            else:
                scale_tau = False
                res = 0
                xs.append(xn.double())
                xs = xs[-2:]
                fs.append(f(xn.double(), y.double()).detach().cpu().numpy())
                taus.append(np.float64(tau))
                xo = xn.double().clone().detach()
                all_xs.append(xn.double())
                num += 1
        # print('ITERS:', num)
        # print('Norm', torch.norm(grad_f_new))
    print('ITERS:', num)
    print(scale_num)
    return xn, taus, fs, num, scale_nums, all_xs



def gradient_descent_function(f, x0, y, tau_model, sig, std, iters=100, tol=1):
    xo = x0
    fs = [f(xo, y).detach().cpu().numpy()]
    xs = [xo]
    taus = []
    scale_nums = []
    num = 1
    scale_num = 0
    grad_fs = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)
    if tol == 1:
        while (num <= iters):
            if num != 1:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
            else:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
            input = input.view(1, 4, x0.shape[0], x0.shape[1])
            tau = tau_model(input)
            xn = xo - tau * grad_f_new.to(device)
            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]
            new_norm = torch.norm(grad_f_new)
            xs.append(xn)
            taus.append(tau)
            grad_fs.append(grad_f_new)
            xo = xn.clone().detach()
            num+=1
        return xs, taus
    else:
        while new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            if num != 1:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
            else:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
            input = input.view(1, 4, x0.shape[0], x0.shape[1])
            if not scale_tau:
                tau1 = tau_model(input)
                tau = tau1.clone().detach()
                tau = torch.min(tau, torch.tensor(50/27).to(device))
                del tau1

            xn = xo - tau* grad_f_new.to(device)
            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]  # Compute gradients explicitly
            new_norm = torch.norm(grad_f_new)
            #print(new_norm)
            #print('f value', f_value)
            if old_norm == new_norm:
                #print(f"Iteration {num}, Initial Norm: {go:.12f}")
            #    print('Equal grads')
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn,y)>f(xo,y):
                #print('REDUCE', num)
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                #print("Scaling Tau")
                scale_tau = True
                if res>1000:
                    #print('No change for a while')
                    #print('ITERS:', num)
                    #print('Norm', torch.norm(grad_f_new))
                    print(scale_num)
                    #return xs, taus
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                grad_fs.append(grad_f_new)
                xo = xn.clone().detach()
                num += 1
        print('ITERS:', num)
        print(scale_num)
        return xn, taus, fs, num, scale_nums







def heavy_ball_function(f, x0, y, tau_model, sig, std, iters=100, tol=1):
    xo = x0
    fs = [f(xo, y).detach().cpu().numpy()]
    xs = [xo]
    taus = []
    betas = []
    scale_nums = []
    num = 1
    scale_num = 0
    grad_fs = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)
    if tol == 1:
        while (num <= iters):
            if num != 1:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
            else:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
            input = input.view(1, 4, x0.shape[0], x0.shape[1])
            tau, beta = tau_model(input)
            if num == 1:
                xn = xo - tau * grad_f_new.to(device) + beta*(xo-xs[-1])
            else:
                xn = xo - tau * grad_f_new.to(device) + beta*(xo-xs[-2])
            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]
            new_norm = torch.norm(grad_f_new)
            xs.append(xn)
            taus.append(tau)
            grad_fs.append(grad_f_new)
            xo = xn.clone().detach()
            num+=1
        return xs, taus
    else:
        while new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            if num != 1:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
            else:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
            input = input.view(1, 4, x0.shape[0], x0.shape[1])
            if not scale_tau:
                tau1, beta = tau_model(input)
                tau = tau1.clone().detach()
                tau = torch.min(tau, torch.tensor(50 / 27).to(device))
                del tau1

            if num == 1:
                xn = xo - tau * grad_f_new.to(device) + beta * (xo - xs[-1])
            else:
                xn = xo - tau * grad_f_new.to(device) + beta * (xo - xs[-2])
            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]  # Compute gradients explicitly
            new_norm = torch.norm(grad_f_new)
            #print(new_norm)
            #print('f value', f_value)
            if old_norm == new_norm:
                #print(f"Iteration {num}, Initial Norm: {go:.12f}")
            #    print('Equal grads')
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn,y)>f(xo,y):
                #print('REDUCE', num)
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                #print("Scaling Tau")
                scale_tau = True
                if res>1000:
                    #print('No change for a while')
                    #print('ITERS:', num)
                    #print('Norm', torch.norm(grad_f_new))
                    print(scale_num)
                    #return xs, taus
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                betas.append(beta)
                grad_fs.append(grad_f_new)
                xo = xn.clone().detach()
                num += 1
        print('ITERS:', num)
        print(scale_num)
        return xn, taus, betas, fs, num, scale_nums





def gradient_descent_correctionNN(f, x0, y, model, iters=10, tol=1):
    xo = x0
    fs = [f(xo, y).detach().cpu().numpy()]
    xs = [xo]
    taus = []
    scale_nums = []
    num = 1
    scale_num = 0
    grad_fs = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)
    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
            grad_fs.append(grad_f_new)
            grad_fs = grad_fs[-2:]
            if num != 1:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
            else:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
            input = input.view(1, 4, 1, x0.shape[0], x0.shape[1])
            correction, tau = model(input)
            correction = correction.squeeze(0).squeeze(0)
            tau = torch.min(tau, torch.tensor(50 / 27).to(device))
            xn = xo - tau * grad_f_new + correction
            del correction
            xs.append(xn)
            taus.append(tau)
            xo = xn.clone().detach()  # Clone and detach for the next iteration
            num += 1
        return xs, taus
    else:
        while num<40:#new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            if num != 1:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
            else:
                input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                     xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
            input = input.view(1, 4, 1, x0.shape[0], x0.shape[1])
            if not scale_tau:
                correction, tau1 = model(input)
                tau = tau1.clone().detach()
                tau = torch.min(tau, torch.tensor(50 / 27).to(device))
                del tau1

            correction = correction.squeeze(0).squeeze(0)
            xn = xo - tau * grad_f_new + correction
            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]  # Compute gradients explicitly
            new_norm = torch.norm(grad_f_new)
            #print(new_norm)
            #print('f value', f_value)
            print(new_norm)
            if old_norm == new_norm:
                #print(f"Iteration {num}, Initial Norm: {go:.12f}")
            #    print('Equal grads')
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            # if f(xn,y)>f(xo,y):
            #     #print('REDUCE', num)
            #     res += 1
            #     tau *= 0.9
            #     if not scale_tau:
            #         scale_num += 1
            #         scale_nums.append(num)
            #     #print("Scaling Tau")
            #     scale_tau = True
                if res>1000:
                    #print('No change for a while')
                    #print('ITERS:', num)
                    #print('Norm', torch.norm(grad_f_new))
                    print(scale_num)
                    #return xs, taus
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                grad_fs.append(grad_f_new)
                xo = xn.clone().detach()
                num += 1
        print('ITERS:', num)
        print(scale_num)
        return xn, taus, fs, num, scale_nums
def gradient_descent_post_correction(f, x0, y, model, tau_model, sig, std, iters):
    xo = x0
    xs = [xo]
    taus = []
    num = 1
    scale_num = 0
    scale_nums = []
    grad_fs = []
    new_f = lambda x: f(x, y)
    while (num <= iters):
        xo.requires_grad_(True)
        f_value = new_f(xo)
        grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
        grad_fs.append(grad_f_new)
        grad_fs = grad_fs[-2:]
        if num != 1:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
        else:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
        input = input.view(1, 4, x0.shape[0], x0.shape[1])
        tau = tau_model(input, sig, torch.tensor(0.).to(device))
        update = xo - tau * grad_f_new.to(device)
        if num != 1:
            input = torch.stack((update.view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
        else:
            input = torch.stack((update.view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
        input = input.view(1, 5, 1, x0.shape[0], x0.shape[1])
        correction = model(input)
        correction = correction.squeeze(0).squeeze(0)
        xn = xo - tau * grad_f_new.to(device) + correction
        xs.append(xn)
        taus.append(tau)
        xo = xn.clone().detach()  # Clone and detach for the next iteration
        num += 1
    return xs, taus


def gradient_descent_update(f, x0, y, update_model, sig, std, iters):
    xo = x0.clone().detach()
    xs = [xo]
    num = 1
    scale_num = 0
    scale_nums = []
    grad_fs = []
    new_f = lambda x: f(x, y)

    while (num <= iters):
        xo.requires_grad_(True)
        f_value = new_f(xo)
        grad_f_new = torch.autograd.grad(f_value, xo)[0]
        grad_fs.append(grad_f_new)
        grad_fs = grad_fs[-2:]

        if num != 1:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
        else:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))

        input = input.view(1, 4, 1, x0.shape[0], x0.shape[1])
        update = update_model(input)
        xn = xo - update

        xs.append(xn)
        xo = xn.clone().detach()
        num += 1

    return xs


def gradient_descent_modelfree(f, x0, y, model, sig, std, iters):
    xo = x0.clone().detach()
    xs = [xo]
    num = 1
    scale_num = 0
    grad_fs = []
    scale_nums = []
    new_f = lambda x: f(x, y)

    while (num <= iters):
        xo.requires_grad_(True)
        f_value = new_f(xo)
        grad_f_new = torch.autograd.grad(f_value, xo)[0]
        grad_fs.append(grad_f_new)
        grad_fs = grad_fs[-2:]

        if num != 1:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
        else:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))

        input = input.view(1, 4, 1, x0.shape[0], x0.shape[1])
        update = model(input)
        xn = update

        xs.append(xn)
        xo = xn.clone().detach()
        num += 1

    return xs


def gradient_descent_unrolling(grad_f, x0, y, taus, iters):
    xo = x0
    xs = [xo]
    num = 0
    while (num < iters):
        xn = xo - taus[num % 10] * grad_f(xo, y).to(device)
        xs.append(xn)
        xo = xn
        num += 1
    return xs, taus


def gradient_descent_fixed_nesterov(f, x0, y, tau_in, beta, iters=1000, tol=1):
    tau = tau_in
    xo = x0
    xs = [xo]
    taus = []
    scale_nums = []
    fs = [f(xo, y).detach().cpu().numpy()]
    num = 1
    scale_num = 0
    grad_fs = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)
    vo = 0
    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]
            grad_fs.append(grad_f_new)

            # Calculate the term grad_f(xo + beta * vo, y) using autodiff
            term_x = xo + beta * vo
            term_f_value = new_f(term_x)
            grad_f_term = torch.autograd.grad(term_f_value, term_x)[0]

            vn = beta * vo - tau * grad_f_term.to(device)
            xn = xo + vn
            xs.append(xn)
            taus.append(tau)
            vo = vn
            xo = xn.clone().detach()
            num += 1
        return xs, taus
    else:
        while new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            xo.requires_grad_(True)

            # Calculate the term grad_f(xo + beta * vo, y) using autodiff
            term_x = xo + beta * vo
            term_f_value = new_f(term_x)
            grad_f_term = torch.autograd.grad(term_f_value, term_x)[0]
            if not scale_tau:
                tau = tau_in
            vn = beta * vo - tau * grad_f_term.to(device)
            xn = xo + vn

            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]
            new_norm = torch.norm(grad_f_new)


            if old_norm == new_norm:
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn, y) > f(xo, y):
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                #print("Scaling Tau")
                scale_tau = True
                if res > 1000:
                    print(scale_num)
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                grad_fs.append(grad_f_new)
                grad_fs = grad_fs[-2:]
                xo = xn.clone().detach()
                num += 1

    print('ITERS:', num)
    print(scale_num)
    return xn, taus, fs, num, scale_nums


def gradient_descent_fixed_momentum(f, x0, y, tau_in, beta, iters=1000, tol=1):
    tau = tau_in.clone().detach()
    xo = x0
    xs = [xo]
    taus = []
    scale_nums = []
    fs = [f(xo, y).detach().cpu().numpy()]
    num = 1
    scale_num = 0
    grad_fs = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)
    vo = 0

    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]
            grad_fs.append(grad_f_new)
            vn = beta * vo - tau * grad_f_new.to(device)
            xn = xo + vn
            xs.append(xn)
            taus.append(tau)
            vo = vn
            xo = xn.clone().detach()
            num += 1
        return xs, taus
    else:
        while new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            xo.requires_grad_(True)

            # Calculate the term grad_f(xo + beta * vo, y) using autodiff
            if not scale_tau:
                tau = tau_in
            vn = beta * vo - tau * grad_f_new.to(device)
            xn = xo + vn

            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]
            new_norm = torch.norm(grad_f_new)

            tau.detach()


            if old_norm == new_norm:
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn, y) > f(xo, y):
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                #print("Scaling Tau")
                scale_tau = True
                if res > 1000:
                    print(scale_num)
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                grad_fs.append(grad_f_new)
                grad_fs = grad_fs[-2:]
                xo = xn.clone().detach()
                num += 1

    print('ITERS:', num)
    print(scale_num)
    return xn, taus, fs, num, scale_nums

def gradient_descent_heavy_ball(f, x0, y, tau_in, beta, iters=1000, tol=1):
    tau = tau_in
    xo = x0
    xm1 = x0
    xs = [xo]
    fs = [f(xo, y).detach().cpu().numpy()]
    taus = []
    scale_nums = []
    num = 1
    scale_num = 0
    grad_fs = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)

    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]
            grad_fs.append(grad_f_new)
            xn = xo - tau * grad_f_new.to(device) + beta * (xo - xm1)
            xs.append(xn)
            taus.append(tau)
            xm1 = xo.clone().detach()
            xo = xn.clone().detach()
            num += 1

        return xs, taus
    else:
        while new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            xo.requires_grad_(True)

            # Calculate the term grad_f(xo + beta * vo, y) using autodiff
            if not scale_tau:
                tau = tau_in
            xn = xo - tau * grad_f_new.to(device) + beta * (xo - xm1)

            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]
            new_norm = torch.norm(grad_f_new)

            if old_norm == new_norm:
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn, y) > f(xo, y):
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                #print("Scaling Tau")
                scale_tau = True
                if res > 1000:
                    print(scale_num)
                    print('Too many')
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                grad_fs.append(grad_f_new)
                grad_fs = grad_fs[-2:]
                xm1 = xo.clone().detach()
                xo = xn.clone().detach()
                num += 1
    print('ITERS:', num)
    print(scale_num)
    return xn, taus, fs, num, scale_nums

def accelerated_gradient_descent(f, x0, y, tau_in, iters=1000, tol=1):
    tau = tau_in
    xo = x0
    xm1 = x0
    xs = [xo]
    taus = []
    scale_nums = []
    num = 1
    scale_num = 0
    fs = [f(xo, y).detach().cpu().numpy()]
    grad_fs = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)
    to=torch.tensor(1.)
    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]
            grad_fs.append(grad_f_new)
            tn = (1 + torch.sqrt(1 + 4 * to ** 2)) / 2
            yk = xo + ((to - 1) / tn) * (xo - xm1)

            # Calculate gradient of f with respect to yk using autodiff
            f_value_yk = new_f(yk)
            grad_f_yk = torch.autograd.grad(f_value_yk, yk)[0]

            xn = yk - tau * grad_f_yk.to(device)
            xs.append(xn)
            taus.append(tau)
            xo = xn.clone().detach()
            xm1 = xo.clone().detach()
            to = tn
            num += 1

        return xs, taus
    else:
        while new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            xo.requires_grad_(True)

            # Calculate the term grad_f(xo + beta * vo, y) using autodiff
            if not scale_tau:
                tau = tau_in
            tn = (1 + torch.sqrt(1 + 4 * to ** 2)) / 2
            yk = xo + ((to - 1) / tn) * (xo - xm1)

            # Calculate gradient of f with respect to yk using autodiff
            f_value_yk = new_f(yk)
            grad_f_yk = torch.autograd.grad(f_value_yk, yk)[0]

            xn = yk - tau * grad_f_yk.to(device)

            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]
            new_norm = torch.norm(grad_f_new)

            if old_norm == new_norm:
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn, y) > f(xo, y):
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                #print("Scaling Tau")
                scale_tau = True
                if res > 1000:
                    print(scale_num)
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                grad_fs.append(grad_f_new)
                grad_fs = grad_fs[-2:]
                xo = xn.clone().detach()
                num += 1

    print('ITERS:', num)
    print(scale_num)
    return xn, taus, fs, num, scale_nums
def adagrad(f, x0, y, tau_in, epsilon, iters=1000, tol=1):
    tau = tau_in
    xo = x0
    xs = [xo]
    taus = []
    fs = [f(xo, y).detach().cpu().numpy()]
    num = 1
    scale_num = 0
    grad_fs = []
    scale_nums = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)
    ro=0
    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]
            grad_fs.append(grad_f_new)
            gn = grad_f_new.to(device)
            rn = ro + gn ** 2
            xn = xo - tau * gn / (torch.sqrt(rn) + epsilon)
            xs.append(xn)
            taus.append(tau)
            ro = rn
            xo = xn.clone().detach()
            num += 1

        return xs, taus
    else:
        while new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            xo.requires_grad_(True)

            # Calculate the term grad_f(xo + beta * vo, y) using autodiff
            if not scale_tau:
                tau = tau_in
            gn = grad_f_new.to(device)
            rn = ro + gn ** 2
            xn = xo - tau * gn / (torch.sqrt(rn) + epsilon)

            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]
            new_norm = torch.norm(grad_f_new)

            if old_norm == new_norm:
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn, y) > f(xo, y):
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                #print("Scaling Tau")
                scale_tau = True
                if res > 1000:
                    print(scale_num)
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                grad_fs.append(grad_f_new)
                grad_fs = grad_fs[-2:]
                xo = xn.clone().detach()
                num += 1

    print(scale_num)
    return xn, taus, fs, num, scale_nums

def rmsprop(f, x0, y, tau_in, beta, epsilon, iters=1000, tol=1):
    tau = tau_in
    xo = x0
    xs = [xo]
    taus = []
    scale_nums = []
    fs = [f(xo, y).detach().cpu().numpy()]
    num = 1
    scale_num = 0
    grad_fs = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)
    ro=0

    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]
            grad_fs.append(grad_f_new)
            gn = grad_f_new.to(device)
            rn = beta * ro + (1 - beta) * gn ** 2
            xn = xo - tau * gn / (torch.sqrt(rn) + epsilon)
            xs.append(xn)
            taus.append(tau)
            ro = rn
            xo = xn.clone().detach()
            num += 1

        return xs, taus
    else:
        while new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            xo.requires_grad_(True)

            # Calculate the term grad_f(xo + beta * vo, y) using autodiff
            if not scale_tau:
                tau = tau_in
            gn = grad_f_new.to(device)
            rn = beta * ro + (1 - beta) * gn ** 2
            xn = xo - tau * gn / (torch.sqrt(rn) + epsilon)

            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]
            new_norm = torch.norm(grad_f_new)

            if old_norm == new_norm:
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn, y) > f(xo, y):
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                #print("Scaling Tau")
                scale_tau = True
                if res > 1000:
                    print(scale_num)
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn, y).detach().cpu().numpy())
                taus.append(tau)
                grad_fs.append(grad_f_new)
                grad_fs = grad_fs[-2:]
                xo = xn.clone().detach()
                num += 1

    print(scale_num)
    return xn, taus, fs, num, scale_nums


def adam(f, x0, y, tau_in, beta1, beta2, epsilon, iters=1000, tol=1):
    tau = tau_in
    xo = x0
    xs = [xo]
    taus = []
    scale_nums = []
    num = 1
    fs = [f(xo, y).detach().cpu().numpy()]
    scale_num = 0
    grad_fs = []
    new_f = lambda x: f(x, y)
    new_norm = 1
    res = 0
    scale_tau = False
    xo.requires_grad_(True)
    f_value = new_f(xo)
    grad_f_new = torch.autograd.grad(f_value, xo)[0]  # Compute gradients explicitly
    grad_fs.append(grad_f_new)
    mo=0
    vo=0

    if tol == 1:
        while (num <= iters):
            xo.requires_grad_(True)
            f_value = new_f(xo)
            grad_f_new = torch.autograd.grad(f_value, xo)[0]
            grad_fs.append(grad_f_new)
            gn = grad_f_new.to(device)
            mn = beta1 * mo + (1 - beta1) * gn
            vn = beta2 * vo + (1 - beta2) * gn ** 2
            mnh = mn / (1 - beta1 ** num)
            vnh = vn / (1 - beta2 ** num)
            xn = xo - tau * mnh / (torch.sqrt(vnh) + epsilon)
            xs.append(xn)
            taus.append(tau)
            mo = mn
            vo = vn
            xo = xn.clone().detach()
            num += 1

        return xs, taus
    else:
        while new_norm > tol:
            torch.cuda.empty_cache()
            old_norm = new_norm
            xo.requires_grad_(True)

            # Calculate the term grad_f(xo + beta * vo, y) using autodiff
            if not scale_tau:
                tau = tau_in
            gn = grad_f_new.to(device)
            mn = beta1 * mo + (1 - beta1) * gn
            vn = beta2 * vo + (1 - beta2) * gn ** 2
            mnh = mn / (1 - beta1 ** num)
            vnh = vn / (1 - beta2 ** num)
            xn = xo - tau * mnh / (torch.sqrt(vnh) + epsilon)

            xn.requires_grad_(True)
            f_value = new_f(xn)
            grad_f_new = torch.autograd.grad(f_value, xn)[0]
            new_norm = torch.norm(grad_f_new)

            if old_norm == new_norm:
                print('ITERS:', num)
                #print('Norm', new_norm)
                print(scale_num)
                return xn, taus, fs, num, scale_nums
            if f(xn, y) > f(xo, y):
                res += 1
                tau *= 0.9
                if not scale_tau:
                    scale_num += 1
                    scale_nums.append(num)
                #print("Scaling Tau")
                scale_tau = True
                if res > 1000:
                    print(scale_num)
                    return xn, taus, fs, num, scale_nums
            else:
                scale_tau = False
                res = 0
                xs.append(xn)
                xs = xs[-2:]
                fs.append(f(xn,y))
                taus.append(tau)
                grad_fs.append(grad_f_new)
                grad_fs = grad_fs[-2:]
                xo = xn.clone().detach()
                num += 1

    print(scale_num)
    return xn, taus, fs, num, scale_nums