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



def gradient_descent_fixed(grad_f, x0, y, tau, iters=10, tol=1, f=lambda x,y: 0):
    xo = x0
    xs = [xo]
    taus = []
    num = 1
    res = 0
    if tol == 1:
        while (num <= iters):
            xn = xo - tau * grad_f(xo, y).to(device)
            xs.append(xn)
            taus.append(tau)
            xo = xn
            num += 1
        return xs, taus
    else:
        while torch.norm(grad_f(xo, y)) > tol:
            go = torch.norm(grad_f(xo, y))
            xn = xo - tau * grad_f(xo, y).to(device)
            gn = torch.norm(grad_f(xn, y))
            if go == gn:
                return xs, taus
            if f(xn,y)>f(xo,y):
                res += 1
                tau *= 0.9
                if res>1000:
                    return xs, taus
            else:
                res = 0
                xs.append(xn)
                taus.append(tau)
                xo = xn
                num += 1
    #print('ITERS:', num)
    return xs, taus


def backtracking_tau(f, grad_f, x_k, y, tau=100, rho=0.9):
    while f(x_k - tau * grad_f(x_k,y),y) > f(x_k,y) - (tau/2) * np.linalg.norm(grad_f(x_k,y))**2:
        tau *= rho
    return tau



def gradient_descent_backtracking(grad_f, x0, y, f, iters):
    xo = x0
    xs = [xo]
    taus = []
    num = 1
    while (num <= iters):
        tau = backtracking_tau(f, grad_f, xo, y)
        xn = xo - tau * grad_f(xo, y)
        xs.append(xn)
        taus.append(tau)
        xo = xn
        num += 1
    return xs, taus

def gradient_descent_function(grad_f, x0, y, tau_model, sig, std, iters):
    xo = x0
    xs = [xo]
    taus = []
    num = 1
    while (num <= iters):
        if num != 1:
            input = torch.stack((grad_f(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device), grad_f(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
        else:
            input = torch.stack((grad_f(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device), grad_f(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
        input = input.view(1, 4, x0.shape[0], x0.shape[1])
        tau = tau_model(input, sig, std)
        xn = xo - tau * grad_f(xo, y).to(device)
        xs.append(xn)
        taus.append(tau)
        xo = xn
        num += 1
    return xs, taus





def gradient_descent_correctionNN(f, grad_f, x0, y, model, sig, std, iters):
    xo = x0
    xs = [xo]
    taus = []
    num = 1
    grad_fs = []
    new_f = lambda x: f(x, y)
    while (num <= iters):
        grad_f2 = grad_f(xo, y)
        xo.requires_grad_(True)
        f_value = new_f(xo)
        f_value.backward()
        grad_f_new = xo.grad
        grad_fs.append(grad_f2)
        if num != 1:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
        else:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
        input = input.view(1, 4, 1, x0.shape[0], x0.shape[1])
        correction, tau = model(input)
        correction = correction.squeeze(0).squeeze(0)
        xn = xo - tau * grad_f2 + correction
        xn = xn.detach().clone()
        xs.append(xn)
        taus.append(tau)
        xo = xn
        num += 1
    return xs, taus

def gradient_descent_post_correction(f, grad_f, x0, y, model, tau_model, sig, std, iters):
    xo = x0
    xs = [xo]
    taus = []
    num = 1
    grad_fs = []
    num = 1
    new_f = lambda x: f(x, y)
    while (num <= iters):
        grad_f2 = grad_f(xo, y)
        xo.requires_grad_(True)
        f_value = new_f(xo)
        f_value.backward()
        grad_f_new = xo.grad
        grad_fs.append(grad_f_new)
        if num != 1:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), grad_fs[-2].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
        else:
            input = torch.stack((grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), grad_fs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
        input = input.view(1, 4, x0.shape[0], x0.shape[1])
        tau = tau_model(input, sig, std)
        update = xo - tau * grad_f(xo, y).to(device)
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
        xn = xo - tau * grad_f(xo, y).to(device) + correction
        xn = xn.detach().clone()
        xs.append(xn)
        taus.append(tau)
        xo = xn
        num += 1
    return xs, taus


def gradient_descent_update(f, grad_f, x0, y, update_model, sig, std, iters):
    xo = x0.clone().detach()
    xs = [xo]
    grad_fs = []
    num = 1
    new_f = lambda x: f(x, y)
    while num <= iters:
        grad_f2 = grad_f(xo, y)
        xo.requires_grad_(True)
        f_value = new_f(xo)
        f_value.backward()
        grad_f_new = xo.grad
        grad_fs.append(grad_f_new)

        if num != 1:
            input = torch.cat((grad_fs[-1].view(x0.shape[0], x0.shape[1]), grad_fs[-2].view(x0.shape[0], x0.shape[1]),
                               xs[-1].view(x0.shape[0], x0.shape[1]), xs[-2].view(x0.shape[0], x0.shape[1])))
        else:
            input = torch.cat((grad_fs[-1].view(x0.shape[0], x0.shape[1]), grad_fs[-1].view(x0.shape[0], x0.shape[1]),
                               xs[-1].view(x0.shape[0], x0.shape[1]), xs[-1].view(x0.shape[0], x0.shape[1])))

        input = input.view(1, 4, 1, x0.shape[0], x0.shape[1])
        update = update_model(input)  # update_model(input, sig, std)
        xn = xo - update
        xn = xn.detach().clone()
        xs.append(xn)
        xo = xn
        num += 1

    return xs


def gradient_descent_modelfree(grad_f, x0, y, model, sig, std, iters):
    xo = x0
    xs = [xo]
    num = 1
    while (num <= iters):
        if num != 1:
            input = torch.concat((grad_f(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device), grad_f(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
        else:
            input = torch.concat((grad_f(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device), grad_f(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device), xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
        input = input.view(1, 4, 1, x0.shape[0], x0.shape[1])
        #input = input.view(1, 4, x0.shape[0], x0.shape[1])
        xn = model(input)#model(input, sig, std)
        xs.append(xn)
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


def gradient_descent_fixed_nesterov(grad_f, x0, y, tau, beta, iters):
    # xo = x0
    # xm1 = x0
    # xs = [xo]
    # taus = []
    # num = 1
    # while (num <= iters):
    #     xn = xo - tau * grad_f(xo, y).to(device) + beta * (xo - xm1)
    #     xs.append(xn)
    #     taus.append(tau)
    #     xm1 = xo
    #     xo = xn
    #     num += 1

    xo = x0
    xs = [xo]
    taus = []
    num = 1
    vo = 0
    while (num <= iters):
        vn = beta*vo - tau* grad_f(xo+beta*vo, y).to(device)
        xn = xo + vn
        xs.append(xn)
        taus.append(tau)
        vo = vn
        xo = xn
        num += 1
    return xs, taus

def gradient_descent_fixed_momentum(grad_f, x0, y, tau, beta, iters):
    xo = x0
    xs = [xo]
    taus = []
    num = 1
    vo = 0
    while (num <= iters):
        vn = beta*vo - tau* grad_f(xo, y).to(device)
        xn = xo + vn
        xs.append(xn)
        taus.append(tau)
        vo = vn
        xo = xn
        num += 1
    return xs, taus


def gradient_descent_heavy_ball(grad_f, x0, y, tau, beta, iters):
    xo = x0
    xm1 = x0
    xs = [xo]
    taus = []
    num = 1
    while (num <= iters):
        xn = xo - tau * grad_f(xo, y).to(device) + beta * (xo - xm1)
        xs.append(xn)
        taus.append(tau)
        xm1 = xo
        xo = xn
        num += 1
    return xs, taus



def accelerated_gradient_descent(grad_f, x0, y, tau, iters):
    xo = x0
    xm1 = x0
    to = torch.tensor([0.]).to(device)
    xs = [xo]
    taus = []
    num = 1
    while (num <= iters):
        tn = (1+torch.sqrt(1+4*to**2))/2
        yk = xo + ((to - 1)/tn) * (xo - xm1)
        xn = yk - tau * grad_f(yk, y).to(device)
        xs.append(xn)
        taus.append(tau)
        xo = xn
        xm1 = xo
        to = tn
        num += 1
    return xs, taus

def adagrad(grad_f, x0, y, tau, epsilon, iters):
    xo = x0
    ro = 0
    xs = [xo]
    taus = []
    num = 1
    while (num <= iters):
        gn = grad_f(xo, y).to(device)
        rn = ro + gn**2
        xn = xo - tau * gn / (torch.sqrt(rn) + epsilon)
        xs.append(xn)
        taus.append(tau)
        ro = rn
        xo = xn
        num += 1
    return xs, taus


def rmsprop(grad_f, x0, y, tau, beta, epsilon, iters):
    xo = x0
    ro = 0
    xs = [xo]
    taus = []
    num = 1
    while (num <= iters):
        gn = grad_f(xo, y).to(device)
        rn = beta*ro + (1-beta)*gn**2
        xn = xo - tau * gn / (torch.sqrt(rn + epsilon))
        xs.append(xn)
        taus.append(tau)
        ro = rn
        xo = xn
        num += 1
    return xs, taus

def adam(grad_f, x0, y, tau, beta1, beta2, epsilon, iters):
    xo = x0
    mo = 0
    vo = 0
    xs = [xo]
    taus = []
    num = 1
    while (num <= iters):
        gn = grad_f(xo, y).to(device)
        mn = beta1*mo + (1-beta1)*gn
        vn = beta2*vo + (1-beta2)*gn**2
        mnh = mn/(1-beta1**num)
        vnh = vn/(1-beta2**num)
        xn = xo - tau * mnh / (torch.sqrt(vnh + epsilon))
        xs.append(xn)
        taus.append(tau)
        mo = mn
        vo = vn
        xo = xn
        num += 1
    return xs, taus



def _bfgs_direction(s, y, x, hessinv_estimate=None):

    r = x.copy()
    alphas = np.zeros(len(s))
    rhos = np.zeros(len(s))

    for i in reversed(range(len(s))):
        rhos[i] = 1.0 / y[i].inner(s[i])
        alphas[i] = rhos[i] * (s[i].inner(r))
        r.lincomb(1, r, -alphas[i], y[i])

    if hessinv_estimate is not None:
        r = hessinv_estimate(r)

    for i in range(len(s)):
        beta = rhos[i] * (y[i].inner(r))
        r.lincomb(1, r, alphas[i] - beta, s[i])

    return r

def bfgs_method(f, grad, x, line_search=1.0, maxiter=1000, tol=1e-15, num_store=None, hessinv_estimate=None):


    ys = []
    ss = []

    grad_x = grad(x)
    for i in range(maxiter):
        # Determine a stepsize using line search
        search_dir = -_bfgs_direction(ss, ys, grad_x, hessinv_estimate)
        dir_deriv = search_dir.inner(grad_x)
        if np.abs(dir_deriv) == 0:
            return  # we found an optimum
        step = line_search(x, direction=search_dir, dir_derivative=dir_deriv)

        # Update x
        x_update = search_dir
        x_update *= step
        x += x_update

        grad_x, grad_diff = grad(x), grad_x
        # grad_diff = grad(x) - grad(x_old)
        grad_diff.lincomb(-1, grad_diff, 1, grad_x)

        y_inner_s = grad_diff.inner(x_update)

        # Test for convergence
        if np.abs(y_inner_s) < tol:
            if grad_x.norm() < tol:
                return
            else:
                # Reset if needed
                ys = []
                ss = []
                continue

        # Update Hessian
        ys.append(grad_diff)
        ss.append(x_update)
        if num_store is not None:
            # Throw away factors if they are too many.
            ss = ss[-num_store:]
            ys = ys[-num_store:]

def LBFGS(grad_f, x0, y, tau_model, sig, std, iters, m=10):
    xo = x0
    xs = [xo]
    taus = []
    num = 1
    q = grad_f(xo, y).view(-1).to(device)
    alpha = []
    rho = []
    s = []
    y_bfgs = []
    while (num <= iters):
        if num != 1:
            input = torch.stack((grad_f(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_f(xs[-2], y).view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-2].view(x0.shape[0], x0.shape[1]).to(device)))
        else:
            input = torch.stack((grad_f(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                 grad_f(xs[-1], y).view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device),
                                 xs[-1].view(x0.shape[0], x0.shape[1]).to(device)))
        input = input.view(1, 4, x0.shape[0], x0.shape[1])
        tau = tau_model(input, sig, std)

        gn = grad_f(xo, y).to(device)
        q = gn.view(-1)
        for i in range(len(s) - 1, -1, -1):
            alpha[i] = torch.dot(s[i], q) * rho[i]
            q = q - alpha[i] * y_bfgs[i]

        r = q / torch.dot(y_bfgs[-1], s[-1]) if y_bfgs else q
        for i in range(len(s)):
            beta = torch.dot(y_bfgs[i], r) * rho[i]
            r = r + s[i] * (alpha[i] - beta)

        xn = xo.view(-1) - tau * r
        xs.append(xn.view(x0.shape[0], x0.shape[1]))
        taus.append(tau)
        sk = (xn - xo.view(-1)).squeeze(0)
        yk = grad_f(xn.view(x0.shape[0], x0.shape[1]), y).view(-1) - gn.view(-1)
        rhok = 1 / torch.dot(yk, sk)
        if len(s) == m:
            s.pop(0)
            y.pop(0)
            rho.pop(0)
        s.append(sk)
        y_bfgs.append(yk)
        rho.append(rhok)
        xo = xn.view(x0.shape[0], x0.shape[1])
        num += 1
    return xs, taus