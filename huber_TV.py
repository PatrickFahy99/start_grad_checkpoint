import numpy as np
from functions import estimate_operator_norm

def huber_penalty(s, epsilon=0.05):
    """Compute the Huber penalty for value s."""
    if abs(s) <= epsilon:
        return (0.5 * s ** 2)/epsilon
    else:
        return abs(s) - 0.5 * epsilon


import torch


def Du(u):
    """Compute the discrete derivative of u using PyTorch."""
    diff_x = torch.diff(u, dim=1)
    diff_y = torch.diff(u, dim=0)

    # Padding zeros to make the tensors of shape N x N
    diff_x = torch.cat([diff_x, torch.zeros(diff_x.size(0), 1)], dim=1)
    diff_y = torch.cat([diff_y, torch.zeros(1, diff_y.size(1))], dim=0)

    # Stacking the tensors along a new third dimension
    result = torch.stack([diff_x, diff_y], dim=2)

    return result


def D_star_u(u1, u2):
    """Compute the adjoint of the discrete derivative."""
    u1 = np.pad(u1, ((0, 0), (0, 1)), 'constant')
    u2 = np.pad(u2, ((0, 1), (0, 0)), 'constant')
    return u1 - u2


def adjoint_Du(p):
    """Compute the adjoint of the discrete derivative (i.e., the discrete divergence) of p."""

    # Extract the x and y components from the last dimension
    p_x, p_y = p[..., 0], p[..., 1]

    # Compute the negative discrete divergence
    # This involves computing the backward difference (opposite of torch.diff)
    div_x = torch.cat([p_x[:, :1], p_x[:, :-1] - p_x[:, 1:]], dim=1)
    div_y = torch.cat([p_y[:1, :], p_y[:-1, :] - p_y[1:, :]], dim=0)

    # Summing them gives the divergence
    div = -(div_x + div_y)

    return div


def power_iteration(func, num_iterations=1000, N=128):
    u = torch.rand(N,N)
    for _ in range(num_iterations):
        v = func(u)
        norm_v = torch.linalg.norm(v.view(-1))
        u = v / norm_v
    operator_norm = torch.linalg.norm(func(u).view(-1)) / torch.linalg.norm(u.view(-1))
    return operator_norm.item()

def power_iteration_adj(func, adjoint, num_iterations=1000, N=128):
    """Approximate the operator norm of the discrete gradient using power iteration."""
    u = torch.rand(N, N)  # Start with a random vector
    for _ in range(num_iterations):
        # Apply the operator and its adjoint
        v = func(u)
        u = adjoint(v)
        # Normalize the result
        u = u / torch.norm(u)
    # After convergence, compute the operator norm
    v = func(u)
    operator_norm = torch.norm(v) / torch.norm(u)
    return operator_norm.item()


def total_variation_norm_L1(image):
    grads = Du(image)
    diff_x = grads[:, :, 0]
    diff_y = grads[:, :, 1]
    tv = torch.sum(torch.sqrt(diff_x ** 2 + diff_y ** 2))
    return tv


def huberised_tv(img, epsilon=0.05):
    """Compute the Huberised Total Variation for a 2D image U."""
    tv = total_variation_norm_L1(img)
    huberised = huber_penalty(tv, epsilon)
    return huberised


def deriv_huber(img, epsilon=0.05):
    return np.sign(img) * np.minimum(np.abs(img), epsilon)
