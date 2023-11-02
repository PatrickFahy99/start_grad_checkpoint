import torch
from torch.fft import fftn, ifftn

def get_blurred_image(img, s=1):
    device = img.device  # Get the device of the input
    n = img.shape[0]
    x = torch.hstack((torch.arange(0, n // 2), torch.arange(-n // 2, 0))).to(device)  # Move tensor to the correct device
    [Y, X] = torch.meshgrid(x, x)
    h = torch.exp((-X ** 2 - Y ** 2) / (2 * s ** 2))
    h = h / torch.sum(h)
    Fh = fftn(h)
    Fu = fftn(img)
    out = ifftn(Fh * Fu)
    return out.real


def estimate_operator_norm(func, input_dim=128, num_trials=1000):
    max_ratio = 0
    for _ in range(num_trials):
        x = torch.rand(input_dim, input_dim)
        Ax = func(x)
        ratio = torch.norm(Ax) / torch.norm(x)
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio


def get_adj_image(img, s=1):
    device = img.device  # Get the device of the input
    n = img.shape[0]
    x = torch.hstack((torch.arange(0, n // 2), torch.arange(-n // 2, 0))).to(device)  # Move tensor to the correct device
    [Y, X] = torch.meshgrid(x, x)
    h = torch.exp((-X ** 2 - Y ** 2) / (2 * s ** 2))
    h = h / torch.sum(h)
    Fh = torch.conj(fftn(h))  # Take the complex conjugate of the Fourier-transformed kernel
    Fu = fftn(img)
    out = ifftn(Fh * Fu)
    return out.real
