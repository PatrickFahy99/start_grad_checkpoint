
import torch
def grad(image):
    if image.shape != torch.Size([128,128]):
        image = image.squeeze(0).squeeze(0)
    # Calculate the gradient
    gradient_x = image[:, 1:] - image[:, :-1]
    gradient_y = image[1:, :] - image[:-1, :]

    # Pad the gradients to maintain the original shape
    gradient_x = torch.nn.functional.pad(gradient_x, (0, 1, 0, 0))  # L, R, T, B
    gradient_y = torch.nn.functional.pad(gradient_y, (0, 0, 0, 1))

    # Combine the gradients into a single array with shape (128, 128, 2)
    gradient = torch.stack((gradient_x, gradient_y), dim=-1)
    return gradient

def grad_norm_squared(image):
    return torch.norm(grad(image), p=2)**2


def laplacian(image):
    if image.shape != torch.Size([128,128]):
        image = image.squeeze(0).squeeze(0)
    # Calculate the gradient
    gradient_x = image[:, 1:] - image[:, :-1]
    gradient_y = image[1:, :] - image[:-1, :]

    # Pad the gradients to maintain the original shape
    gradient_x = torch.nn.functional.pad(gradient_x, (0, 1, 0, 0))  # L, R, T, B
    gradient_y = torch.nn.functional.pad(gradient_y, (0, 0, 0, 1))

    # Calculate the second order derivatives (Laplacian)
    laplacian_x = gradient_x[:, 1:] - gradient_x[:, :-1]
    laplacian_y = gradient_y[1:, :] - gradient_y[:-1, :]

    # Pad the Laplacians to maintain the original shape
    laplacian_x = torch.nn.functional.pad(laplacian_x, (0, 1, 0, 0))  # L, R, T, B
    laplacian_y = torch.nn.functional.pad(laplacian_y, (0, 0, 0, 1))

    # Combine the Laplacians into a single array with shape (128, 128, 2)
    laplacian = laplacian_x + laplacian_y
    return laplacian


