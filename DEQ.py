import torch
import torch.nn as nn
import torch.optim as optim

class DEQModel(nn.Module):
    def __init__(self, update_model, f, grad_f, x0, y, tol=1e-6):
        super(DEQModel, self).__init__()
        self.update_model = update_model ## provides x_{k+1} given z_k = (x_k, x_{k-1}, grad_f(x_k), grad_f(x_{k-1})
        self.f = f
        self.grad_f = grad_f
        self.x0 = x0
        self.y = y
        self.tol = tol

    def forward(self, x):
        xs = []
        err=self.tol+1
        xo = x
        while err>self.tol:
            with torch.no_grad():
                xn = self.update_model(x, xo, grad_fx, grad_fxm1)
                xo = x
                x = xn
                xs.append(x)

                grad_fx = self.grad_f(x)
                grad_fxm1 = self.grad_f(xo)

                err = torch.norm(grad_fx)
        return xs

    def jacobian_vector_product(self, v, z_star):
        """Compute the JVP: (I - dT/dz*) v."""
        v = v.detach().requires_grad_()
        z_star = z_star.detach().requires_grad_()

        # Compute T(z_star)
        T_z = self.T(z_star)

        # Compute the product dT/dz* v
        jvp, = torch.autograd.grad(T_z, z_star, grad_outputs=v, retain_graph=True, create_graph=True)

        # Return (I - dT/dz*) v
        return v - jvp


    # \begin{align*}
    # \frac{\partial \mathcal{L}}{\partial z^*} & \\
    # J(z^*) & = \frac{\partial T}{\partial z^*} \\
    # v & = \left( I - J(z^*) \right)^{-1} \frac{\partial \mathcal{L}}{\partial z^*} \\
    # \frac{\partial \mathcal{L}}{\partial \theta} & = v \frac{\partial T}{\partial \theta}
    # \end{align*}
    def implicit_backward(self, output_loss, z_star, n_iter=10):
        # 1. Compute dL/dz*
        grad_loss, = torch.autograd.grad(output_loss, z_star, retain_graph=True, create_graph=True)

        # 2. Use conjugate gradient method to solve: (I - dT/dz*) v = dL/dz* for v
        # Note: This is a basic and naive implementation for clarity.
        v = torch.zeros_like(grad_loss)
        r = grad_loss.clone()
        p = r.clone()
        rsold = torch.sum(r * r)

        for _ in range(n_iter):
            Ap = self.jacobian_vector_product(p, z_star)
            alpha = rsold / torch.sum(p * Ap)
            v = v + alpha * p
            r = r - alpha * Ap
            rsnew = torch.sum(r * r)
            if torch.sqrt(rsnew) < 1e-10:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        # 3. Compute the final gradient dL/dtheta using v
        torch.autograd.backward(z_star, grad_outputs=v)


# Create dummy data and model for demonstration
x = torch.randn(32, 10)
y = torch.randn(32, 10)

model = DEQModel(10, 50, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    z, aux_losses = model(x)

    loss = model.criterion(z, y)
    model.implicit_backward(loss, aux_losses)
    optimizer.step()

    total_aux_loss = sum(aux_losses).item()
    print(f"Epoch {epoch + 1}, Main Loss: {loss.item()}, Total Aux Loss: {total_aux_loss}")
