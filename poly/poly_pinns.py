import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import os
import argparse
import sys

# Get slurm job_id from argument
parser = argparse.ArgumentParser()
parser.add_argument('--job-id', type=int, help='the Slurm job ID')
parser.add_argument('--overwrite', action='store_true', help='overwrite existing saved model')
args = parser.parse_args()
job_id = args.job_id

# Define parameters for data generation, network model, and optimization
n_samples = 90000  # number of samples
ylim = 6.0  # |y|_max <=ylim
xlim = 6.0  # |x|_max <=ylim
lr = 0.001  # learning rate
n_epochs = 200  # number of epochs
layer = 2  # number of hidden layers
width = 30  # number of neurons in each hidden layer
model_name = "Poly_PINNS"
c1 = 0.01
c2 = 1
mu = 0.1
batch_size = 32
mean_err = 1e-5
max_err = 1e-2

n_ground_truth = 900

# Define the dynamics
def poly(z):
    x1, x2 = z[:,0], z[:,1]
    dx1_dt = x2
    dx2_dt = -2.0*x1 + 1.0/3.0*x1**3 - x2
    return torch.stack([dx1_dt, dx2_dt],dim=1)

def reverse_poly(t, z):
    x1, x2 = z
    dx1_dt = - x2
    dx2_dt = 2.0*x1 - 1.0/3.0*x1**3 + x2
    return [dx1_dt, dx2_dt]

# Solve the ODE to get the limit cycle as ROA boundary
x_init1 = [np.sqrt(6), 1e-9]
x_init2 = [np.sqrt(6), -1e-9]
x_init3 = [-np.sqrt(6), 1e-9]
x_init4 = [-np.sqrt(6), -1e-9]
T_max = 10.0
t_eval = np.linspace(0, T_max, 1000)  #
vsol1 = solve_ivp(reverse_poly, [0, T_max], x_init1, rtol=1e-6, atol=1e-9, t_eval=t_eval)
vsol2 = solve_ivp(reverse_poly, [0, T_max], x_init2, rtol=1e-6, atol=1e-9, t_eval=t_eval)
vsol3 = solve_ivp(reverse_poly, [0, T_max], x_init3, rtol=1e-6, atol=1e-9, t_eval=t_eval)
vsol4 = solve_ivp(reverse_poly, [0, T_max], x_init4, rtol=1e-6, atol=1e-9, t_eval=t_eval)

def sample_points(n, x_range=(-xlim, xlim), y_range=(-ylim, ylim)):
    x1 = np.random.uniform(x_range[0], x_range[1], n)
    x2 = np.random.uniform(y_range[0], y_range[1], n)
    return np.column_stack((x1, x2))

x_train = torch.FloatTensor(sample_points(n_samples))

# Define the quadratic Lyapunov function
def quad_func(x1, x2):
    return 1.75*x1**2 + 0.5*x1*x2 + 0.75*x2**2

def custom_loss(y_pred, dy_pred, x, x_ground_truth, y_ground_truth):
    norm_x_sq = torch.norm(x, dim=1) ** 2
    term1 = 0
    term2 = 0
    term3 = 0
    term1 = (dy_pred + mu * norm_x_sq * (1.0-y_pred.squeeze()) * (1+y_pred.squeeze())) ** 2
    
    lyap_mask2 = norm_x_sq < 0.5**2 
    term2 = torch.where(lyap_mask2, torch.clamp(y_pred.squeeze() - torch.tanh(c1 * norm_x_sq), max=0) ** 2, torch.zeros_like(dy_pred))
    term3 = torch.where(lyap_mask2, torch.clamp(y_pred.squeeze() - torch.tanh(c2 * norm_x_sq), min=0) ** 2, torch.zeros_like(dy_pred))

    # Add an additional term for imposing boundary conditions
    boundary_term1 = 0
    boundary_mask1 = (x[:,0] >= 3.0) & (x[:,1]>= 3.0)
    boundary_term1 = torch.where(boundary_mask1, y_pred.squeeze() - 1.0, torch.tensor(0.0))
        
    boundary_term2 = 0
    # boundary_mask2 = norm_x_sq <= 1e-2 
    # boundary_term2 = torch.where(boundary_mask2, y_pred.squeeze(), torch.tensor(0.0))

    # Evaluate the neural network on the (approximate) ground truth data
    y_ground_truth_pred = net(x_ground_truth)
    mse = 0
    mse = torch.mean((y_ground_truth_pred.squeeze() - y_ground_truth.squeeze()) ** 2)

    loss = term1 + term2 + term3 + boundary_term1 ** 2 + boundary_term2 ** 2 + mse

    #print("loss",loss)
    max_loss_per_sample, _ = torch.max(loss, dim=0)
    #print("max_loss_per_sample",max_loss_per_sample)
    return torch.mean(loss), max_loss_per_sample


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, num_layers, width):
        super(Net, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(2, width))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))

        self.fc = nn.Linear(width, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.fc(x)
        return x


## Load data file augment the PINNS training
filename = "Poly_train_data_90000_samples.npy"
data = np.load(filename)
x_data, y_data = data[:, :-1], data[:, -1]
indices = np.random.choice(len(x_data), n_ground_truth, replace=False)
x_ground_truth = x_data[indices]
y_ground_truth = y_data[indices]
x_ground_truth_tensor = torch.FloatTensor(x_ground_truth)
y_ground_truth_tensor = torch.FloatTensor(y_ground_truth).unsqueeze(1)

model_save_path = f"{model_name}_layer_{layer}_width_{width}_lr_{lr}_epoch_{n_epochs}.pt"

if not os.path.exists(model_save_path) or args.overwrite:
    print(f"Model not found at {model_save_path} or --overwrite=True flag set. Training new model...")

    net = Net(num_layers=layer, width=width)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    mean_losses = []
    start_time = time.time()

    net.train()  # set the network in training mode

    for epoch in range(n_epochs):
        max_loss = 0.0
        indices = torch.randperm(x_train.shape[0])
        for batch_start in range(0, x_train.shape[0], batch_size):
            batch_indices = indices[batch_start:batch_start+batch_size]
            batch_x = x_train[batch_indices]
            batch_x.requires_grad_()

            optimizer.zero_grad()

            y_pred = net(batch_x)
            dy_pred = (torch.autograd.grad(y_pred.sum(), batch_x, 
                create_graph=True)[0] * poly(batch_x)).sum(dim=1)

            mean_loss, max_loss_per_sample = custom_loss(y_pred, dy_pred, batch_x, x_ground_truth_tensor, y_ground_truth_tensor)
            max_loss = max(max_loss, max_loss_per_sample.item())
            #print(f"Batch: {batch_start//batch_size+1}, Mean Loss: {mean_loss.item():.8f}, Max Loss: {max_loss:.8f}")

            mean_loss.backward()
            optimizer.step()

        mean_losses.append(mean_loss.item())
        print(f"Epoch: {epoch + 1}, Mean Loss: {mean_loss.item():.16f}, Max Loss: {max_loss:.16f}")
        
        if max_loss < max_err:
            print(f"Stopping training early. Max Loss is {max_loss}.")
            break
        elif mean_loss.item() < mean_err:
            print(f"Stopping training early. Mean Loss is {mean_loss.item()}.")
            break

    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time} seconds")
    torch.save(net, model_save_path)

else:
    net = torch.load(model_save_path)
    print(f"Model loaded from {model_save_path}.")

# # Test the model at specific points
# # Define the points where you want to evaluate the loss
# x_eval = torch.FloatTensor([[0.0, 0.0]])
# # Compute the predicted output and the gradient of the predicted output with respect to the input
# x_eval.requires_grad_()
# y_pred = net(x_eval)
# dy_pred = (torch.autograd.grad(y_pred.sum(), x_eval, create_graph=True)[0] * reverse_van_der_pol(x_eval)).sum(dim=1)
# # Compute the loss using the custom loss function
# loss, _ = custom_loss(y_pred, dy_pred, x_eval, x_ground_truth_tensor, y_ground_truth_tensor)
# # Print the loss value
# print("Loss at x_eval:", loss.item())

# Generate a grid of points to evaluate the learned function
x1, x2 = np.mgrid[-xlim:xlim:0.01, -ylim:ylim:0.01]
xy_grid = np.column_stack((x1.ravel(), x2.ravel()))
xy_grid_tensor = torch.FloatTensor(xy_grid)
y_grid = net(xy_grid_tensor).detach().numpy().reshape(x1.shape)

# # Plot the learned function
# fig = plt.figure(figsize=(12, 6))
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(x1, x2, y_grid, cmap="viridis", alpha=0.6)
# ax.set_xlabel("x1")
# ax.set_ylabel("x2")
# ax.set_zlabel("y")
# ax.set_title(f"Learned Function for the {model_name} Oscillator")

# Plot the learned function
fig = plt.figure(figsize=(12, 6))

# Subplot 1: 3D surface plot of learned function
ax1 = fig.add_subplot(121, projection="3d")
#ax1.plot_surface(x1, x2, y_grid, cmap="viridis", alpha=0.6)
ax1.plot_surface(x1, x2, y_grid)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("y")
ax1.set_title(f"Learned Lyapunov Function")
ax1.set_zlim(-0.1, 1.1) 

# Subplot 2: Contour plot of target set and level sets
ax2 = fig.add_subplot(122)
levels = [1.05]
cs1 = ax2.contour(x1, x2, quad_func(x1, x2), levels=levels, colors='r', linewidths=2, linestyles='--')
levels = [0.95]
cs = ax2.contour(x1, x2, y_grid, colors='b', levels=levels)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.clabel(cs, inline=1, fontsize=10)
ax2.plot(vsol1.y[0], vsol1.y[1], color='red', linewidth=2, label='Approximate ROA boundary')
ax2.plot(vsol2.y[0], vsol2.y[1], color='red', linewidth=2)
ax2.plot(vsol3.y[0], vsol3.y[1], color='red', linewidth=2)
ax2.plot(vsol4.y[0], vsol4.y[1], color='red', linewidth=2)
ax2.set_xlim(-5.9,5.9)
ax2.set_ylim(-5.9,5.9)
ax2.legend()
ax2.set_title('Level sets')
plt.tight_layout()

plot_save_path = f"{model_name}_results_layer_{layer}_width_{width}_lr_{lr}_epoch_{n_epochs}.pdf"
plt.savefig(plot_save_path)
plt.show()
