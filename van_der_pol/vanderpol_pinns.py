"""
Description: solve Zubov's PDE using physics-informed neural network 
to obtain a near-maximal Lyapunov function.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
import time
from datetime import datetime
import pytz
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
ylim = 3.5  # |y|_max <=ylim
xlim = 2.5  # |x|_max <=ylim
lr = 0.001  # learning rate
n_epochs = 200  # number of epochs
layer = 2  # number of hidden layers
width = 10  # number of neurons in each hidden layer
model_name = "Van_der_Pol_PINNS_data"
c1 = 0.01
c2 = 1
mu = 0.1
batch_size = 32
mean_err = 1e-5
max_err = 1e-2

n_ground_truth = 900

# Define the dynamics of the reversed van der Pol oscillator
def reverse_van_der_pol(xy, mu=1.0):
    x, y = xy[:, 0], xy[:, 1]
    return torch.stack([-y, -mu * (1.0 - x ** 2) * y + x], dim=1)

# Define the van der Pol system
def van_der_pol(t, xy, mu=1.0):
    x, y = xy[0], xy[1]
    return [y, mu * (1.0 - x ** 2) * y - x]

def sample_points(n, x_range=(-xlim, xlim), y_range=(-ylim, ylim)):
    x1 = np.random.uniform(x_range[0], x_range[1], n)
    x2 = np.random.uniform(y_range[0], y_range[1], n)
    return np.column_stack((x1, x2))

x_train = torch.FloatTensor(sample_points(n_samples))

# Define the quadratic Lyapunov function
def quad_func(x1, x2):
    return 1.5*x1**2 - x1*x2 + x2**2

# Solve the van der Pol system to get the limit cycle as ROA boundary
x_init = [0.1, 0.1]
T_max = 100.0
t_eval = np.linspace(T_max-14.14, T_max, 1000)  # Evaluate only over the last ~14.14 second (one period)
vsol = solve_ivp(van_der_pol, [0, T_max], x_init, rtol=1e-6, atol=1e-9, t_eval=t_eval)

def custom_loss(y_pred, dy_pred, x, x_ground_truth, y_ground_truth):
    norm_x_sq = torch.norm(x, dim=1) ** 2
    term1 = 0
    term2 = 0
    term3 = 0
    #term1 = torch.where(norm_x_sq < 1.0**2, torch.zeros_like(dy_pred), torch.clamp(dy_pred + norm_x_sq * (1-y_pred.squeeze()), min=0) ** 2)
    #lyap_mask1 = norm_x_sq < 4.0**2 
    #term1 = torch.where(lyap_mask1, torch.zeros_like(dy_pred), torch.clamp(dy_pred + mu*norm_x_sq * (1-y_pred.squeeze()) * (1+y_pred.squeeze()), min=0) ** 2)
    #term1 = torch.where(lyap_mask1, (dy_pred + mu * norm_x_sq * (1.0-y_pred.squeeze()) * (1+y_pred.squeeze())) ** 2, torch.zeros_like(dy_pred)) 
    term1 = (dy_pred + mu * norm_x_sq * (1.0-y_pred.squeeze()) * (1+y_pred.squeeze())) ** 2
    
    lyap_mask2 = norm_x_sq < 1**2 
    term2 = torch.where(lyap_mask2, torch.clamp(y_pred.squeeze() - torch.tanh(c1 * norm_x_sq), max=0) ** 2, torch.zeros_like(dy_pred))
    term3 = torch.where(lyap_mask2, torch.clamp(y_pred.squeeze() - torch.tanh(c2 * norm_x_sq), min=0) ** 2, torch.zeros_like(dy_pred))


    # print("norm_x_sq is: ",norm_x_sq)
    # print("y_pred.squeeze() is: ", y_pred.squeeze())
    # print("torch.tanh(c1 * norm_x_sq) is: ",torch.tanh(c1 * norm_x_sq))
    # print("torch.tanh(c2 * norm_x_sq) is: ",torch.tanh(c2 * norm_x_sq))
    # print("torch.clamp(y_pred.squeeze() - torch.tanh(c1 * norm_x_sq), max=0) ** 2 is: ",torch.clamp(y_pred.squeeze() - torch.tanh(c1 * norm_x_sq), max=0) ** 2)
    # print("torch.clamp(y_pred.squeeze() - torch.tanh(c2 * norm_x_sq), min=0) ** 2 is:",torch.clamp(y_pred.squeeze() - torch.tanh(c2 * norm_x_sq), min=0) ** 2)
    # print("term 2 is: ",term2)
    # print("term 3 is: ",term3)
    # print("lyap_mask2 is: ",lyap_mask2)

    # Add an additional term for imposing boundary conditions
    boundary_term1 = 0
    boundary_mask1 = norm_x_sq >= 3.0**2 
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
filename = "Van_der_Pol_train_data_90000_samples_ylim_[-3.5,3.5].npy"
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
                create_graph=True)[0] * reverse_van_der_pol(batch_x)).sum(dim=1)

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
levels = [0.29]
cs1 = ax2.contour(x1, x2, quad_func(x1, x2), levels=levels, colors='r', linewidths=2, linestyles='--')
levels = [0.9]
cs = ax2.contour(x1, x2, y_grid, colors='b', levels=levels)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.clabel(cs, inline=1, fontsize=10)
ax2.plot(vsol.y[0], vsol.y[1], color='red', linewidth=2, label='Limit cycle (ROA boundary)')
ax2.legend()
ax2.set_title('Level sets')
plt.tight_layout()

plot_save_path = f"{model_name}_results_layer_{layer}_width_{width}_lr_{lr}_epoch_{n_epochs}.pdf"
plt.savefig(plot_save_path)
plt.show()
