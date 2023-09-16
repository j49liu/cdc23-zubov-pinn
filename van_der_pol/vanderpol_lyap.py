"""
Description: solve Zubov's PDE using a data-driven approach 
to obtain a near-maximal Lyapunov function.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import os
import random
from torch.utils.data import DataLoader, TensorDataset

# Define paramters for data generation, network model, and optimization
n_samples = 300**2  # number of samples
ylim = 3.5          # |x_2|<=ylim
lr = 0.001          # learning rate
n_epochs = 200      # number of epochs
layer = 2          # number of hiddent layers
width = 10          # number of neurons in each hidden layer
err = 1e-5          # terminating condition for loss
bach_size = 32
model_name = "Van_der_Pol"

# Define the dynamics of the van der Pol oscillator
def reverse_van_der_pol(t, z):
    x1, x2, z_ = z
    dx1_dt = -x2
    dx2_dt = x1 - (1.0 - x1**2) * x2
    dz_dt = x1**2 + x2**2
    return [dx1_dt, dx2_dt, dz_dt]

# Define the van der Pol system
def van_der_pol(t, xy, mu=1.0):
    x, y = xy[0], xy[1]
    return [y, mu * (1.0 - x ** 2) * y - x]


# Define the quadratic Lyapunov function
def quad_func(x1, x2):
    return 1.5*x1**2 - x1*x2 + x2**2


# Solve the van der Pol system to get the limit cycle as ROA boundary
x_init = [0.1, 0.1]
T_max = 100.0
t_eval = np.linspace(T_max-14.14, T_max, 1000)  # Evaluate only over the last ~14.14 second (one period)
vsol = solve_ivp(van_der_pol, [0, T_max], x_init, rtol=1e-6, atol=1e-9, t_eval=t_eval)

# Define the rules for determining the output
def get_output(x, z, T):
    x_norm = np.linalg.norm(x)
    if x_norm <= 0.001:
        y = np.tanh(0.1*z)
    elif z >= 200:
        y = 1.0
    else:
        sol = solve_ivp(reverse_van_der_pol, [0, T], [x[0], x[1], z], rtol=1e-6, atol=1e-9)
        x_T = np.array([sol.y[0][-1], sol.y[1][-1]])
        y = get_output(x_T, sol.y[2][-1], T)
    return y

# Generate training data
N = 500
T = 10
filename = f'{model_name}_train_data_{n_samples}_samples_ylim_[-{ylim},{ylim}].npy'

if os.path.exists(filename):
    # Load the data from the file
    print("Loading data...",flush=True)
    data = np.load(filename)
    x_train, y_train = data[:, :-1], data[:, -1]
else:
    # Generate new data
    print("Generating new data...",flush=True)
    start_time = time.time()

    x1_pts = np.linspace(-2.5, 2.5, int(np.sqrt(n_samples)))
    x2_pts = np.linspace(-ylim, ylim, int(np.sqrt(n_samples)))
    x1_grid, x2_grid = np.meshgrid(x1_pts, x2_pts)
    x_train = np.stack((x1_grid.reshape(-1), x2_grid.reshape(-1)), axis=1)
    y_train = np.zeros(n_samples)

    for i in range(n_samples):
        if i % N == 0:
            print("Generating sample", i, "to", i+N,flush=True)
        sol = solve_ivp(reverse_van_der_pol, [0, T], [x_train[i,0], x_train[i,1], 0], rtol=1e-6, atol=1e-9)
        x_T = np.array([sol.y[0][-1], sol.y[1][-1]])
        y_train[i] = get_output(x_T, sol.y[2][-1], T)
    # Save the data to a file
    print("Saving data...",flush=True)
    data = np.column_stack((x_train, y_train))
    np.save(filename, data)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Data saved. Total time for data generation: {total_time:.2f} seconds")

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=bach_size, shuffle=True)

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

# Check if a model with same hyperparameters exists
model_file = f"{model_name}_NN_Lyap_layer_{layer}_width_{width}_samples_{n_samples}_lr_{lr}_epoch_{n_epochs}.pt"

# Check if model file exists
if os.path.isfile(model_file):
    # Load the model from the file
    print("Loading model...",flush=True)
    net = torch.load(model_file)
else:
    # Create an instance of the neural network
    net = Net(num_layers=layer, width=width)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the model: {num_params}")

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #optimizer = pbSGD(net.parameters(), lr=lr, gamma=0.7)
    # Train the network
    losses = []
    start_time = time.time()
    print("Training network...",flush=True)
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = net(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss / len(train_loader):.5f}", flush=True)
        if epoch_loss / len(train_loader) < err:
            print(f"Stopping training after epoch {epoch} because loss is less than {err}", flush=True)
            break


    torch.save(net, model_file)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Model saved. Total training time: {total_time:.2f} seconds")

# Create a grid of points for testing the network
x1_vals, x2_vals = np.meshgrid(np.linspace(-2.5, 2.5, 500), np.linspace(-ylim, ylim, 500))
x_test = np.vstack((x1_vals.reshape(-1), x2_vals.reshape(-1))).T
x_test = torch.from_numpy(x_test).float()

# Evaluate the network on the test points
with torch.no_grad():
    y_test = net(x_test)

# Reshape the output to match the input grid
y_vals = y_test.numpy().reshape(x1_vals.shape)

# Generate plots
# Plot the training data
fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(1, 3, 1)
ax.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='coolwarm', s=2)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Training data')

# Plot the predicted function
ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot_surface(x1_vals, x2_vals, y_vals)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('W')
ax.set_title('Learned Lyapunov function')

ax = fig.add_subplot(1, 3, 3)
# Plot the target set described by the quadratic function
levels = [0.29]
cs1 = ax.contour(x1_vals, x2_vals, quad_func(x1_vals, x2_vals), levels=levels, colors='r', linewidths=2, linestyles='--')
# Plot the level sets
levels = [0.02, 0.7]
cs = ax.contour(x1_vals, x2_vals, y_vals, colors='b', levels=levels)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.clabel(cs, inline=1, fontsize=10)
ax.plot(vsol.y[0], vsol.y[1], color='red', linewidth=2, label='Limit cycle (ROA boundary)')
ax.legend()
ax.set_title('Level sets')
plt.tight_layout()

plt.savefig(f"{model_name}_Results_layer_{layer}_width_{width}_samples_{n_samples}_lr_{lr}_epoch_{n_epochs}.pdf")
print("Results saved.")
