import numpy as np
import torch
import torch.nn as nn

import sys

if len(sys.argv) < 3:
    print("Please provide a filename as a command line argument")
    sys.exit()

model_file = sys.argv[1]
c2 = float(sys.argv[2])

# Define the neural network architecture and load the model
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

net = torch.load(model_file)

## Load data file augment the PINNS training
filename = "Poly_train_data_90000_samples.npy"
data = np.load(filename)
x_data, y_data = data[:, :-1], data[:, -1]

outputs = net(torch.Tensor(x_data)).squeeze().detach().numpy()
print(f"The size of 'outputs' is {len(outputs)}")

num_cases = np.count_nonzero(y_data < 1)
print(f"The size of 'winning set' is {num_cases}")
count = 0

for i in range(len(outputs)):
    if outputs[i] <= c2:
        count += 1

ratio = count / num_cases * 100

print(f"The ratio of cases where the model output is less than or equal to {c2} and y_data < 1 is {ratio:.2f}%")

