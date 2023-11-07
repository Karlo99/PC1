#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define a neural network architecture with more layers
class DeeperNN(nn.Module):
    def __init__(self):
        super(DeeperNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the 28x28 grid into a 784 vector
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize data
])

# Instantiate model and criterion
model = DeeperNN()
criterion = nn.CrossEntropyLoss()

# Function to compute gradient with respect to inputs
def compute_input_gradient(batch, target):
    batch.requires_grad_(True)
    output = model(batch)
    loss = criterion(output, target)
    model.zero_grad()  # Zero the gradients for all model parameters
    loss.backward()  # Backward pass to compute gradients with respect to inputs (since they require_grad)
    grad_input = batch.grad.data.clone()
    batch.grad = None  # Zero the gradients for the inputs
    return grad_input

# Define the number of batches to check
num_batches_to_check = 100

# DataLoader for MNIST dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# Store the gradients to compute the differences
gradients = []

# Loop over the batches
for i, (batch, target) in enumerate(train_loader):
    if i >= num_batches_to_check:
        break
    gradients.append(compute_input_gradient(batch, target))

# Compute differences between consecutive batch gradients and their norms
grad_diff_norms = []
for i in range(len(gradients) - 1):
    grad_diff = (gradients[i] - gradients[i + 1]).norm()
    grad_diff_norms.append(grad_diff)

# Compute the average or maximum of gradient differences
# It gives us an empirical estimate of L
L_estimate_avg = torch.mean(torch.tensor(grad_diff_norms))
L_estimate_max = torch.max(torch.tensor(grad_diff_norms))

print(f"Average estimated L-smoothness constant: {L_estimate_avg}")
print(f"Max estimated L-smoothness constant: {L_estimate_max}")





# In[5]:


# DataLoader for MNIST dataset with a specific batch size
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Function to compute the squared norm of gradients for individual samples
def compute_individual_grads(model, criterion, batch, target):
    squared_norms = []
    for i in range(batch.size(0)):
        model.zero_grad()
        output = model(batch[i].unsqueeze(0))
        loss = criterion(output, target[i].unsqueeze(0))
        loss.backward()
        # Compute the squared norm of gradients
        squared_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)
        squared_norms.append(squared_norm.item())
    return squared_norms

# Check SGC for a few batches
rho_values = []
for batch, target in train_loader:
    # Compute the gradient norm for the mini-batch
    model.zero_grad()
    output_batch = model(batch)
    loss_batch = criterion(output_batch, target)
    loss_batch.backward()
    grad_norm_batch = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)

    # Compute the sum of the squared norms of the gradients for individual samples
    individual_grads = compute_individual_grads(model, criterion, batch, target)
    sum_individual_grads = sum(individual_grads)

    # Estimate rho for the SGC
    rho = grad_norm_batch / sum_individual_grads
    rho_values.append(rho.item())

# Report the maximum alpha found which gives us an indication of SGC
max_rho = max(rho_values)
print(f"Maximum rho for the strong growth condition: {max_rho}")

# If max_alpha is not much greater than 1, the strong growth condition may be satisfied.


# In[7]:


# Now that we have found our rho and our L we can calculate the constant step size for our non-convex loss function


# In[ ]:




