"""PyTorch has a whole submodule dedicated to neural networks, called torch.nn. It contains the building blocks
needed to create all sorts of neural network architectures. Those building blocks are called modules in PyTorch
parlance (such building blocks are often referred to as layers in other frameworks). A PyTorch module is a Python
class deriving from the nn.Module base class. A module can have one or more Parameter instances as attributes,
which are tensors whose values are optimized during the training process (think w and b in our linear model). A
module can also have one or more submodules (subclasses of nn.Module) as attributes, and it will be able to track
their parameters as well.
NOTE The submodules must be top-level attributes, not buried inside list or dict instances!
Otherwise, the optimizer will not be able to locate the submodules (and, hence, their parameters). For situations
where your model requires a list or dict of submodules, PyTorch provides nn.ModuleList and nn.ModuleDict. """

import torch
import torch.nn as nn
import torch.optim
from collections import OrderedDict

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(
    1
)  # Adds a dimension at 1 place, makes it 11x1 instead of 11
t_u = torch.tensor(t_u).unsqueeze(1)
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]
val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]
train_t_un = 0.1 * train_t_u  # Smaller size
val_t_un = 0.1 * val_t_u

linear_model = nn.Linear(1, 1, bias=True)
print(linear_model(torch.tensor([1.0])))

# Calling an instance of nn.Module with a set of arguments ends up calling a method named forward with the same
# arguments. The forward method is what executes the forward computation, while __call__ does other rather important
# chores before and after calling forward. So, it is technically possible to call forward directly, and it will
# produce the same output as __call__, but this should not be done from user code

# DO NOT DO THIS! even though this gives same results as above
# print(linear_model.forward(torch.tensor([1.0])))

print(linear_model.weight)
print(linear_model.bias)

print(linear_model(torch.zeros(1)))  # The output is just bias ehh?

# We have a model that takes one input and produces one output, but PyTorch nn.Module and its subclasses are designed
# to do so on multiple samples at the same time. To accommodate multiple samples, modules expect the zeroth dimension
# of the input to be the number of samples in the batch. We encountered this concept in chapter 4, when we learned
# how to arrange real-world data into tensors.
print(
    linear_model(torch.zeros(10, 1))
)  # this will work assuming that the first dimension is just batches

##########
# Modify temperature input to use batches
##########

print(t_c.shape)

##################
# Writing the training loop
##################
temp_model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(
    temp_model.parameters(),  # This call recurses into submodules defined in the module’s
    # init constructor and returns a flat list of all parameters encountered, so that we can
    # conveniently pass it to the optimizer constructor as we did previously.
    lr=1e-3,
)
print(
    f"So what did we pass to optimizer? - \n{list(temp_model.parameters())}\n .. it's a list of parameters with "
    f"required gradient! "
)


def loss_fn(y_hat, y):
    return ((y_hat - y) ** 2).mean()


def training_loop(x, y, x_val, y_val, optimizer, model, loss_fn, n_epochs):
    for epoch in range(1, n_epochs + 1):
        # Forward pass
        y_hat = model(x)
        y_hat_val = model(x_val)
        # Calculate loss
        training_loss = loss_fn(y_hat, y)
        val_loss = loss_fn(y_hat_val, y_val)
        # Before Back Prop, zero out the accumulated gradients
        optimizer.zero_grad()
        # Back propagation
        training_loss.backward()  # Populates gradients
        # Optimizer step
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Train Loss = {training_loss}\nValidation Loss = {val_loss}")


# There’s one last bit that we can leverage from torch.nn: the loss. Indeed, nn comes with several common loss
# functions, among them nn.MSELoss (MSE stands for Mean Square Error), which is exactly what we defined earlier as
# our loss_fn. Loss functions in nn are still subclasses of nn.Module, so we will create an instance and call it as a
# function. In our case, we get rid of the handwritten loss_fn and replace it:
# training_loop(x=train_t_u, y=train_t_c, x_val=val_t_u, y_val=val_t_c, optimizer=optimizer,
#               model=temp_model, loss_fn=nn.MSELoss(), n_epochs=10000)
# Switch in the loss with our custom loss and you'll see the values staying pretty much the same
print(linear_model.weight)  # ~ -0.48
print(linear_model.bias)  # ~-0.61


##############
# No more linear!
##############
first_nn_model = nn.Sequential(
    OrderedDict(
        [
            ("hidden", nn.Linear(1, 7)),
            ("activation", nn.Tanh()),
            ("output", nn.Linear(7, 1)),
        ]
    )
)
print(first_nn_model)

for name, param in first_nn_model.named_parameters():
    print(name, param.shape)

training_loop(
    x=train_t_u,
    y=train_t_c,
    x_val=val_t_u,
    y_val=val_t_c,
    optimizer=optimizer,
    model=first_nn_model,
    loss_fn=nn.MSELoss(),
    n_epochs=10000,
)
print(first_nn_model.hidden.weight)
print(first_nn_model.hidden.weight.grad)
