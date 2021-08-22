import torch
import torch.optim

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


# Initialize model
w = torch.ones(len(t_u))
b = torch.zeros(len(t_u))
t_p = model(t_u=t_u, w=w, b=b)
print(t_p)

loss = loss_fn(t_p=t_p, t_c=t_c)
print(loss)


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


def dmodel_dw(t_u, w, b):
    return t_u


def dmodel_db(t_u, w, b):
    return 1.0


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


def training_loop_2(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        # Forward pass
        # Backward pass
        params = params - learning_rate * grad
        print("Epoch %d, Loss %f" % (epoch, float(loss)))
    return params


# w_est, b_est = training_loop_2(100, 1e-4, params=(torch.tensor([1, 0])), t_u=t_u, t_c=t_c)
# print(w_est, b_est)

# There! We have our parameters however, there is another way in PyTorch with a PyTorch component called autograd.
# Chapter 3 presented a comprehensive overview of what tensors are and what functions we can call on them. We left
# out one very interesting aspect, however: PyTorch tensors can remember where they come from, in terms of the
# operations and parent tensors that originated them, and they can automatically provide the chain of derivatives of
# such operations with respect to their inputs. This means we won’t need to derive our model by hand;10 given a
# forward expression, no matter how nested, PyTorch will automatically provide the gradient of that expression with
# respect to its input parameters.

params = torch.tensor([1.0, 0.0], requires_grad=True)
# Notice the requires_grad=True argument to the tensor constructor? That argument is telling PyTorch to track the
# entire family tree of tensors resulting from operations on params. In other words, any tensor that will have params
# as an ancestor will have access to the chain of functions that were called to get from params to that tensor. In
# case these functions are differentiable (and most PyTorch tensor operations will be), the value of the derivative
# will be automatically populated as a grad attribute of the params tensor.
assert params.grad is None
print(*params)
loss = loss_fn(model(t_u, *params), t_c)

print(loss.backward())
# Loss is a child of the parent tensors and calling backward on it goes up the tree, at each point assigning gradient
# to it.
print(params.grad)
# At this point, the grad attribute of params contains the derivatives of the loss with respect to each element of
# params

# Computing without autograd
dloss_dw = 2 * t_u * (model(t_u, torch.tensor([1]), torch.tensor([0])) - t_c)
print(torch.sum(dloss_dw) / len(t_c))


#############################
# Part 2
#############################

# NOTE Calling backward will lead derivatives to accumulate at leaf nodes. We need to zero the gradient explicitly
# after using it for parameter updates.
# Let's see this in action (Note - this seems not to be allowed in newer version of PyTorch, so commenting it out here)
# print(loss.backward())
# print(f"grad values after another backward pass = {params.grad}")


def training_loop_1(
    x,
    y,
    num_epochs,
    params,
    lr,
):
    for epoch in range(num_epochs):
        # reset any gradients if present
        if params.grad is not None:
            params.grad.zero_()
        # Forward pass
        y_hat = model(x, *params)
        loss = loss_fn(y_hat, y)
        # Start Backward pass
        loss.backward()
        # Update params
        with torch.no_grad():
            params -= (
                lr * params.grad
            )  # HEre it's important to use -= instead of = params - stuff as thisleads to losing the grad
        # End backward pass
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}\n\tLoss = {loss}\n\tGradients={params.grad}")


params = torch.tensor([1.0, 0.0], requires_grad=True)
training_loop_1(x=t_u, y=t_c, num_epochs=5000, params=params, lr=1e-4)
print(f"Params after training with our optimzer = {params}")

#############################
# Optimizers in PyTorch
#############################

# PyTorch abstracts the optimization strategy away from user code using optim module that is, the training loop we’ve
# examined. This saves us from the boilerplate busywork of having to update each and every parameter to our model
# ourselves. The torch module has an optim submodule where we can find classes implementing different optimization
# algorithms


print(f"optimizers in pytorch = {dir(torch.optim)}")


# Every optimizer constructor takes a list of parameters (aka PyTorch tensors, typically with requires_grad set to
# True) as the first input. All parameters passed to the optimizer are retained inside the optimizer object so the
# optimizer can update their values and access their grad attribute,


def training_loop_with_optimizer(x, y, num_epochs, params, optimizer):
    for epoch in range(num_epochs):
        # reset any gradients if present
        optimizer.zero_grad()
        # Forward pass
        y_hat = model(x, *params)
        loss = loss_fn(y_hat, y)
        # Start Backward pass
        loss.backward()
        # Update params
        optimizer.step()
        # End backward pass
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}\n\tLoss = {loss}\n\tGradients={params.grad}")


params = torch.tensor([1.0, 0.0], requires_grad=True)
optimizer = torch.optim.SGD(params=[params], lr=1e-4)
training_loop_with_optimizer(
    x=t_u, y=t_c, num_epochs=5000, params=params, optimizer=optimizer
)
print(params)


def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        with torch.no_grad():  # Use this context manager to not track the computations in the computation graph when
            # it just needs inference
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert not val_loss.requires_grad
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


# Using the related set_grad_enabled context, we can also condition the code to run with autograd enabled or
# disabled, according to a Boolean expression—typically indicating whether we are running in training or inference
# mode. We could, for instance, define a calc_forward function that takes data as input and runs model and loss_fn
# with or without autograd according to a Boolean train_is argument
def calc_forward(t_u, t_c, is_train):
    with torch.set_grad_enabled(is_train):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
    return loss


# Summary  The optim module in PyTorch provides a collection of ready-to-use optimizers for updating parameters and
# minimizing loss functions.  Optimizers use the autograd feature of PyTorch to compute the gradient for each
# parameter, depending on how that parameter contributes to the final output. This allows users to rely on the
# dynamic computation graph during complex forward passes.  Context managers like with torch.no_grad(): can be used
# to control autograd’s behavior.
