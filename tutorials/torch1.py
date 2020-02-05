import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y) # Will now have a gradient function

z = y * y * 3
out = z.mean()
print("***** z, out *****")
print(z)
print(out)

out.backward() # Perform the backprop. Equivalent to out.backward(torch.tensor(1)) as out is a scalar
print(x.grad) # x now has a grad property after we ran backward()

x = torch.randn(3, requires_grad=True)
print(x)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print("***** y *****")
print(y)
# y.backward() # This will fail, as we now have an output (y) which is no longer a scalar value.
# print(x.grad)

# ********* Good explanation of the backward() function ********
# https://stackoverflow.com/questions/57248777/backward-function-in-pytorch
# By default, backward expects to be applied to a scalar loss function, and will calculate the gradients of this
# loss function with respect to all input variables. If backward is applied to a non-scalar tensor (say out[i,j]), # then you must pass in a tensor of values which are the partial gradients of dLoss/dOut[i,j], which are then multiplied
# by the dOut[i,j]/dX[i,j] matrix to give: dLoss/dOut[i,j] * dOut[i,j]/dX[i,j] = dLoss/dX[i,j]
# When loss.backward() is called, the whole graph is diff. w.r.t the loss, and all Tensors in the graph that have requires_grad=True will have their .grad Tensor accumulated with the gradient. Make sure to net.zero_grad() before calling backward() to ensure that all the gradient buffers are zerod

# ********* Neural Networks **********
# the torch.nn depends on autograd to define models and differentiate them. A nn.Module contains layers, a method forward(input) that returns the output
# You just have to define a forward() function in your class, and the backward function is automatically defined for you using autograd
# The learnable parameters of a model are returned by net.parameters()
# 
# Recap:
# torch.Tensor - A multi-dimensional array wth support for autograd operations like backward(). Also holds the gradient w.r.t the tensor
# nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to the GPU, exporting, loading, etc
# nn.Parameter - A kind of Tensor that is automatically registered as a parameter when assigned as an attribute to a Module
# autograd.Function - implements the forward and backward definitions of an autograd operation. Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.

# ********* Updating Weights **********
# torch.optim implements varios optimisation methods (SGD, Nesterov-SGD, Adam, RMSProp, etc)
