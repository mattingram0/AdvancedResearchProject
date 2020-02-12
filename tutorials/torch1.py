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

# ********* LSTMs **********
lstm = nn.LSTM(input_size, hidden_size, num_layers, [other])
 - input_size: The number of features in the input - 1, for a single time series input
 - hidden_size: The number of neurons in the hidden layer of the LSTM cell
 - num_layers: The number of layers in the LSTM

 output, (h_n, c_n) = lstm(input, (h_0, c_0))
 - input = of shape (seq_len, batch, input_size)
    - seq_len = the number of values in each sliding window of input
    - batch =  
    - input_size = the number of features in the input

Using DataLoaders, you can divide your dataset into batches, and each element of a batch will contain seq_length window of samples

Suppose you have 5 samples, each of which has 6 features, and you want to have 2 batches, with each element in a batch corresponding to a sequence of 2 samples. 
Then batch 1 would be:
#             <--- input_size --->
    tensor([[[3, 12, 13, 17, 9, 100],   | <-|-- one element of a batch
             [4, 13, 14, 18, 9, 100]],  | <-|

            [[4, 13, 14, 18, 9, 100],    <- one sample within an element
             [5, 14, 15, 19, 9, 100]]]) 

batch 2 would be:
    tensor([[[5, 14, 15, 19, 9, 100],   | <- seq_length |
             [1, 10, 11, 15, 9, 100]],  |      = 2      |
                                                        | <- batch_size 
            [[1, 10, 11, 15, 9, 100],                   |       = 2
             [2, 11, 12, 16, 9, 100]]])                 |

So, as you can see, we have two batches, each of which contains two elements, with each element containing two samples. This corresponds to the following set of time series (read horizontally)
3, 4, 5, 1, 2
12, 13, 14, 10, 11
13, 14, 15, 11, 12
9, 9, 9, 9, 9,
10, 10, 10, 10, 10

An easier example, with 6 samples, each of which has 1 feature, split into a single batch of 4 elements, each of which has a sequence of length 3, is as follows:

3, 6, 1, 3, 5, 3

              | <- Just one feature
    tensor([[[3],  |                |
             [6],  | <- 3 samples   | 
             [1]], |   per element  |
                                    |
            [[6],                   |
             [1],                   |
             [3]],                  | 4 elements in the single
                                    | batch
            [[1],                   |
             [3],                   |
             [5]],                  |
                                    |
            [[3],                   |
             [5],                   |
             [3]]])                 |

Written out as one long 3D vector we would have the following input:
This has shape: (
    tensor([[[3], [6], [1]], [[6], [1], [3]], [[1], [3], [5]], [[3], [5], [3]]])




