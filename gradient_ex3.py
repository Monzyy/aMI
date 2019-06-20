import math

import torch

sigmoid = torch.nn.Sigmoid()
relu = torch.nn.ReLU()


def f(x1, x2):
    return math.log(x1 ** + 2 * x2 ** - 2 * x1 * x2 - 2 * x2 +2)


x = torch.tensor(1.0, requires_grad=True)
w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)
w3 = torch.tensor(1.0, requires_grad=True)
w4 = torch.tensor(1.0, requires_grad=True)
w5 = torch.tensor(1.0, requires_grad=True)

h1 = sigmoid(x * w1)
h1.register_hook(lambda grad: print("Grad h1 = {}".format(grad)))

h2 = sigmoid(h1*w2)
h2.register_hook(lambda grad: print("Grad h2 = {}".format(grad)))

h3 = sigmoid(h2*w3)
h3.register_hook(lambda grad: print("Grad h3 = {}".format(grad)))

h4 = sigmoid(h3*w4)
h4.register_hook(lambda grad: print("Grad h4 = {}".format(grad)))

o = sigmoid(h4*w5)
o.register_hook(lambda grad: print("Grad h5 = {}".format(grad)))


e = (1.0 - o)**2

e.backward()

print("Grad x = {}".format(x.grad))
print("Grad w1 = {}".format(w1.grad))
print("Grad w2 = {}".format(w2.grad))
print("Grad w3 = {}".format(w3.grad))
print("Grad w4 = {}".format(w4.grad))
print("Grad w5 = {}".format(w5.grad))


print("Done")

print('Som man kan se bliver derivatives mindre og mindre, ergo har vi vanishing gradients.')