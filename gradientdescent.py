import math

import torch

sigmoid = torch.nn.Sigmoid()
relu = torch.nn.ReLU()


def f(x1, x2):
    return math.log(x1 ** + 2 * x2 ** - 2 * x1 * x2 - 2 * x2 +2)

x1 = torch.tensor(1.0, requires_grad=True)
w1 = torch.tensor(1.0, requires_grad=True)

x2 = torch.tensor(0.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)

wh = torch.tensor(1.0, requires_grad=True)

y1 = x1*w1
y1.register_hook(lambda grad: print("Grad y1 = {}".format(grad)))

y2 = x2*w2
y2.register_hook(lambda grad: print("Grad y2 = {}".format(grad)))

y3 = y1+y2
y3.register_hook(lambda grad: print("Grad y3 = {}".format(grad)))

y4 = relu(y3)
y4.register_hook(lambda grad: print("Grad y4 = {}".format(grad)))

y5 = y4 * wh
y5.register_hook(lambda grad: print("Grad y5 = {}".format(grad)))

o = sigmoid(y5)
o.register_hook(lambda grad: print("Grad o = {}".format(grad)))

#e = (1.0 - o)**2

target = 1.0
e = -target * torch.log(o) - (1 - target) * torch.log(1 - o)

e.backward()

print("Grad x1 = {}".format(x1.grad))
print("Grad x2 = {}".format(x2.grad))
print("Grad w1 = {}".format(w1.grad))
print("Grad w2 = {}".format(w2.grad))

print("Done")
