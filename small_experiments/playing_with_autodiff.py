import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import torch
import numpy as np
from torch.autograd.functional import vjp, vhp, jacobian, hessian, jvp, hvp
import matplotlib.pyplot as plt


"""
Playing around with autograd... VJP (Vector Jacobian Product) VHP (Vector Hessian Product)
"""


def calc_fun(x):
    # torch.pow(x, 2)
    return x ** 3


def calc_fun_sum(x):
    # torch.pow(x, 2)
    return (x ** 3).sum()


n = 20
v = torch.ones((n))
x_range = np.linspace(-2, 2, n)
y = calc_fun(x_range)
plt.plot(y)
input_points = torch.from_numpy(x_range).float()

# But why...
predict_vjp = vjp(calc_fun, input_points, v, create_graph=True)
predict_vhp = vhp(calc_fun_sum, input_points, v, create_graph=True)
vhp_0 = predict_vhp[0].detach().numpy()
vhp_1 = predict_vhp[1].detach().numpy()

vjp_0 = predict_vjp[0].detach().numpy()
vjp_1 = predict_vjp[1].detach().numpy()
# Normal values
plt.plot(y, 'r')
# First derivative
plt.plot(vjp_1, '-.k')
# Second derivative
plt.plot(vhp_1, '-.b')

"""
Playing around with autograd... VJP (Vector Jacobian Product) VHP (Vector Hessian Product)
"""


def calc_fun(x):
    # torch.pow(x, 2)
    return x[0] ** 3 + x[1] ** 2

def calc_fun_x(x):
    return 3 * x[0] ** 2

def calc_fun_y(x):
    return 2 * x[1] ** 1

def calc_fun_xx(x):
    return 6 * x[0]

def calc_fun_xy(x):
    return 0

def calc_fun_yy(x):
    return 2

def calc_fun_sum(x):
    # torch.pow(x, 2)
    return calc_fun(x).sum()


n_points = 20
n_dim = 2
v = torch.ones((n_dim))
v0 = torch.ones((n_dim))
v0[0] = 0
x_range = np.linspace(-2, 2, n_points)
y_range = np.linspace(-2, 2, n_points)
xy_range = np.stack([x_range, y_range], axis=-1)
z = calc_fun(xy_range.T)
plt.plot(z, 'r*')

input_points = torch.from_numpy(xy_range).float()

# First check the first derivatives...
z_x = []
z_y = []
for i_point in input_points:
    predict_vjp = vjp(calc_fun, i_point.reshape(2))
    vjp_x, vjp_y = [x.detach().numpy() for x in predict_vjp[1]]
    z_x.append(vjp_x)
    z_y.append(vjp_y)

plt.plot(z_y)

predict_vhp = vhp(calc_fun_sum, input_points[0:1].reshape(2), v=v0)
vhp_x, vhp_y = [x.detach().numpy() for x in predict_vhp[1]]
calc_fun_xx(input_points[0:1].reshape(2))
calc_fun_yy(input_points[0:1].reshape(2))

vhp_0 = predict_vhp[0].detach().numpy()
vhp_1 = predict_vhp[1].detach().numpy()

vjp_0 = predict_vjp[0].detach().numpy()
vjp_1 = predict_vjp[1].detach().numpy()
# Normal values
plt.plot(y, 'r')
# First derivative
plt.plot(vjp_1, '-.k')
# Second derivative
plt.plot(vhp_1, '-.b')

"""
Now try it with a 2D approach...
"""


def calc_fun_2d_no_y(X, Y):
    return X ** 2 + X ** 3


def calc_fun_2d(X, Y):
    return X ** 2 + Y ** 3 + X ** 3 * Y ** 3


def calc_fun_sum_2d(X, Y):
    return (calc_fun_2d(X, Y)).sum()


def fun_x(X, Y):
    return 2 * X + 3 * X ** 2 * Y ** 3


def fun_xx(X, Y):
    return 2 + 6 * X * Y ** 3


def fun_y(X, Y):
    return 3 * Y ** 2 + 3 * X ** 3 * Y ** 2


def fun_yy(X, Y):
    return 6 * Y + 6 * X ** 3 * Y


def fun_xy(X, Y):
    return 9 * X ** 2 * Y ** 2

n = 20
x_range = np.linspace(-2, 2, n)
X, Y = np.meshgrid(x_range, x_range)
# Z = calc_fun_2d_no_y(X, Y)
Z = calc_fun_2d(X, Y)
# Inspect the functionh
hplotc.SlidingPlot(Z, ax_3d=True)

X_tens = torch.from_numpy(X).float()
Y_tens = torch.from_numpy(Y).float()
input_points = (X_tens, Y_tens)
v = torch.ones((n, n))
v0 = torch.zeros((n, n))

# Check first derivative...
predict_vjp = vjp(calc_fun_2d, input_points, v, create_graph=True)
vjp_value = predict_vjp[0].detach().numpy()
vjp_x, vjp_y = [x.detach().numpy() for x in predict_vjp[1]]

Z_x = fun_x(X, Y)
Z_y = fun_y(X, Y)
hplotf.plot_3d_list([vjp_y, Z_y, vjp_y - Z_y])
hplotf.plot_3d_list([vjp_x, Z_x, vjp_x - Z_x])

# Check second derivatives
predict_vhp_1 = vhp(calc_fun_sum_2d, input_points, (v, v0))
predict_vhp_2 = vhp(calc_fun_sum_2d, input_points, (v0, v))

def return_tuple(predict_vhp):
    vhp_value = predict_vhp[0].detach().numpy()
    vhp_x, vhp_y = [x.detach().numpy() for x in predict_vhp[1]]
    return vhp_x, vhp_y

V_xx, V_xy = return_tuple(predict_vhp_1)
V_xy, V_yy = return_tuple(predict_vhp_2)

Z_xx = fun_xx(X, Y)
Z_yy = fun_yy(X, Y)
Z_xy = fun_xy(X, Y)

hplotf.plot_3d_list([V_xx, V_yy, Z_xx, Z_yy, Z_xy])

# Now check out why he ash used this weird loop
second_grad_list_x = []
second_grad_list_y = []
for i in range(n):
    v = torch.cat([torch.ones(n, 1) * (i == j) for j in range(n)], 1)
    hessian_vhp = vhp(calc_fun_sum_2d, input_points, (v, v), create_graph=True)[1]
    second_grad_list_x.append(hessian_vhp[0][:, i].detach().numpy())
    second_grad_list_y.append(hessian_vhp[1][:, i].detach().numpy())

grad_xx = np.stack(second_grad_list_x)
grad_yy = np.stack(second_grad_list_y)
hplotf.plot_3d_list([grad_xx, grad_yy])

"""
Now we have done something with meshgrid X, Y stuf...

lets now condense it to a more realistic/praktical relation
"""

import torch
import numpy as np
from torch.autograd.functional import vjp, jvp
import matplotlib.pyplot as plt


def calc_fun_single(X):
    return X ** 3


def calc_fun_single_sum(X):
    return calc_fun_single(X).sum()

n = 20
x_range = np.linspace(-2, 2, n)
y_range = np.linspace(0, 2, n)
x1, x2 = np.meshgrid(x_range, y_range)
X = x1 * x2
Z = calc_fun_single(X)
Z_x = 3 * x1 ** 2 * x2 ** 3
Z_xx = 6 * x1 * x2 ** 3
Z_y = 3 * x2 ** 2 * x1 ** 3
Z_yy = 6 * x2 * x1 ** 3
Z_xy = 9 * x2 ** 2 * x1 ** 2

X_tens = torch.from_numpy(X).float()
input_points = X_tens
v_points = torch.ones((n, n))

# Check first derivative...
# JVP is now...
predict_jvp = jvp(calc_fun_single, input_points, v_points)
fig, axes = plt.subplots(3)
axes[0].imshow(predict_jvp[1])
axes[0].set_title('Derivative by autograd')
axes[1].imshow(Z_x)
axes[1].set_title('x-derivative by analytic calculation')
axes[2].imshow(Z_x - predict_jvp[1].numpy())
axes[2].set_title('difference')

hplotf.plot_3d_list([Z, predict_jvp[0], predict_jvp[1], Z_x, Z_y])

predict_vjp = vjp(calc_fun_single, input_points, v_points)
hplotf.plot_3d_list([Z, predict_vjp[0], predict_vjp[1], Z_x, Z_y])

# YUS, This is how we do d_y... heh I really did it like this...
# Whut did I do
predict_jvp = jvp(calc_fun_single, input_points, v)
jvp_value = predict_jvp[0].detach().numpy()
jvp_y = predict_jvp[1].detach().numpy()

hplotf.plot_3d_list([jvp_y, Z_x, Z_y])

"""
Stack Overflow Question text/code
"""


"""
Question about autodiff to replace tf.gradients

Hi,

I would like to use autodiff since I have heard many great stories about it. However, I want to check if 
it is even possible what I want.

Given a function that returns an image X (generated by one single input image).. I want to compute the 
x- and y-directional derivative. Normally I would use a kernel method, but I want to see if it is possible with this autodiff.

I understand that autodif is based on the Jacobian, and since I only have one input for my function.. it will only calculate
the derivative based on that variable. Hence, I expect that it will not be able to do these directional derivative.

Is this true? Or is it possible in another way?

"""

def calc_fun_single(X):
    return X ** 3


def calc_fun_single_sum(X):
    return calc_fun_single(X).sum()


n = 20
x_range = np.linspace(-2, 2, n)
y_range = np.linspace(0, 2, n)
x1, x2 = np.meshgrid(x_range, y_range)
X = x1 * x2
Z = calc_fun_single(X)
Z_x = 3 * x1 ** 2 * x2 ** 3
Z_y = 3 * x2 ** 2 * x1 ** 3

X_tens = torch.from_numpy(X).float()
v_points = torch.ones((n, n))
v_points = torch.from_numpy(np.diag(np.ones(n)))

# Check first derivative...
predict_jvp = jvp(calc_fun_single, X_tens, v_points)
fig, axes = plt.subplots(3)
axes[0].imshow(Z_x)
axes[1].imshow(predict_jvp[1])
axes[2].imshow(Z_x - predict_jvp[1].numpy())

from torch.autograd.functional import jacobian
predict_jvp = jacobian(calc_fun_single, X_tens)
hplotf.plot_3d_list([predict_jvp.sum(axis=0).sum(axis=0)])
fig, axes = plt.subplots(3)
axes[0].imshow(Z_x)
axes[1].imshow(predict_jvp[1])
axes[2].imshow(Z_x - predict_jvp[1].numpy())
