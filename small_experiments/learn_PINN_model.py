import numpy as np
import model.NlayerNet
import model.PINN
import torch
from torch.autograd.functional import vjp, vhp, jacobian, hessian, jvp, hvp
import torch.optim
import torch.nn
import matplotlib.pyplot as plt


"""
Playing around with JVP and HVP (autodiff) and a simple neural network
"""

# Create the 1-1 model object
# model_obj = model.NlayerNet.NLayerNet(n_start=1, n_layer=3, n_hidden=100, n_end=1, activation='relu')
model_obj = model.PINN.Network(num_inputs=1, layers=(32, 16, 32), activation='tanh', num_outputs=1)

# Define the gridpoints...
n_points = 100
x_range = np.linspace(0, 2, n_points)
input_points = np.array(x_range)
# We are going to solve u_x + 2u = 0
lambda_eq = 2
# Analytical solution
y_analytical = np.exp(-2 * x_range)

# Choose loss/optimizer..
loss_obj = torch.nn.L1Loss()
# The LBFGS requires something called a 'closure'
# optim_obj = torch.optim.LBFGS(params=model_obj.parameters())
optim_obj = torch.optim.Adam(params=model_obj.parameters(), lr=0.01)
n_epoch = 1000
n_batch = 10

model_obj.train()
loss_list = []
parameter_norm_list = []
for i_epoch in range(n_epoch):
    loss = 0
    n_batch = np.random.randint(5, 15)
    for ipoint in range(n_points):
        def closure():
            optim_obj.zero_grad()
            # Calculate stuff...
            starting_index = np.random.randint(0, n_points - n_batch)
            sel_index = np.arange(starting_index, starting_index + n_batch)
            input_tensor = torch.from_numpy(input_points[sel_index]).float()[:, np.newaxis]
            v_tensor = torch.ones((n_batch, 1)).float()
            y_pred = model_obj(input_tensor)
            vjp_output = vjp(model_obj, input_tensor, v=v_tensor)[1]
            # First index should be the x-derivative...
            res = vjp_output + lambda_eq * y_pred
            # loss += loss_obj(res, torch.zeros(n_batch, 1))
            loss = loss_obj(res, torch.zeros(n_batch, 1))
            loss.backward()
            return loss

        # loss += closure()
        optim_obj.step(closure)

        # Creating a batch-like training method...
        if ipoint % 100 == 0:
            print(i_epoch, '\t', ipoint, '-', closure())
            # loss = loss / 100
            # loss.backward()
            # optim_obj.step(closure)
            # optim_obj.step()
            loss = 0

            temp_norm = [np.linalg.norm(x.detach().numpy()) for x in list(model_obj.parameters())]
            parameter_norm_list.append(temp_norm)


for i in range(len(parameter_norm_list[0])):
    plt.plot([x[i] for x in parameter_norm_list])

# Inference phase...
model_obj.eval()
input_points = x_range
res = []
for i_point in input_points:
    i_tens = torch.from_numpy(np.array([i_point])).float()
    with torch.no_grad():
        y_pred = model_obj(i_tens)
    res.append(y_pred.numpy()[0])

res = np.array(res)

plt.plot(res)
plt.plot(np.exp(-2 * x_range))

v_tensor = torch.ones(len(input_points), 1)
input_tensor = torch.from_numpy(input_points.reshape((-1, 1))).float()
vjp_output = vjp(model_obj, input_tensor, v=v_tensor)[1]
plt.plot(vjp_output + lambda_eq * model_obj(input_tensor).detach())
plt.plot(model_obj(input_tensor).detach())

"""
Okay.. ander model..??
"""
