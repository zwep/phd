from pynufft import NUFFT
patch_size = 192

model_params = {
  'input_shape': (2, patch_size, patch_size),
  'output_shape': (2, patch_size, patch_size),
  'tfx_params': {
    'nrow': patch_size,
    'ncol': patch_size,
    'nch_in': 2,
    'kernel_size': 1,
    'nl': 'relu',
    'init_fourier': False,
    'init': 'xavier_uniform_',
    'bias': True,
    'share_tfxs': False,
    'learnable': True,
  },
  'depth': 2,
  'nl':'relu'
}

from model.dAUTOMAP import *
import torch
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import random


def radial_density(traj):
    kdim = traj.shape

    # 1D Ramlak
    ram_lak = np.abs(np.linspace(-1, 1, kdim[1] + 1))
    ram_lak = ram_lak[:-1]
    middle_idx = len(ram_lak) // 2
    ram_lak[middle_idx] = 1 / (2 * kdim[1])
    return np.array([ram_lak, ] * kdim[2]).T


def radial_trajectory(kdim_in, goldenangle, random_angle):
    kdim = np.array([0] * 12)
    for i in range(len(kdim_in)):
        kdim[i] = kdim_in[i]
    pi = np.pi
    sqrt5 = 2.2360679775
    extra_dims = np.prod([kdim[2]] + kdim[4:])
    if extra_dims != 0:
        # Pre-allocate trajectory matrix
        traj = np.zeros((3, kdim[0], kdim[1], extra_dims), dtype=complex)
    else:
        traj = np.zeros((3, kdim[0], kdim[1]), dtype=complex)

    # Get radial angles for uniform (rev) or golden angle
    if goldenangle > 0:
        d_ang = (pi / (((1 + sqrt5) / 2) + goldenangle - 1))
    else:
        d_ang = pi / (kdim[1])

    np.random.seed()
    rad_ang = np.arange(0, d_ang * (kdim[1] - 1) + 1, d_ang) + random_angle

    # Line reversal for uniform
    if goldenangle == 0:
        rad_ang[1::2] = rad_ang[1::2] + pi
        rad_ang = np.mod(rad_ang, 2 * pi)

    # Calculate samples for single spoke
    # kx=linspace(0,2*kdim(1)-1,kdim(1)+1)'-(2*kdim(1)-1)/2;kx(end)=[];
    # kx=linspace(0,kdim(1)-1,kdim(1)+1)'-(kdim(1)-1)/2;kx(end)=[];
    kx = np.linspace(0, kdim[0] - 1, kdim[0] + 1).T - (kdim[0] - 1) / 2
    kx = kx[:-1]

    # Modulate successive spokes
    for ky in range(kdim[1]):
        traj[0, :, ky] = np.tile(kx * np.exp(1j * rad_ang[ky]), (1, 1, 1 if len(traj.shape) <= 3 else traj[3].shape))

    # Reshape to image size
    dims = [item for sublist in [[3], list(kdim[:2]), list(kdim[4:])] for item in sublist]
    traj = np.reshape(traj, list(filter(lambda x: x != 0, dims)))

    # Split in channels
    traj[1, :] = np.real(traj[0, :])
    traj[0, :] = np.imag(traj[0, :])

    # Simple z for stack-of-stars (linear increment)
    if kdim[2] > 1:
        if np.mod(kdim[2], 2) == 0:  # is_odd:
            kz = np.linspace(-kdim[2] / 2, kdim[2] / 2, kdim[2] + 1)
            kz = kz[:-1]
        else:
            kz = np.linspace(-kdim[2] / 2, kdim[2] / 2, kdim[2])

    for z in range(kdim[2]):
        traj[2, :, :, :, :, :, :, :, :, :] = np.tile(np.transpose(kz), (0, kdim[0], kdim[1], 1, kdim[4:]))

    return traj.real, rad_ang


# In[3]:


def generate_nufft_obj(N, random_angle, undersampling_factor):
    overgridding = 1
    spokes_to_keep = int(round(1 / undersampling_factor * N * np.pi / 2))

    Nd = (N, N)  # image size
    Kd = (N * overgridding, N * overgridding)  # k-space size
    Jd = (6, 6)  # interpolation size

    kdim = [Kd[0], spokes_to_keep]
    k, rad_ang = radial_trajectory(kdim, 1, random_angle)
    dcf = radial_density(k)

    # print(self.dcf())
    k = (k / np.max(np.abs(k))) * np.pi
    k = np.real(np.reshape(k[:2], (2, np.prod(kdim))).T)
    NufftObj = NUFFT()
    NufftObj.plan(k, Nd, Kd, Jd)
    dcf_vec = np.reshape(dcf, (np.prod(kdim)))
    return NufftObj, dcf_vec


angles = np.random.random(10) * 2 * np.pi
nuffts = []
factor = 10
for angle in angles:
    item = (*generate_nufft_obj(patch_size, angle, factor), factor)
    nuffts.append(item)


class SSIM(torch.nn.Module):
    def __init__(self, msssim=False, cs_map=False, mean_metric=True, size=11, sigma=1.5, device='cuda'):
        super(SSIM, self).__init__()
        if msssim:
            self.cs_map = True
            self.mean_metric = False
            self.msssim_mean_metric = mean_metric
        else:
            self.cs_map = cs_map
            self.mean_metric = mean_metric
        self.size = size
        self.sigma = sigma
        self.use_msssim = msssim
        self.window = self._tf_fspecial_gauss().to(device)

    def _tf_fspecial_gauss(self, max_size=11):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """

        sz = np.min([max_size, self.size])
        x, y = np.mgrid[-sz // 2 + 1:sz // 2 + 1, -sz // 2 + 1:sz // 2 + 1]

        x = np.expand_dims(x, axis=0)
        x = torch.tensor(np.expand_dims(x, axis=0), dtype=torch.float32)

        y = np.expand_dims(y, axis=0)
        y = torch.tensor(np.expand_dims(y, axis=0), dtype=torch.float32)

        g = torch.exp(-((x ** 2 + y ** 2) / (2.0 * self.sigma ** 2)))
        return g / g.sum()

    def msssim(self, img1, img2, level=5):
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=img1.device, dtype=torch.float32)
        mssim = []
        mcs = []
        filters = (2, 2)
        for l in range(level):
            self.window = self._tf_fspecial_gauss(np.min([img1.shape[2], img1.shape[3]])).to(img1.device)
            ssim_map, cs_map = self.ssim(img1, img2)
            mssim.append(ssim_map.mean())
            mcs.append(cs_map.mean())
            filtered_im1 = F.avg_pool2d(img1, filters)
            filtered_im2 = F.avg_pool2d(img2, filters)

            img1 = filtered_im1
            img2 = filtered_im2

        mcs = torch.stack(mcs)
        mssim = torch.stack(mssim)

        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
        pow1 = mcs ** weights
        pow2 = mssim ** weights

        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        value = torch.prod(pow1[:-1] * pow2[-1])

        if self.msssim_mean_metric:
            value = value.mean()
        return value

    def ssim(self, img1, img2):
        K1 = 0.01
        K2 = 0.03
        L = 1  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = F.conv2d(img1, self.window, stride=1, padding=0)
        mu2 = F.conv2d(img2, self.window, stride=1, padding=0)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.window, stride=1, padding=0) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, stride=1, padding=0) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, stride=1, padding=0) - mu1_mu2
        if self.cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                  (sigma1_sq + sigma2_sq + C2)),
                     (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                 (sigma1_sq + sigma2_sq + C2))

        if self.mean_metric:
            value = value.mean()
        return value

    def forward(self, img1, img2):
        if self.use_msssim:
            return self.msssim(img1, img2)
        else:
            return self.ssim(img1, img2)


# device = 'cuda'
device = 'cpu'
device = torch.device(device)
model = dAUTOMAP(model_params['input_shape'], model_params['output_shape'], model_params['tfx_params'])
model.eval()
model = model.to(device)
DWEIGHTS_DAUTOMAP = '/home/bugger/Documents/code/dAUTOMAP/dAUTOMAP-radial_undersampling_10_with_phase.pt'
model.load_state_dict(torch.load(DWEIGHTS_DAUTOMAP, map_location=torch.device('cpu')))

import objective_helper.reconstruction as obj_helper
from objective_configuration.reconstruction import DSCAN_cartesian, DPLOT, DSCAN_us_radial, DSCAN_us_spokes
import helper.array_transf as harray
dplot_undersample_scan = os.path.join(DPLOT, 'FBPconvnet_undersampled_scan.png')

"""
Load cardiac data
"""

# Load data and undersample
sel_cpx_array = np.load(DSCAN_cartesian)
traj = obj_helper.undersample_trajectory(sel_cpx_array.shape)
radial_undersampled_image = obj_helper.undersample_img(sel_cpx_array,  traj)

# Taking the absolute value of the image to avoid destructive interference patterns
abs_image = np.abs(radial_undersampled_image)
abs_image = harray.scale_minmax(abs_image)
import skimage.transform as sktransf
radial_undersampled_image_resize = harray.resize_complex_array(radial_undersampled_image, (1, 192, 192))
kspace_stuff = harray.transform_image_to_kspace_fftn(radial_undersampled_image_resize)
radial_undersampled_image_stacked = np.stack([kspace_stuff.real, kspace_stuff.imag], axis=1)
input_tensor = torch.from_numpy(radial_undersampled_image_stacked).float()
import helper.plot_class as hplotc

model.eval()
with torch.no_grad():
    res = model(input_tensor)

hplotc.ListPlot([res, radial_undersampled_image_stacked])

"""
Now provide a proper input..
"""

import sigpy
import objective_helper.reconstruction as objective_helper
from objective_configuration.reconstruction import DSCAN_cartesian
from data_prep.objective.reconstruction.initialize_testing_data import CART_FILE_PATH
cart_sampled_img, cart_sampled_kspace = objective_helper.get_cartesian_sampled_cardiac_data(CART_FILE_PATH)
n_points, _ = cart_sampled_kspace.shape
img_shape = cart_sampled_kspace.shape
n_spokes = 192

p_undersample = 25
width = 6
ovs = 2.5

# Define a trajectory
trajectory_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=img_shape)
trajectory_radial = trajectory_radial.reshape(-1, 2)
dcf = np.sqrt(trajectory_radial[:, 0] ** 2 + trajectory_radial[:, 1] ** 2)

n_undersample = int((p_undersample / 100) * n_spokes)
undersampled_trajectory = np.array(np.split(trajectory_radial, n_spokes))
random_lines = np.random.choice(range(n_spokes), size=(n_spokes - n_undersample), replace=False)
undersampled_trajectory[random_lines] = None
undersampled_trajectory = undersampled_trajectory.reshape(-1, 2)

temp_kspace = sigpy.nufft(cart_sampled_img, coord=undersampled_trajectory, width=width, oversamp=ovs)
from scipy.ndimage import map_coordinates
selected_lines = ~np.isnan(undersampled_trajectory[:, 0])

from scipy.interpolate import griddata
min_x, min_y = undersampled_trajectory[selected_lines].min(axis=0)
max_x, max_y = undersampled_trajectory[selected_lines].max(axis=0)
x_range = np.arange(min_x, max_x)
y_range = np.arange(min_y, max_y)
X, Y = np.meshgrid(x_range, y_range)
grid_coords = np.stack([X.ravel(), Y.ravel()], -1)
res = griddata(undersampled_trajectory[selected_lines], temp_kspace[selected_lines] * dcf[selected_lines], grid_coords, fill_value=0,
               method='nearest')
A = res.reshape((len(y_range), len(x_range)))
import helper.plot_class as hplotc
hplotc.ListPlot(np.abs(A))

kspace_stuff = harray.resize_complex_array(A, (192, 192))
radial_undersampled_image_stacked = np.stack([kspace_stuff.real, kspace_stuff.imag], axis=0)
input_tensor = torch.from_numpy(radial_undersampled_image_stacked).float()
import helper.plot_class as hplotc
model.eval()
with torch.no_grad():
    res = model(input_tensor[None])

hplotc.ListPlot([res, radial_undersampled_image_stacked])


# In[6]:

# eval_files = sorted(glob('/home/mterpst5/local_scratch/AugmentableMotionData/valid/n/images/**/*-0-*.jpg'))
# print(len(eval_files))

# In[7]:


num_random_phases = 100
lxspace = np.linspace(-np.pi, np.pi, patch_size * 2)
lyspace = np.linspace(-np.pi, np.pi, patch_size * 2)
meshx, meshy = np.meshgrid(lxspace, lyspace)


def get_random_phase(size, lower, upper):
    freqs = lower + np.multiply((upper - lower), np.random.rand(1, 2))
    freqs = freqs[0]
    angles = [np.random.randint(0, 360), np.random.randint(0, 360)]
    img1 = np.array(Image.fromarray(np.cos(2 * np.pi * freqs[0] * meshx)).rotate(angles[0]))
    img2 = np.array(Image.fromarray(np.sin(2 * np.pi * freqs[1] * meshy)).rotate(angles[1]))
    bla = centercrop(img1, *size) + centercrop(img2, *size)
    img = (bla - np.min(bla)) / (np.max(bla) - np.min(bla))
    img = img * 2 * np.pi
    random_phase = img - np.pi
    return random_phase


def centercrop(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


# In[8]:


def generate_subplots(nrows, ncols, row_labels=None, col_labels=None, figsize=(13, 15)):
    fig, all_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for axes in all_axes:
        try:
            for i, ax in enumerate(axes):
                ax.set_xticks([])
                ax.set_yticks([])
        except:
            axes.set_xticks([])
            axes.set_yticks([])

    if row_labels is not None and row_labels != []:
        for ax, row in zip(all_axes[:, 0], row_labels):
            ax.set_ylabel(row, rotation=90, size='large')
    if col_labels is not None and col_labels != []:
        for ax, col in zip(all_axes[0, :], col_labels):
            ax.set_title(col)

    return fig, all_axes


def clear_axis(ax, row_label=None, col_label=None):
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    if col_label is not None:
        ax.set_title(col_label)
    if row_label is not None:
        ax.set_ylabel(row_label, rotation=90, size='large')


# In[9]:


ssims = []
ssim = SSIM()
fig, ax = generate_subplots(2, 2, row_labels=['Abs', 'Phase'], col_labels=['Target', 'Recon'], figsize=(8, 8))
plt.ion()
fig.show()
fig.canvas.draw()
ssim_metric = SSIM()

for i, f in enumerate(eval_files):
    with Image.open(f) as pil_img:
        x, y = np.array(pil_img).shape
        data = torch.zeros((2, x, y), dtype=torch.float32)
        target = torch.zeros((2, x, y), dtype=torch.float32)
        image_with_phase = (np.array(pil_img) / 255.0) * np.exp(
            -1j * get_random_phase((patch_size, patch_size), lower=0.1, upper=0.5))
        NufftObj, dcf_vec, undersampling_factor = random.choice(nuffts)
        y = NufftObj.forward(image_with_phase)
        ksp = np.fft.fftshift(NufftObj.y2k(np.multiply(y, dcf_vec)).reshape(image_with_phase.shape))

        data[0, ...] = torch.tensor(np.real(ksp), dtype=torch.float32)
        data[1, ...] = torch.tensor(np.imag(ksp), dtype=torch.float32)
        target[0, ...] = torch.tensor(np.real(image_with_phase), dtype=torch.float32)
        target[1, ...] = torch.tensor(np.imag(image_with_phase), dtype=torch.float32)
        data = data.unsqueeze(0).to(device)

        target = target.unsqueeze(0).to(device)
        output = model(data)
        tm1 = torch.sqrt(torch.pow(target[0, 0, ...], 2) + torch.pow(target[0, 1, ...], 2))
        om1 = torch.sqrt(torch.pow(output[0, 0, ...], 2) + torch.pow(output[0, 1, ...], 2))

        complex_recon = output[0, 0, ...].cpu().detach().numpy() + 1j * output[0, 1, ...].cpu().detach().numpy()
        #         ssims.append(ssim(tm1.unsqueeze(0).unsqueeze(0), om1.unsqueeze(0).unsqueeze(0)).detach().item())

        ax[0, 0].imshow(np.abs(tm1.cpu().detach().numpy()), vmin=0, vmax=1)
        ax[0, 1].imshow(np.abs(om1.cpu().detach().numpy()), vmin=0, vmax=1)
        ssim = ssim_metric(tm1.unsqueeze(0).unsqueeze(0), om1.unsqueeze(0).unsqueeze(0))
        ax[0, 1].set_title(f'Recon - SSIM: {ssim:.3f}')
        ax[1, 0].imshow(np.angle(image_with_phase), vmin=-np.pi, vmax=np.pi)
        ax[1, 1].imshow(np.angle(complex_recon), vmin=-np.pi, vmax=np.pi)
        patnum = f.split('/')[-2]
        if not os.path.isdir(f'/home/mterpst5/Pictures/trackgif_dAUTOMAP_{patnum}'):
            os.makedirs(f'/home/mterpst5/Pictures/trackgif_dAUTOMAP_{patnum}')
        fig.savefig(f'/home/mterpst5/Pictures/trackgif_dAUTOMAP_{patnum}/{i}.jpg')

        fig.canvas.draw()

# In[ ]:



