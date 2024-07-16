from torchio import Transform
import torchio
from helper_torch.resize_right.resize import resize
import skimage.data
import skimage.transform as sktransf
import numpy as np
import torch
import helper.plot_class as hplotc


class CropResize(Transform):
    def __init__(self, resize, crop_center=None, crop_size=None, random_crop=True):
        super().__init__()

        self.random_crop = random_crop
        # This should also be of dimension two
        self.crop_center = crop_center
        if isinstance(crop_size, int):
            self.crop_size_x = self.crop_size_y = crop_size // 2
        elif isinstance(crop_size, list) or isinstance(crop_size, tuple):
            # Maybe test if it is a list/tuple of length 2 as well..?
            self.crop_size_x, self.crop_size_y = crop_size
            self.crop_size_x = self.crop_size_x // 2
            self.crop_size_y = self.crop_size_y // 2
        else:
            self.crop_size_x = self.crop_size_y = None

        # This should also be of dimension two
        if isinstance(resize, int):
            self.resize_x = self.resize_y = resize
        elif isinstance(crop_size, list) or isinstance(crop_size, tuple):
            self.resize_x, self.resize_y = resize
        else:
            self.resize_x = self.resize_y = None

    def apply_transform(self, subject):
        for image in subject.get_images():
            image.set_data(image.data.float())
            transformed_tensors = []
            print("What is this ", image.data.shape)
            left, right, down, up = self.crop_coords(image.data)
            for i_tensor in image.data:
                print("shape tensor", i_tensor.shape)
                img = i_tensor[left:right, down:up, 0]
                # transformed_img = resize(img, out_shape=(self.resize_x, self.resize_y),pad_mode='constant')
                transformed_img = sktransf.resize(img, output_shape=(self.resize_x, self.resize_y), anti_aliasing=False)
                # transformed_img = resize(img, out_shape=(self.resize_x, self.resize_y), pad_mode='constant')
                print("Output shape ", transformed_img.shape)
                transformed_tensors.append(torch.from_numpy(transformed_img).float())
            image.set_data(torch.stack(transformed_tensors)[:, :, :, None])
        return subject

    @staticmethod
    def crop_coords(img):
        # img here is a 4D tensor..
        sel_img = torch.sum(torch.abs(img[:, :, :, 0]), dim=0)
        bin_image = sel_img > (0.8 * torch.mean(sel_img))
        sel_index = np.random.choice(np.argwhere(bin_image.flatten()).flatten())
        x_c, y_c = np.unravel_index(sel_index, sel_img.shape)[-2:]
        crop_size_x = np.random.randint(int(0.20 * sel_img.shape[-2]), int(0.80 * sel_img.shape[-2]))
        crop_size_y = np.random.randint(int(0.20 * sel_img.shape[-1]), int(0.80 * sel_img.shape[-1]))
        left = max(0, x_c - crop_size_x)
        right = min(img.shape[-3], x_c + crop_size_x)
        down = max(0, y_c - crop_size_y)
        up = min(img.shape[-2], y_c + crop_size_y)
        return left, right, down, up


class RandomCrop:
    def __init__(self, scale=(0.8, 1)):
        for n in scale:
            if not 0 < n <= 1:
                raise ValueError(f'Invalid scale value {n}')
        self.scale = scale

    def __call__(self, sample):
        multiplier = torch.FloatTensor(A_tens.ndim).uniform_(*self.scale)
        spatial_shape = torch.Tensor([sample.shape]).squeeze()
        crop_shape = (spatial_shape * multiplier).round().int()
        sampler = torchio.data.UniformSampler(crop_shape)
        patch = list(sampler(sample, 1))[0]
        return patch

torchio.CropOrPad

if __name__ == "__main__":
    A = skimage.data.astronaut()[:, :, 0]
    A_tens = torch.from_numpy(A).float()
    crop_resize_obj2 = RandomCrop()
    transform_obj2 = torchio.Compose([crop_resize_obj2])
    res2 = transform_obj2(A_tens)

    multiplier = torch.FloatTensor(3).uniform_(0.2)
    crop_resize_obj = CropResize(resize=256)
    transform_obj = torchio.Compose([crop_resize_obj])

    res = transform_obj(colin)

    res2.shape
    hplotc.ListPlot(res[:, :, :, 0])
    colin = torchio.datasets.Colin27()
    type(colin)
    type(colin['t1'])
    colin.spatial_shape
