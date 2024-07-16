import torch
import torch.utils.data
import numpy as np


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, limit):
        self.data = [torch.from_numpy(x) for x in np.random.rand(64, 3, 24, 24)]
        self.limit = limit
        self.resample_size(0)

    def resample_size(self, index):
        size = torch.empty(2).uniform_(self.limit[0], self.limit[1]).long().tolist()
        # self.crop_coord = transforms.RandomCrop.get_params(self.data[index], size)

    def __getitem__(self, index):
        i, j, h, w = self.crop_coord
        img = self.data[index]
        x = img[:, i:i+h, j:j+w]
        return x

    def __len__(self):
        return len(self.data)


import skimage.data


class MyDataset2(torch.utils.data.Dataset):
    def __init__(self, size_dataset=5, max_size=10):
        self.data = [torch.from_numpy(np.moveaxis(skimage.data.astronaut(), -1, 0)) for x in range(size_dataset)]
        # self.tranform_obj = transforms.RandomResizedCrop(max_size, ratio=(1.0, 1.0), scale=(1.0, 1.0))

    def __getitem__(self, index):
        x = self.data[index]
        x = self.tranform_obj(x)
        return x

    def __len__(self):
        return len(self.data)


dataset = MyDataset(limit=[18, 22])
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4
)

counter = 0
for data in loader:
    loader.dataset.resample_size(0)
    print(counter, data.shape)
    counter += 1


# # #
import helper.plot_class as hplotc
dataset = MyDataset2(size_dataset=16, max_size=100)
loader = torch.utils.data.DataLoader(dataset, batch_size=4)

counter = 0
for data in loader:
    print(counter, data.shape)
    hplotc.SlidingPlot(data)
    counter += 1