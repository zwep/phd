"""

Here we define very simple functions that can alter the loss-weights during training.
Most of the time they are simple heavyside functions
"""

import numpy as np


class Constant:
    def __init__(self, weight, **kwargs):
        self.weight = weight

    def __call__(self, epoch):
        current_weight = self.weight
        return current_weight


class HeaviSide:
    def __init__(self, weight, epoch_on, **kwargs):
        self.weight = weight
        self.epoch_on = epoch_on

    def __call__(self, epoch):
        if epoch > self.epoch_on:
            current_weight = self.weight
        else:
            current_weight = 0

        return current_weight


class HeaviSideLog:
    def __init__(self, weight, epoch_on, **kwargs):
        self.weight = weight
        self.epoch_on = epoch_on

    def __call__(self, epoch):
        if epoch > self.epoch_on:
            current_weight = self.weight * np.log(epoch - self.epoch_on)
        else:
            current_weight = 0

        return current_weight


class HeaviSideArcSinh:
    def __init__(self, weight, epoch_on, **kwargs):
        self.weight = weight
        self.epoch_on = epoch_on

    def __call__(self, epoch):
        if epoch > self.epoch_on:
            current_weight = self.weight * np.arcsinh(epoch - self.epoch_on)
        else:
            current_weight = 0

        return current_weight


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    epoch_range = np.arange(0, 1000)
    epoch_on = 20
    lambda_weight = 0.1

    lambda_obj = HeaviSide(0.1, 20)
    returned_weight = [lambda_obj(epoch=x) for x in epoch_range]
    plt.plot(epoch_range, returned_weight)

    lambda_obj = HeaviSideLog(0.1, 20)
    returned_weight = [lambda_obj(epoch=x) for x in epoch_range]
    plt.plot(epoch_range, returned_weight)


    lambda_obj = HeaviSideArcSinh(0.1, 20)
    returned_weight = [lambda_obj(epoch=x) for x in epoch_range]
    plt.plot(epoch_range, returned_weight)