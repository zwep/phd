import helper.plot_class as hplotc
from skimage.data import cells3d

A = cells3d()

hplotc.SlidingPlot(A[None, None, None], ax_3d=True)