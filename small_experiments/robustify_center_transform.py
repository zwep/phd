"""
The center transform is fragile to images to move over the image boundary...

We are going to investigate that and make it better
"""


# # # SOme misc code...
import helper.dummy_data
import helper.plot_class as hplotc
import helper.array_transf as harray
A, _ = helper.dummy_data.get_elipse(20, 20)
affine_coords, crop_coords = harray.get_center_transformation_coords(A)
affine_corners = [(-10, 10), (10, -10), (-10, -10), (10, 10)]
affine_edges = [(-10, 0), (0, -10), (10, 0), (0, 10)]
# affine_corners = [(-3, -4), (-2, -5), (-4, -7)]
# affine_corners = [(3, 4), (2, 5), (4, 7)]
affine_edges = []
for affine_coords in affine_edges + affine_corners:
    pass
    A_shifted = harray.apply_center_transformation(x=A, affine_coords=affine_coords, crop_coords=crop_coords)
    hplotc.ListPlot([A, A_shifted])