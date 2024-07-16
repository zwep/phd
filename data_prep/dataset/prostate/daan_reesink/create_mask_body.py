"""
Here I need to put the script on how I create the mask...
"""

if (mask_array.sum() == 0) or (not os.path.isfile(mask_file_name)):
    if pixel_array.ndim == 2:
        print('Shape pixel array ', pixel_array.shape)
        initial_mask = harray.get_treshold_label_mask(pixel_array)
        mask_array, verts = harray.convex_hull_image(initial_mask)
    else:
        print('Number of dimensions is now.. ', ndim)
        initial_mask = [harray.get_treshold_label_mask(x) for x in pixel_array]
        mask_array, verts = zip(*[harray.convex_hull_image(x) for x in initial_mask])
        mask_array = np.array(mask_array)

    np.save(mask_file_name, mask_array)
    # np.save(mask_file_name, mask_obj.mask)