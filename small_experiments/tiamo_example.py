
"""
SoS recon with each single file
"""
import matplotlib.pyplot as plt
import itertools

res_solo = []
n_files = len(sel_file_list)
for x in range(n_files):
    derp = np.sqrt((np.abs(loaded_img_files[x]) ** 2).sum(axis=0))
    res_solo.append(derp)

hplotf.plot_3d_list(np.stack(res_solo, axis=0)[None], augm='np.abs', subtitle=list([range(n_files)]),
                    title='SoS for each single file')

"""
Sum of squares reconstruction with ALL the shimsettings combo
"""

res_sos_img_combination = []
n_files = len(sel_file_list)
for x, y in itertools.product(range(n_files), range(n_files)):
    # derp = np.abs((np.concatenate([res[x], res[y]], axis=0) ** 2)).sum(axis=0)
    derp = np.sqrt((np.abs(np.concatenate([loaded_img_files[x], loaded_img_files[y]], axis=0)) ** 2).sum(axis=0))
    res_sos_img_combination.append(derp)

hplotf.plot_3d_list(np.stack(res_sos_img_combination, axis=0)[None], augm='np.abs', subtitle=[list(itertools.product(range(n_files), range(n_files)))],
                    title='SoS for each combination of shimsettings')

# Okay now we combine the two kspaces of differnet shim settings
# Now we do the same... but combine the kspace data in half space
import helper.array_transf as harray
comb_kspace_list = []
for index_i in range(n_files):
    for index_j in range(n_files):
        one_img_combi_sos = np.sqrt((np.abs(np.concatenate([loaded_img_files[index_i], loaded_img_files[index_j]], axis=0)) ** 2).sum(axis=0))

        result_kspace_comb = np.zeros(loaded_img_files[0].shape)
        nchan, ny, nx = result_kspace_comb.shape

        res_k_i = harray.transform_image_to_kspace_fftn(loaded_img_files[index_i], dim=(-2, -1))
        res_k_j = harray.transform_image_to_kspace_fftn(loaded_img_files[index_j], dim=(-2, -1))
        derp_k = np.zeros(res_k_j.shape, dtype=complex)
        derp_k[:, ::2] = res_k_j[:, ::2]
        derp_k[:, 1::2] = res_k_i[:, 1::2]
        res_combo = harray.transform_kspace_to_image_fftn(derp_k, dim=(-2, -1))
        single_k_sos_combo = np.sqrt((np.abs(res_combo) ** 2).sum(axis=0))
        # hplotf.plot_3d_list(single_k_sos_combo[None], augm='np.abs')

        comb_kspace_list.append(res_combo)



temp_array = [np.sqrt((np.abs(x) ** 2).sum(axis=0)) for x in comb_kspace_list]
hplotf.plot_3d_list(np.stack(temp_array, axis=0)[None], augm='np.abs', subtitle=[list(itertools.product(range(n_files), range(n_files)))],
                    title='Result of SoS with kspace comb')

