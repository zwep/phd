import os
import helper.misc as hmisc
import nibabel

"""
Probleem is dat Christina haar data op UMCU servers staan..

Ik heb natuurlijk ook ACDC data..
en ook de MM1 data..


We are going with 1 example first...
"""

dsource = '/data/seb/data/mm_christina/original_image'
dtarget = '/data/seb/data/mm_christina/preproc_image'

for i_file in os.listdir(dsource):
    print(i_file)
    base_name = hmisc.get_base_name(i_file)
    base_ext = hmisc.get_ext(i_file)
    source_file = os.path.join(dsource, i_file)
    dest_file = os.path.join(dtarget, base_name)
    nib_obj = nibabel.load(source_file)
    nib_affine = nib_obj.affine
    print('struct', nib_affine)
    nib_array = nib_obj.get_fdata().T
    for ii, i_phase in enumerate(nib_array):
        dest_phase_file = dest_file + f'_{ii}_0000' + base_ext
        nibabel_obj = nibabel.Nifti1Image(i_phase.T, nib_affine)
        print(dest_phase_file)
        nibabel.save(nibabel_obj, dest_phase_file)