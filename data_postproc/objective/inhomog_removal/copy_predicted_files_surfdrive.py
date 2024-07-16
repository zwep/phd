import shutil
import os
import re

"""
Shutil copy some files because we can
"""

ddorig = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Seb_pred'
ddest = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Seb_biasfield_pred_for_surfdrive'

for d, _, f in os.walk(ddorig):
    if len(f):
        patient_id = os.path.basename(d)
        dest_patient = os.path.join(ddest, patient_id)
        if not os.path.isdir(dest_patient):
            os.makedirs(dest_patient)

        filter_f = [x for x in f if ('uncor' in x) or ('biasf' in x)]

        for i_file in filter_f:
            orig_file = os.path.join(ddorig, patient_id, i_file)
            i_file_new = re.sub('_15_juli', '', i_file)
            dest_file = os.path.join(dest_patient, i_file_new)

            shutil.copy(orig_file, dest_file)