"""

wget https://s3.aiforoncology.nl/direct-project/lpdnet.zip --no-check-certificate
wget https://s3.aiforoncology.nl/direct-project/iterdualnet.zip --no-check-certificate
wget https://s3.aiforoncology.nl/direct-project/conjgradnet.zip --no-check-certificate
wget https://s3.aiforoncology.nl/direct-project/rim.zip --no-check-certificate
wget https://s3.aiforoncology.nl/direct-project/varnet.zip --no-check-certificate
wget https://s3.aiforoncology.nl/direct-project/jointicnet.zip --no-check-certificate
wget https://s3.aiforoncology.nl/direct-project/xpdnet.zip --no-check-certificate
wget https://s3.aiforoncology.nl/direct-project/kikinet.zip --no-check-certificate
wget https://s3.aiforoncology.nl/direct-project/multidomainnet.zip --no-check-certificate
wget https://s3.aiforoncology.nl/direct-project/unet.zip --no-check-certificate

"""

import os
from objective_configuration.reconstruction import DPRETRAINED
import helper.misc as hmisc

file_list = [x for x in os.listdir(DPRETRAINED) if x.endswith('zip')]

for i_zip in file_list:
    base_name = hmisc.get_base_name(i_zip)
    zip_path = os.path.join(DPRETRAINED, i_zip)
    dest_path = os.path.join(DPRETRAINED, base_name)
    if os.path.isdir(dest_path):
        continue
    else:
        os.makedirs(dest_path)
        os.system(f'unzip {zip_path} -d {DPRETRAINED}')