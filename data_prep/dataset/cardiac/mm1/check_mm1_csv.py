import pandas as pd
import os
import helper.misc as hmisc
import re

"""
So the csv has like 350 rows...

It also shows that vendor A has some 20 more images than we use...

"""

ddata = '/media/bugger/MyBook/data/m&m/OpenDataset'
dcsv = '/media/bugger/MyBook/data/m&m/OpenDataset/201014_M&Ms_Dataset_Information_-_opendataset.csv'
csv_mm1 = pd.read_csv(dcsv)

overview_vendors = []
for ii, irow in csv_mm1.iterrows():
    external_code = irow['External code']
    vendor_name = irow['VendorName']
    vendor = irow['Vendor']
    print(ii, external_code)
    for d, _, f in os.walk(ddata):
        filter_f = [x for x in f if external_code in x]
        if len(filter_f):
            # print('\t', d, vendor)
            base_dir = os.path.dirname(re.sub(ddata, '', d))
            overview_vendors.append((base_dir, vendor))


sorted_overview = sorted(overview_vendors, key=lambda x: x[1])

vendor_A = [x[0] for x in sorted_overview if x[1] == 'A']
len([x for x in vendor_A if 'label' in x.lower()])
len([x for x in vendor_A if 'validation' in x.lower()])

vendor_B = [x[0] for x in sorted_overview if x[1] == 'B']
len([x for x in vendor_B if 'label' in x.lower()])
len([x for x in vendor_B if 'validation' in x.lower()])

vendor_C = [x[0] for x in sorted_overview if x[1] == 'C']
[x for x in vendor_C if 'label' in x.lower()]
[x for x in vendor_C if 'validation' in x.lower()]

vendor_D = [x[0] for x in sorted_overview if x[1] == 'D']
[x for x in vendor_D if 'label' in x.lower()]
[x for x in vendor_D if 'validation' in x.lower()]



