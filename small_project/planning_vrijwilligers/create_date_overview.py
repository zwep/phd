import os
import pandas
import datetime
from pandas_ods_reader import read_ods
import locale
locale.setlocale(locale.LC_TIME, 'nl_NL.UTF-8')

"""
We got a planning in a csv file.. dont want to manually change the dates etc
"""


ddata_ods = '/home/bugger/Documents/7Tscanning/vrijwilligers_batch_3_4/resultaat_datumprikker_batch_3_4.ods'
ddest = '/home/bugger/Documents/7Tscanning/vrijwilligers_batch_3_4'

n_space = 30
datum_prikker_result = read_ods(ddata_ods, 'sheet')
date_time = []
for i, irow in datum_prikker_result.iterrows():
    iname = irow['name'].strip()
    iemail = irow['email'].strip()
    idate = irow['date'].strip()
    idate_obj = datetime.datetime.strptime(idate, "%Y-%m-%d")
    idate_str = idate_obj.strftime("%a %d %b")
    # itime = [x.strip() for x in irow['time'].split('–')]
    date_time.append((idate_str, idate_obj))

date_time_list = [x[0] for x in sorted(set(date_time), key=lambda x: x[1])]
# date_time_list = ['\t'.join(x) for x in set(date_time)]
with open(os.path.join(ddest, 'date_time_list' + '.txt'), 'w') as f:
    f.write('\n'.join(date_time_list))