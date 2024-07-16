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
ddest = '/home/bugger/Documents/7Tscanning/vrijwilligers_batch_3_4/uitnodiging'

n_space = 30
datum_prikker_result = read_ods(ddata_ods, 'sheet')
date_time = []
for i, irow in datum_prikker_result.iterrows():
    iname = irow['name'].strip()
    iemail = irow['email'].strip()
    idate = irow['date'].strip()
    idate_obj = datetime.datetime.strptime(idate, "%Y-%m-%d")
    idate_str = idate_obj.strftime("%a %d %b")
    itime = [x.strip() for x in irow['time'].split('–')]
    standaard_tekst = f'{iemail}\nCardiac onderzoek planning\nBeste {iname}, \n\nBedankt voor het opgeven van je beschikbaarheid in de datumprikker. Naar aanleiding van de gegeven beschikbaarheid wil ik je inplannen op {idate_str} van {itime[0]} tot {itime[1]}. Kan je voor jezelf checken of deze tijd voor jou lukt? Zo niet, laat dan even weten waarom niet dan kan ik makkelijker een alternatief aanbieden. \n' \
                      f'Het is voor de planning prettig als je er +-15 min van te voren bent. De week voor de geplande datum ontvang je nog een mail met route beschrijving en wat extra informatie.\n\nMet vriendelijke groet, \nSeb Harrevelt'

    print(iname, (n_space - len(iname)) * ' ', idate, (n_space - len(idate)) * ' ', itime)
    dest_file = os.path.join(ddest, iname + '.txt')
    # date_time.append((idate_str, itime[0]))
    date_time.append((idate_str, idate_obj))
    with open(dest_file, 'w') as f:
        f.write(standaard_tekst)
