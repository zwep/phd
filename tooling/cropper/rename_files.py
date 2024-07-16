
import os
import pandas as pd
import shutil
import helper.misc as hmisc

dcsv = '/home/bugger/Documents/Politie/PSI/namen_nummers.csv'
ddata = '/home/bugger/Documents/Politie/PSI/klas_proc'
ddest = '/home/bugger/Documents/Politie/PSI/klas_proc_named/'

namen_csv = pd.read_csv(dcsv)
namen_csv = namen_csv.set_index('Naam')
namen_dict = namen_csv.to_dict()

list_files = os.listdir(ddata)
for i_file in list_files:
    naam = hmisc.get_base_name(i_file)
    nummer = namen_dict['Nummer'][naam]
    achternaam = namen_dict['Achternaam'][naam]
    source_file = os.path.join(ddata, i_file)
    dest_file = os.path.join(ddest, f"{str(int(nummer))}-{achternaam.lower()}.jpg")
    shutil.copy(source_file, dest_file)