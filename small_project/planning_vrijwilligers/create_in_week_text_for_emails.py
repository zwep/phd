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
ddest = '/home/bugger/Documents/7Tscanning/vrijwilligers_batch_3_4/herinering_en_route'

n_space = 30
datum_prikker_result = read_ods(ddata_ods, 'sheet')
date_time = []
for i, irow in datum_prikker_result.iterrows():
    if irow['date']:
        iname = irow['name'].strip()
        iemail = irow['email'].strip()
        idate = irow['date'].strip()
        idate_obj = datetime.datetime.strptime(idate, "%Y-%m-%d")
        idate_str = idate_obj.strftime("%a %d %b")
        idate_file_str = idate_obj.strftime("%Y%m%d")
        itime = [x.strip() for x in irow['time'].split('–')]
        standaard_tekst = f'{iemail}\nCardiac onderzoek herinnering\nBeste {iname},\n\nBij deze de extra informatie mail (en herinnering) aan de scan sessie die deze week gaat plaatsvinden. Check voor de exacte datum/tijd de vorige mail die ik je hebt gestuurd.\n\nOp de dag zelf, voordat je de scanner in gaat, moet je nog even twee formulieren invullen. Het eerste formulier komt redelijk overeen met de eisen of je wel de MRI in mag, in het andere formulier geef je toestemming voor het uitvoeren van de scan en het gebruik van je gegevens.\n\nAls je in de MRI-scanner ligt kun je op elk moment met mij praten via het communicatie systeem. Hiermee zal ik je ook een aantal keer vragen om een breath-hold te doen. Dit houdt in dat je rustig uit ademt en dit dan 10-15 sec vast houdt. Precies in die periode maak ik een scan zodat er geen ademhalingsbeweging is in het gemaakte plaatje. Deze handeling zal ik je nogmaals uitleggen voordat je de scanner in gaat.\n\nIn de bijlage vindt je verder nog de route beschrijving om bij de 7T te komen. Mocht je in de buurt zijn en niet precies weten hoe je er komt, bel mij dan even. Dan komen we er het snelst uit, het ziekenhuis kan namelijk als een doolhof voelen.\n\nQua kleding kun je voor de MRI het beste een shirt of trui dragen zonder metalen knoopjes of frutsels er aan. Lukt dit niet? Dan hebben we een MR-proof shirt voor je. Voor veel broeken kan dit wat lastig zijn, dus vanuit het ziekenhuis hebben we ook een fancy ziekenhuis broek voor je, die is sowieso MR-proof en one-size-fits-all. Verder, als je graag een BH-draagt, dan het liefst eentje zonder metalen beugel er in. Er zijn  omkleed ruimtes beschikbaar mocht je daar gebruik van willen maken.\nVanwege de corona maatregelen vragen we je vriendelijk om een mondkapje te dragen bij binnenkomst. Deze moet je af doen als je de scanner in gaat, want er zit vaak een metalen strip in.\n\nBuiten de scanner zal er een tussendoortje voor jullie liggen, en staat er koffie/thee voor jullie klaar.\n\nEven in de herhaling, vanuit het ziekenhuis zijn er de volgende eisen wanneer je niet mee mag doen met een MRI onderzoek:\n\tje iets van metaal in je lichaam hebt	\n\tje een beugel hebt                                     (een draad achter je tanden hebben is prima)\n\tje een niet-verwijderbare piercing hebt\n\tje zwanger bent of denkt zwanger te zijn\n\tje last hebt van claustrofobie of het benauwd krijgt in kleine ruimtes\n\nAls er verder nog vragen zijn, dan hoor ik ze graag.\n\nMet vriendelijke groet,\nSeb Harrevelt\n\ntel: 06 303 62 178'

        print(iname, (n_space - len(iname)) * ' ', idate, (n_space - len(idate)) * ' ', itime)
        dest_file = os.path.join(ddest, idate_file_str + "_" + iname + '.txt')
        # date_time.append((idate_str, itime[0]))
        date_time.append((idate_str, idate_obj))
        with open(dest_file, 'w') as f:
            f.write(standaard_tekst)