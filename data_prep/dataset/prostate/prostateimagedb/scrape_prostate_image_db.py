
import numpy as np
import time
import os
import requests
from bs4 import BeautifulSoup
import urllib.parse

main_url = 'https://prostatemrimagedatabase.com/Database/'
target_path = '/media/bugger/MyBook/data/prostatemriimagedatabase_request'
target_link_table = '/media/bugger/MyBook/data/prostatemriimagedatabase_request/patient_link.txt'

current_data_list = os.listdir(target_path)
current_patient_id_list = sorted(list(set([x.split('.')[0] for x in current_data_list if x.endswith('nrrd')])))
# Remove last one, it might have been half-acquired
_ = current_patient_id_list.pop()

s = requests.Session()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'}
response_main = s.get(main_url, headers=headers)
print(response_main.status_code)
soup_obj_main = BeautifulSoup(response_main.text)

patient_table = soup_obj_main.find_all('tr')
counter = 0
for i_patient in patient_table:
    temp_text = i_patient.get_text(separator=',')
    if 'View' in temp_text:
        view_patient = i_patient.find_all(href=True)
        # We should have one link associated with 'View'
        view_patient_href = [x['href'] for x in view_patient if 'View' in x.text]
        if len(view_patient_href) == 1:
            sel_view_patient_href = view_patient_href[0]
            patient_id = sel_view_patient_href.split('/')[0]
            if patient_id not in current_patient_id_list:
                # Create the link to view this specific patient
                view_patient_url = urllib.parse.urljoin(main_url, sel_view_patient_href)

                view_content = s.get(view_patient_url, headers=headers)
                view_soup = BeautifulSoup(view_content.text)
                view_patient_seroes = view_soup.find_all('tr')
                for i_view in view_patient_seroes:
                    temp_text = i_view.get_text(separator=',')
                    if 'Download Series' in temp_text:
                        download_series = i_view.find_all(href=True)
                        description_series = i_view.find_all('td')
                        download_series_href = [x['href'] for x in download_series if 'Download Series' in x.text]
                        # Now we can download each series from the patient
                        for i_series in download_series_href:
                            # Get the description of this series..
                            series_number = i_series.split('/')[0]
                            selected_description = [x for x in description_series if series_number in x.get_text()]
                            # We should have only one....
                            if len(selected_description) == 1:
                                # Extract the actual description text
                                description_text_list = [x.strip() for x in selected_description[0].get_text(separator='*').split('*') if x.strip()]
                                index_series = description_text_list.index(f'Series {series_number}')
                                description_text = description_text_list[index_series + 1]

                                # Extract the ....html part from the url
                                temp_url = os.path.dirname(view_patient_url)
                                download_series_url = urllib.parse.urljoin(temp_url + '/', i_series)


                                download_content = s.get(download_series_url, headers=headers)
                                download_soup = BeautifulSoup(download_content.text)
                                download_series_extension = download_soup.find_all('tr')
                                # Now find the right extension and its name to create the final url to download it...
                                for isubsub_lines in download_series_extension:
                                    # Get the nrrd file name
                                    nrrd_file_name = [x for x in isubsub_lines.get_text(separator='*').split('*') if 'nrrd' in x]
                                    if nrrd_file_name:
                                        break
                                temp_url = os.path.dirname(download_series_url)
                                file_name = nrrd_file_name[0]
                                file_name_no_ext, ext = os.path.splitext(file_name)
                                download_nrrd = urllib.parse.urljoin(temp_url + '/', file_name)

                                sleep_time = np.random.randint(30, 60)
                                print(f'\t\t\t Sleeping for {sleep_time}')
                                time.sleep(sleep_time)
                                print(f'Downloading from {download_nrrd}')
                                download_file = s.get(download_nrrd, headers=headers, allow_redirects=True)
                                print('Storing file ', file_name, '\t', description_text)
                                target_file = os.path.join(target_path, file_name)
                                with open(target_file, 'wb') as f:
                                    for chunk in download_file.iter_content(100000):
                                        f.write(chunk)

                                with open(target_link_table, 'a') as f:
                                    f.write('\t'.join([file_name_no_ext, description_text]))
                            else:
                                print('We have found more than one description ', series_number)
            else:
                print('We have already processed ', patient_id)
        else:
            print('We have found more than one view ', view_patient)