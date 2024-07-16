import os
import re
import sys
import subprocess
import pathlib

"""
Script to push data to surfdrive.

You need to enable this on your own surfdrive account. See for more details:

https://wiki.surfnet.nl/display/SURFdrive/Bestanden+benaderen+via+WebDAV
"""


def check_presence_url(dest_url):
    """
    Checks whether a specified folder/file already exists

    :param dest_url: an url to a surfdrive location
    :return: True/False if it exists or not
    """
    response = False
    curl_check_dir = f"curl -u {encoder}:{decoder} --head --silent '{dest_url}' | head -n 1"
    curl_response = subprocess.check_output(curl_check_dir, shell=True).decode().strip()
    if 'OK' in curl_response:
        response = True

    return response


def make_dir_surfdrive(dir_name):
    """
    Surfdrive needs to have locations defined on order to store them there..
    If the folder is already there, then we report that with no further action

    :param dir_name: The name of the directory
    :return: None
    """
    dest_url = f"https://surfdrive.surf.nl/files/remote.php/nonshib-webdav/{dir_name}"
    # Make sure that spaces are properly copied..
    dest_url = re.sub('\s+', '%20', dest_url)
    curl_response = check_presence_url(dest_url)
    if curl_response:
        print(f"Directory already there: {dir_name}")
    else:
        curl_create_dir = f'curl -u {encoder}:{decoder} --silent -X MKCOL "{dest_url}"'
        subprocess.check_output(curl_create_dir, shell=True)


def move_file_surfdrive(source_name, target_name):
    """
    Move a specific file (source_name) to a specific location (target_name).
    If it is already there, then we report that with no further action. Hence, we do not overwrite existing files for now.

    :param source_name: absolute path (for your PC) to the source file
    :param target_name: absolute path (for surfdrive) to the target file
    :return: None
    """
    target_url = f"https://surfdrive.surf.nl/files/remote.php/nonshib-webdav/{target_name}"
    # Make sure that spaces are properly copied..
    target_url = re.sub('\s+', '%20', target_url)
    curl_response = check_presence_url(target_url)
    if curl_response:
        print(f"File already there: {target_name}")
    else:
        curl_move_file = f"curl -u {encoder}:{decoder} -T '{source_name}' '{target_url}'"
        subprocess.check_output(curl_move_file, shell=True)


if __name__ == "__main__":
    # This is the path to my data folder that I wanted to upload.
    ddata = os.path.expanduser('/media/bugger/MyBook/data/7T_scan/cardiac')

    # This location retrieves the credentials for surfdrive
    # full_path_decoder = os.path.expanduser('~/.surfdrive/decoder')
    full_path_decoder = os.path.expanduser('~/.surfdrive/decoder2')
    with open(full_path_decoder, 'r') as f:
        encoder, decoder = [x.strip() for x in f.read().split(", ")]

    """
    Here we loop/walk over the directories and files.
    There is one important parameter here, that is `n_parts`.
    
    This makes sure that the destination path (for surfdrive) does not contain
    the first `n_part` directories that you have on your file system. 
    Example:
    n_parts = 4
    path = '/A/B/C/D/E/F'
    path_parts = pathlib.Path(path).parts[n_parts:]
    print('Processing directory ', path_parts)
        'D', 'E', 'F'
    
    By choosing these options, the folder structure from `D` and deeper will be re-created on surfdrive.
    """
    dprocessed_files = '/media/bugger/MyBook/data/7T_scan/cardiac/processed_files.txt'
    with open(dprocessed_files, 'r') as f:
        processed_files = f.readlines()

    total_file_count = 0
    for d, _, f in os.walk(ddata):
        n_files = len(f)
        if n_files > 0:
            total_file_count += n_files

    # Below
    n_parts = 6
    counter = 0
    break_ind = False
    current_file_count = 0
    for d, _, f in os.walk(ddata):
        n_files = len(f)
        if n_files > 0:
            # This creates all the subparts..
            path_parts = pathlib.Path(d).parts[n_parts:]
            print('Processing directory ', path_parts)
            for i in range(1, len(path_parts)+1):
                dest_dir = os.path.join(*path_parts[:i])
                make_dir_surfdrive(dest_dir)
            for i_file in f:
                current_file_count += 1
                print('File count  ', current_file_count, "/", total_file_count)
                source_path = os.path.join(d, i_file)
                if source_path + '\n' in processed_files:
                    print('Found ', source_path)
                    continue
                else:
                    target_path = os.path.join(*path_parts, i_file)
                    move_file_surfdrive(source_path, target_path)
                    # Now add it.
                    if source_path not in processed_files:
                        with open(dprocessed_files, 'a') as f:
                            f.write(f"{source_path}\n")
