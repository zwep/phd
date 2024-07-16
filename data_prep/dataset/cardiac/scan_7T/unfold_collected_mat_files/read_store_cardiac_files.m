%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc
addpath 'D:\gyrotools\MRecon-3.0.533 Multix\par'
addpath 'D:\gyrotools\MRecon-3.0.533 Multix'
addpath 'D:\Seb\magical_matlab'
addpath 'D:\Seb'

% Here we load the file with all the locations of the .lab files we need
path_to_file = 'F:\data\7T_scan\cardiac\radial_file_names.csv';
radial_file_names = readtable(path_to_file);
n_lines = height(radial_file_names);

%
starting_line = 1;
n_lines = 3;
for i_index = starting_line:n_lines
    v_number = char(radial_file_names{i_index, 1});
    file_name = char(radial_file_names{i_index, 2});
    directory = char(radial_file_names{i_index, 3});
    _ = char(radial_file_names{i_index, 4}); # date_time
    _ = char(radial_file_names{i_index, 5}); # ext
    slice_name = char(radial_file_names{i_index, 6});
    sense_file = char(radial_file_names{i_index, 7});
    # Cut out the /media/-path
    # Put in the F:/ thing
    dest_file = char(radial_file_names{i_index, 8});
    # LIke this..?
    disp(dest_file(21:end))
    disp(directory(21:end))
    dest_file = fullfile("F:/", dest_file(21:end))
    directory = fullfile("F:/", directory(21:end))

    # Create destination directories
    source_path = fullfile(directory, file_name)
    sense_path = fullfile(directory, sense_file)

    proc_obj = ProcessLabFile(source_path, sense_path, dest_file);
    proc_obj.ProcessData
    proc_obj.SaveData
end
