%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Reading radial data... storing it... using a predefined file from Python
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc
addpath 'D:\gyrotools\MRecon-3.0.533 Multix\par'
addpath 'D:\gyrotools\MRecon-3.0.533 Multix'
addpath 'D:\Seb\magical_matlab'
addpath 'D:\Seb'

%% Here we load the file with all the locations of the .lab files we need
path_to_file = 'F:\data\7T_scan\cardiac\radial_file_names.csv';
radial_file_names = readtable(path_to_file);
n_lines = height(radial_file_names);

%%
starting_line = 1;
for i_index = starting_line:n_lines
    source_dir = char(radial_file_names{i_index, 1});
    dest_dir = char(radial_file_names{i_index, 2});
    file_name = char(radial_file_names{i_index, 3});
    disp(source_dir)
    proc_obj = ProcessRadialLabFile(source_dir, dest_dir, file_name);
    proc_obj.ProcessData
    proc_obj.SaveData
end


