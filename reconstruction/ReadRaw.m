% Testing a bit of a Matlab script


file_name = '/media/bugger/WORK_USB/2021_12_01/ca_29045/ca_01122021_1021250_19_2_transverse_dyn_100p_radial_no_triggerV4.lab'
labfid = fopen(file_name,'r');
[unparsed_labels, readsize] = fread (labfid,[16 Inf], 'uint32=>uint32');
info.nLabels = size(unparsed_labels,2);
fclose(labfid);
for ii in 1:16
    print(ii)
    print(unparsed_labels(ii, 1:10))
end

% Dit gaan we morgen verder doen...