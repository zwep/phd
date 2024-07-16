%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Script om te kijken naar retro radial data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fantoom - 8 coils
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath 'D:\gyrotools\MRecon-3.0.533 Multix';
addpath 'D:\gyrotools\MRecon-3.0.533 Multix\par';
addpath 'D:\Seb'

data_path = 'D:\Seb\vrijwilligersdata\phantom_data_retro_radial';

radial_10_dyn = fullfile(data_path, 'se_20102021_1910181_25_2_surveylr_10_phasesV4.lab');
recon_obj = MRecon(radial_10_dyn);
recon_obj.Perform
recon_obj.ShowData

radial_10_dyn = fullfile(data_path, 'se_20102021_1910181_25_2_surveylr_10_phasesV4.lab');
recon_obj = MRecon(radial_10_dyn);
recon_obj.ReadData;
recon_obj.RandomPhaseCorrection;
recon_obj.PDACorrection;
recon_obj.DcOffsetCorrection;

% Shows some nice results
imshow(abs(recon_obj.Data{1}(:, 1:8:end)), [0, 1000])

recon_obj.SortData

% Now.. the trouble starts around 8*506
% Lets cut that off and reconstruct
% This didnt work the way I wanted it to......
% Funny enough also here I dont get Kpos or RadialAngles filled in in the recon object

recon_obj = MRecon(radial_10_dyn);
recon_obj.ReadData;
recon_obj.Data{1} = recon_obj.Data{1}(:, 8*506)

recon_obj.RandomPhaseCorrection;
recon_obj.PDACorrection;
recon_obj.DcOffsetCorrection;
recon_obj.SortData;
recon_obj.GridData;
recon_obj.RingingFilter;
recon_obj.ZeroFill;
recon_obj.K2IM;
recon_obj.EPIPhaseCorrection;
recon_obj.K2IP;
recon_obj.GridderNormalization;
recon_obj.SENSEUnfold;
recon_obj.PartialFourier;
recon_obj.ConcomitantFieldCorrection;
recon_obj.DivideFlowSegments;
recon_obj.CombineCoils;
recon_obj.Average;
recon_obj.GeometryCorrection;
recon_obj.RemoveOversampling;
recon_obj.FlowPhaseCorrection;
recon_obj.ReconTKE;
recon_obj.ZeroFill;
recon_obj.RotateImage;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Matthijs data - 24  coils
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath 'D:\gyrotools\MRecon-3.0.533 Multix';
addpath 'D:\gyrotools\MRecon-3.0.533 Multix\par';

data_path = 'F:\2021_11_24\V9_28674';
radial_mathijs = fullfile(data_path, 'v9_24112021_1650264_3_2_transradialfast_retroV4.lab');
recon_obj = MRecon(radial_mathijs);

recon_obj.ReadData;
recon_obj.RandomPhaseCorrection;
recon_obj.PDACorrection;
recon_obj.DcOffsetCorrection;

% With this we visualize the information from ONE coil
imshow(abs(recon_obj.Data{1}(:, 1:24:end)), [0, 1000])

data_path = 'F:\2021_11_24\V9_28674';
radial_mathijs = fullfile(data_path, 'v9_24112021_1650264_3_2_transradialfast_retroV4.lab');
recon_obj = MRecon(radial_mathijs);
recon_obj.Perform;
recon_obj.ShowData


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Christine Data should be here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Not sure where it is


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bart data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_path = 'F:\2021_12_01\ca_29045';
addpath 'D:\gyrotools\MRecon-3.0.533 Multix';
addpath 'D:\gyrotools\MRecon-3.0.533 Multix\par';

% Check the B1 shim series
% Want to see if there is data from all receive channels..
% Somehow it isnt...

bshim_bart = fullfile(data_path, 'ca_01122021_1009432_10_2_b1shimseriesV4.lab');
b1shim_obj = MRecon(bshim_bart);
b1shim_obj.ReadData;
b1shim_obj.RandomPhaseCorrection;
b1shim_obj.PDACorrection;
b1shim_obj.DcOffsetCorrection;
b1shim_obj.SortData

% % % 
% Get the cartesian cine
dcart_bart = fullfile(data_path, 'ca_01122021_1013390_12_2_cine1slicer2_traV4.lab');
dcart_bart_sense = fullfile(data_path, 'ca_01122021_1012046_11_2_senserefscanclassicV4.lab');

cart_bart = MRecon(dcart_bart);
S = MRsense(dcart_bart_sense, dcart_bart);

body_abs = sum(abs(S.CoilData), 4);
body_angle = angle(sum(S.CoilData, 4));
body_coil_combined = body_abs .* exp(1i * body_angle);
S.BodycoilData = body_coil_combined ;  
S.Perform

cart_bart.Parameter.Recon.Sensitivities = S;
cart_bart.Perform
cart_bart.ShowData
cart_reconstructed_data = cart_bart.Data;
save('F:\bart_data\cart_bart.mat', 'cart_reconstructed_data');

% % %
% Data dimension
% 536 x 7 x 1 x 24 x 1 x 30
radial_bart_gelukt = fullfile(data_path, 'ca_01122021_1019026_17_2_transverse_retro_radialV4.lab');
radial_bart_gelukt = MRecon(radial_bart_gelukt);

radial_bart_gelukt.ReadData;
radial_bart_gelukt.RandomPhaseCorrection;
radial_bart_gelukt.PDACorrection;
radial_bart_gelukt.DcOffsetCorrection;

bart_17_2_unsorted = radial_bart_gelukt.Data{1};
radial_bart_gelukt.SortData;
radial_bart_gelukt.GridderCalculateTrajectory;
bart_17_2_sorted = radial_bart_gelukt.Data{1};
bart_17_2_angles = radial_bart_gelukt.Parameter.Gridder.RadialAngles;
bart_17_2_kpos = radial_bart_gelukt.Parameter.Gridder.Kpos;

save('F:\bart_data\bart_17_2_unsorted.mat', 'bart_17_2_unsorted');
save('F:\bart_data\bart_17_2_sorted.mat', 'bart_17_2_sorted');
save('F:\bart_data\bart_17_2_angles.mat', 'bart_17_2_angles');
save('F:\bart_data\bart_17_2_kpos.mat', 'bart_17_2_kpos');

radial_bart_gelukt = fullfile(data_path, 'ca_01122021_1019026_17_2_transverse_retro_radialV4.lab');
radial_bart_gelukt = MRecon(radial_bart_gelukt);
radial_bart_gelukt.Perform
radial_bart_gelukt.ShowData

save('F:\bart_data\bart_17_2_kpos.mat', 'bart_17_2_kpos');


% Data dimension
% 536 x 8 x 1 x 24 x 1 x 30

dradial_bart_mislukt = fullfile(data_path, 'ca_01122021_1016141_14_2_transverse_retro_radialV4.lab');
radial_bart_mislukt = MRecon(dradial_bart_mislukt);

radial_bart_mislukt.ReadData;
radial_bart_mislukt.RandomPhaseCorrection;
radial_bart_mislukt.PDACorrection;
radial_bart_mislukt.DcOffsetCorrection;

bart_14_2_unsorted = radial_bart_mislukt.Data{1};
radial_bart_mislukt.SortData;
radial_bart_mislukt.GridderCalculateTrajectory;
bart_14_2_sorted = radial_bart_mislukt.Data{1};
bart_14_2_angles = radial_bart_mislukt.Parameter.Gridder.RadialAngles;
bart_14_2_kpos = radial_bart_mislukt.Parameter.Gridder.Kpos;

save('F:\bart_data\bart_14_2_unsorted.mat', 'bart_14_2_unsorted');
save('F:\bart_data\bart_14_2_sorted.mat', 'bart_14_2_sorted');
save('F:\bart_data\bart_14_2_angles.mat', 'bart_14_2_angles');
save('F:\bart_data\bart_14_2_kpos.mat', 'bart_14_2_kpos');


% Gelukt - 536 x 6240 dimension
figure()
imshow(abs(bart_14_2_unsorted(:, 1:24:end)), [0, 500])
% Mislukt - 536 x 5736 dimension
figure()
imshow(abs(bart_17_2_unsorted(:, 1:24:end)), [0, 500])



dd_notrig_20 = 'F:\2021_12_01\ca_29045\ca_01122021_1020068_18_2_transverse_dyn_20p_radial_no_triggerV4.raw'
recon_obj = get_unsorted_data(dd_notrig_20)



dd_phantom_radial_10 = 'F:\2021_12_09\ca_29447\ca_09122021_1846218_10_2_transverse_retro_radial_50p_50phaseV4.raw'
recon_obj = get_unsorted_data(dd_phantom_radial_10)
export_unsorted_data(recon_obj, 'F:\2021_12_09\mat_data\ca_09122021_1846218_10_2_transverse_retro_radial_50p_50phaseV4.mat')


dd_phantom_radial_10 = 'D:\Seb\rawdata_export_test\ca_09122021_1839250_9_2_transverse_retro_radial_100pV4.raw'
recon_obj = MRecon(dd_phantom_radial_10)
recon_obj.ReadData

dd_phantom_radial_10 = 'D:\Seb\rawdata_export_test\ca_09122021_1803362_7_2_transverse_retro_radial_010pV4.raw'
recon_obj = MRecon(dd_phantom_radial_10)
recon_obj.ReadData



%% % % % % Test for number of spokes

addpath 'D:\gyrotools\MRecon-3.0.533 Multix';
addpath 'D:\gyrotools\MRecon-3.0.533 Multix\par';
addpath 'D:\Seb';

dd_test = 'F:\2022_01_05\fa_30068\fa_05012022_1743157_1_1_transverse_retro_radialV4.raw'
recon_obj = get_unsorted_data(dd_test)

export_unsorted_data(recon_obj, 'F:\2021_12_09\mat_data\ca_09122021_1846218_10_2_transverse_retro_radial_50p_50phaseV4.mat')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Matthijs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath 'D:\Seb';
addpath 'D:\gyrotools\MRecon-3.0.533 Multix';
addpath 'D:\gyrotools\MRecon-3.0.533 Multix\par';

data_path = 'F:\2022_01_19\ca_30409';
dest_path = 'F:\2022_01_19\mat_data';
my_files = dir(fullfile(data_path,'*.lab'));

for k = 1:length(my_files)
  base_name = my_files(k).name;
  full_name = fullfile(data_path, base_name);
  dest_full_name = fullfile(dest_path, base_name);
  disp(full_name)
  if exist(dest_full_name, 'file') == 2
	continue
  else
    try
		recon_obj = get_unsorted_data(full_name);
		% Replace the lab extension with mat
		export_unsorted_data(recon_obj, dest_full_name(1:end-3), "mat");
	except
		disp('Error....')
	end
   end
end

path_file = 'F:/data/7T_scan/cardiac/2020_11_04/V9_13975/v9_04112020_1759228_14_2_4chradialV4.lab'
MR = Seb_Recon(path_file);
MR.Perform
res = sum(abs(MR.Data), 4);
imshow(res(:,:, 1, 1, 1, 2), [0, 3000])


%%%%% Fantom

addpath 'D:\gyrotools\MRecon-3.0.533 Multix';
addpath 'D:\gyrotools\MRecon-3.0.533 Multix\par';

path_file = 'F:\2022_02_16\ca_31817\ca_16022022_1653241_8_1_transverse_retro_radialV4.lab';
MR = MRecon(path_file);
MR.ShowData
MR.Perform

MR.ReadData
MR.Data

path_file = 'F:\2022_02_16\ca_31817\ca_16022022_1650344_7_2_cine1slicer2_traV4.lab';
MR = MRecon(path_file);
MR.ReadData
MR.Data


%%%%% both Alexander's data
clear all
clc
addpath 'D:\gyrotools\MRecon-3.0.533 Multix\par'
addpath 'D:\gyrotools\MRecon-3.0.533 Multix'
addpath 'D:\Seb'

data_path = 'G:\2022_03_07\V9_32580';
dest_path = 'G:\2022_03_07\V9_32580\matdata';
my_files = dir(fullfile(data_path,'*.lab'));

for k = 1:length(my_files)
  base_name = my_files(k).name;
  full_name = fullfile(data_path, base_name);
  dest_full_name = fullfile(dest_path, base_name);
  disp(full_name)
  if exist(dest_full_name, 'file') == 2
	continue
  else
    try
        proc_obj = ProcessRadialLabFile(full_name, dest_path, base_name);
        proc_obj.ProcessData
        proc_obj.SaveData        
	except
		disp('Error....')
	end
   end
end