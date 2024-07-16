clear all; close all; clc;
addpath('subroutines');
tic
nports = 16;
filename = '16 Channel Loop Array small';
pathname = 'D:\Sim4Life\Sim4Life Export\'; % Set path
port_selection = [1:nports];
load(strcat(pathname,filename,'_ProcessedData.mat'));
clear ProcessedData.Q % Not 10g averaged, remove
head_weight = 5; % kg
offset = 13;

% Show conductivity distributions 
figure(1); 
subplot(1,3,1); imagesc(ProcessedData.sigma(:,:,round(end/2)+offset));
subplot(1,3,2); imagesc(squeeze(ProcessedData.sigma(:,round(end/2),:)));
subplot(1,3,3); imagesc(squeeze(ProcessedData.sigma(round(end/2),:,:)));

sig_val = 0.5026; % target_mask is selected based on conductivity, gray matter
Targ_mask = zeros(size(ProcessedData.sigma));
Targ_mask(abs(ProcessedData.sigma-sig_val)<0.001) = 1;

% Get B1+ and B1- per channel
B1plus = squeeze(ProcessedData.B1(:,:,:,1,:))*1e6; %select B1+, go to muT
B1minus = squeeze(ProcessedData.B1(:,:,:,2,:))*1e6;  %select B1-, go to muT
B1m = reshape(B1minus,[size(B1minus,1)*size(B1minus,2)*size(B1minus,3) size(B1minus,4)]); % flatten B1-
B1p = reshape(B1plus,[size(B1plus,1)*size(B1plus,2)*size(B1plus,3) size(B1plus,4)]); % flatten B1+


% determine noise covariance matrix Psi for SNR calculation
Ex_flat = double(reshape(ProcessedData.Ex,[size(ProcessedData.Ex,1)*size(ProcessedData.Ex,2)*size(ProcessedData.Ex,3) size(ProcessedData.Ex,4) size(ProcessedData.Ex,5)]));
Ey_flat = double(reshape(ProcessedData.Ey,[size(ProcessedData.Ey,1)*size(ProcessedData.Ey,2)*size(ProcessedData.Ey,3) size(ProcessedData.Ey,4) size(ProcessedData.Ex,5)]));
Ez_flat = double(reshape(ProcessedData.Ez,[size(ProcessedData.Ez,1)*size(ProcessedData.Ez,2)*size(ProcessedData.Ez,3) size(ProcessedData.Ez,4) size(ProcessedData.Ex,5)]));


% Calculate power deposition matrix Psi, for calculation of total power deposition and
% SNR
for i =1:nports
    for j =1:nports
        Psi(i,j) = sum(
        ProcessedData.sigma(:)/2
        .*((Ex_flat(:,i)).*conj(Ex_flat(:,j))+(Ey_flat(:,i)).*conj(Ey_flat(:,j))+(Ez_flat(:,i)).*conj(Ez_flat(:,j)))
        ).*(ProcessedData.res^3);
    end
end

% Calculate SNR
SNR_flat = abs(sqrt(sum(conj(B1m)*inv(Psi).*B1m,2))); 
SNR = reshape(SNR_flat,[size(B1minus,1) size(B1minus,2) size(B1minus,3)]);

% Calculate B1 and SAR for random excitation vectors
for iShim = 1:1000
    exc = rand(nports,1).*exp(1j.*2.*pi.*rand(nports,1));
    Pdep_rand(iShim) = exc'*Psi*exc;
    Pcorrection = 3.2./(Pdep_rand(iShim)./head_weight);  % correction factor to set head SAR to 3.2 W/kg
    % Iterate over all VOPs 
    for iVOP = 1:size(ProcessedData.VOPm,1)
        VOPi = squeeze(ProcessedData.VOPm(iVOP,:,:));
        SARvop(iVOP) = exc'*VOPi*exc;
    end
    % Calculate peak SAR
    SARmax_rand(iShim) = abs(max(SARvop)) .* Pcorrection;
    display(iShim);
    
    % Calculate B1+ for give random excitation vector exc
    B1p_shimmed = B1p*exc.*Pcorrection.^2;
    B1p_shimmed_3D = reshape(B1p_shimmed,[size(B1plus,1) size(B1plus,2) size(B1plus,3)]);
    
    % Calculate only in mask
    B1p_masked = B1p_shimmed_3D.*Targ_mask; 
    B1p_masked = B1p_masked(B1p_masked~=0);
    B1p_masked(isnan(B1p_masked))=0;
    
    % Calculate CoV
    B1p_cov(iShim) = mean(abs(B1p_masked))./std(abs(B1p_masked)).*100;
end


% Show SNR (unit uT/sqrt(W))
figure(2);
subplot(1,3,1); imagesc(abs(SNR(:,:,round(end/2)+offset))); axis off; colorbar; title('Intrinsic SNR transverse slice')
subplot(1,3,2); imagesc(abs(squeeze(SNR(:,round(end/2),:)))); axis off; colorbar; title('Coronal slice')
subplot(1,3,3); imagesc(abs(squeeze(SNR(round(end/2),:,:)))); axis off; colorbar; title('Sagittal slice')

% Show random peak SAR vs B1+ CoV
figure(3);
plot(B1p_cov,SARmax_rand,'.'); 
xlabel('B_1^+ CoV [%]'); ylabel('Peak SAR [W/kg]'); title('B_1^+ uniformity and peak SAR for random RF shims')

% Show random peak SAR vs global SAR
figure(4);
histogram(abs(SARmax_rand));
xlabel('W/kg'); ylabel('Counts');
