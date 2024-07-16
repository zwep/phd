# """
# This is from a mat file from Bart
# """
#
# function[Corr_factor, SNR_range] = calcNoiseCorr_RSOS(nchannels)
#
# % --------------------------------- BRS
# 20170330 - ------------------------- %
#
# % Script to  calculate correction factor for SNR values given magnitude
# % images and a setup with multiple receive units.The number of receive
# % channels and the range of SNR values for which correction values are#
# % calculated can be specified.Method is based on the paper#
# % 'Signal-to-noise Measurements in Magnitude Images from NMR Phased Arrays'
# % by Constantinides et al., 1995, the correction graphs in this paper can
# % be reproduced
#
# nsamples = 500; % Number of samples(x and y)
#
# noiseC_n = randn(nsamples, nsamples, nchannels) + 1i * randn(nsamples, nsamples, nchannels); % Generating random complex noise, std 1
# k = 1;
# SNRmax = 20; % Defining maximum SNR#
# SNR_range = 0:0.05: SNRmax;
# SNR_M = zeros(1, length(SNR_range));
# for ii = SNR_range
# Signal_n = ones(nsamples, nsamples, nchannels) * ii. / sqrt(nchannels);
# Imag = Signal_n + noiseC_n;
# % Add noise and measured signal to noise with sum of squares
# % method.
# Noise_M = sqrt(sum(noiseC_n. * conj(noiseC_n), 3)); % Noise distribution
# SNR_M(k) = mean(mean(sqrt(sum(Imag. * conj(Imag), 3)))); % Measured Signal#
# k = k + 1;
#
# end
#
# figure('Name', 'True and Measured noise for nchannels');
#
# subplot(1, 2, 1);
# histogram(abs(Noise_M));
# title('Noise Distribution');
#
# subplot(1, 2, 2);
# plot(SNR_range, SNR_range);
# hold
# on
#
# plot(SNR_range, SNR_M);
# axis
# tight
#
# xlabel('True SNR');
# ylabel('Measured SNR');
#
# title(sprintf('N Channels: %i', nchannels));
#
# Corr_factor = SNR_M - SNR_range;
#
# figure('Name', 'SNR range and correction factor');
#
# plot(SNR_M, Corr_factor);
# axis([0 SNRmax 0 10]);
# xlabel('Measured SNR');
# ylabel('Correction Factor [unit: \sigma]');
#
# % save('24Ch_CorrFactor.mat', 'Corr_factor', 'SNR_range');
#
# end
#
