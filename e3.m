clear all;
Fs=1000;
t=0:1/Fs:1;
xn=sin(2*pi*100*t)+2*sin(2*pi*200*t)+randn(size(t));




subplot(4,1,1);
Px=abs(fft(xn,1024)).^2/length(xn);
plot((0:1023)/1024*Fs,10*log10(Px));
title('周期图谱估计法');
xlabel('频率（Hz）');
ylabel('功率谱(dB)');

subplot(4,1,2);
% 计算信号的非偏自相关函数
max_lag = length(xn) - 1;
Rxx = xcorr(xn, max_lag, 'coeff');
% 对自相关函数进行FFT得到频谱估计
N = length(Rxx);
frequencies = 0:(Fs/N):(Fs - Fs/N);
Sxx = (fft(Rxx));
plot(frequencies, 10*log10(abs(Sxx)));
title('B-T法功率谱估计');
xlabel('频率（Hz）');
ylabel('功率谱(dB)');


subplot(4,1,3);
% 将数据分成10个小段，每段长度为100
segment_length = 100;
overlap = 50;

% 初始化功率谱估计矩阵
num_segments = 10;
Pxx_avg = zeros(2001, num_segments);

% 对每个小段进行Blackman-Tukey法功率谱估计
for i = 1:num_segments
    start_index = (i-1) * (segment_length - overlap) + 1;
    end_index = start_index + segment_length - 1;
    
    % 提取当前小段数据
    segment_data = xn(start_index:end_index);
    
    % 计算信号的非偏自相关函数
    max_lag = length(segment_data) - 1;
    Rxx = xcorr(segment_data, length(xn) - 1, 'coeff');
    
    % 保留单边部分
    
    
    % 对自相关函数进行FFT得到频谱估计
    N = length(Rxx);
    frequencies = 0:(Fs/N):(Fs - Fs/N);
    Sxx = fft(Rxx);
    
    % 存储功率谱估计
    Pxx_avg(:, i) = abs(Sxx).^2 / (Fs * length(segment_data));
end

% 计算平均功率谱
mean_Pxx = mean(Pxx_avg, 2);
% 绘制平均功率谱图
plot(frequencies, 10*log10(mean_Pxx*100000));
title('平均谱估计法');
xlabel('频率（Hz）');
ylabel('功率谱(dB)');

subplot(4,1,4);
w=hanning(256)';
Pxxx=(abs(fft(w.*xn(1:256))).^2+abs(fft(w.*xn(129:384))).^2+abs(fft(w.*xn(257:512))).^2+abs(fft(w.*xn(385:640))).^2+abs(fft(w.*xn(513:768))).^2+abs(fft(w.*xn(641:896))).^2)/(norm(w)^2*6);
plot((0:255)/256*Fs,10*log10(Pxxx));
title('Welch谱估计法');
xlabel('频率（Hz）');
ylabel('功率谱(dB)');



