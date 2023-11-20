
%%
%高斯白噪声
clear all;
rng(42);
mu = [0, 0, 0]; % 均值
sigma = [1, 0.5, 2]; % 标准差
num_samples = 1000; % 样本数量

for i = 1:3
    figure(i);
    white_noise = mu(i) + sigma(i) * randn(1, num_samples);
    subplot(4,1,1)
    plot(white_noise);
    title('高斯分布的白噪声图');
    xlabel('样本');
    ylabel('值');
   
    % 绘制直方图
    subplot(4,1,2)
    histogram(white_noise, 'Normalization', 'probability');
    title('高斯分布的白噪声直方图');
    xlabel('值');
    ylabel('概率');

    hold on;
    x=-10:0.1:10;
    y=gaussmf(x,[sigma(i),mu(i)]);
    plot(x,y/8,LineWidth=2);
    legend('生成的随机数', '目标概率密度函数');

    [r,lags]=xcorr(white_noise, 'coeff'); % 使用 'coeff' 标准化自相关函数
    subplot(4,1,3);plot(lags,r);grid on;
    title('自相关函数');
    xlabel('样本延迟');
    ylabel('自相关系数');
    str1=['均值=',num2str(mean(white_noise))];
    str2=['方差=',num2str(std(white_noise)^2)];
    text(-600,0.5,str1)
    text(200,0.5,str2)
    
    subplot(4,1,4); 
    f=fftshift(fft(r));%频谱校正
    x=(0:length(f)-1)*200/length(f)-100; %x轴
    y=abs(f);
    plot(x,y,'g');grid on;
    title('功率谱密度');
    xlabel('角频率');
    ylabel('功率谱密度 ');
    

end
%%

%%
% 生成均匀分布的白噪声
clear all;
rng(42);
lower_bound = [-1,-2,-0.5];   % 均匀分布的下界
upper_bound = [1,2,0.5];    % 均匀分布的上界
num_samples = 1000; % 样本数量
for i = 1:3
    figure(i);
    uniform_noise = lower_bound(i) + (upper_bound(i) - lower_bound(i)) * rand(1, num_samples);
    % 绘制噪声图
    subplot(4,1,1)
    plot(uniform_noise);
    title('均匀分布的白噪声图');
    xlabel('样本');
    ylabel('值');
    % 绘制直方图
    subplot(4,1,2)
    histogram(uniform_noise, 'Normalization', 'probability');
    title('均匀分布的白噪声直方图');
    xlabel('值');
    ylabel('概率');
    legend('生成的随机数');
    
    [r,lags]=xcorr(uniform_noise, 'coeff'); % 使用 'coeff' 标准化自相关函数
    subplot(4,1,3);plot(lags,r);grid on;
    title('自相关函数');
    xlabel('样本延迟');
    ylabel('自相关系数');
    str1=['均值=',num2str(mean(uniform_noise))];
    str2=['方差=',num2str(std(uniform_noise)^2)];
    text(-600,0.5,str1)
    text(200,0.5,str2)
    
    subplot(4,1,4);
    f=fftshift(fft(r));%频谱校正
    x=(0:length(f)-1)*200/length(f)-100; %x轴
    y=abs(f);
    plot(x,y,'g');grid on;
    title('功率谱密度');
    xlabel('角频率');
    ylabel('功率谱密度 ');

end

%%
%任意分布白噪声y=|x|，【-1，1】
% 设定生成随机数的数量
clear all;
num_samples = 1000;% 初始化数组来存储生成的随机数
generated_samples = zeros(1, num_samples);% 设定目标概率密度函数 y = 2x
target_pdf = @(x) abs(x);
% 设定上界（最大可能值）
upper_bound = 1;
low_bound = -1;
% 生成随机数
count = 1;
while count <= num_samples
    % 生成候选随机数
    x_candidate = 2*rand()-1;
    % 生成接受/拒绝标志
    acceptance_prob = target_pdf(x_candidate) / (upper_bound);
    % 接受随机数
    if rand() < acceptance_prob
        generated_samples(count) = x_candidate;
        count = count + 1;
    end
end
subplot(4,1,1)
plot(generated_samples);
title('任意分布的白噪声图');
xlabel('样本'); 
ylabel('值');
subplot(4,1,2)
% 绘制生成的随机数的直方图
histogram(generated_samples, 'Normalization', 'pdf');
hold on;
% 绘制目标概率密度函数
x_values = linspace(-1, 1, 100);
y_values = target_pdf(x_values) / integral(target_pdf, -1, 1);
plot(x_values, y_values, 'LineWidth', 2);
xlabel('随机数');
ylabel('概率密度');
title('使用舍选抽样法生成随机数');
legend('生成的随机数', '目标概率密度函数');
hold off;
[r,lags]=xcorr(generated_samples, 'coeff'); % 使用 'coeff' 标准化自相关函数
subplot(4,1,3);plot(lags,r);grid on;
title('自相关函数');
xlabel('样本延迟');
ylabel('自相关系数');
str1=['均值=',num2str(mean(generated_samples))];
str2=['方差=',num2str(std(generated_samples)^2)];
text(-600,0.5,str1)
text(200,0.5,str2)
subplot(4,1,4);
f=fftshift(fft(r));%频谱校正
x=(0:length(f)-1)*200/length(f)-100; %x轴
y=abs(f);
plot(x,y,'g');grid on;
title('功率谱密度');
xlabel('角频率');
ylabel('功率谱密度');


%%
clear all;
%随相正弦信号
rng(42);
lower_bound = [0,pi,0];   % 均匀分布的下界
upper_bound = [pi,2*pi,2*pi];    % 均匀分布的上界
% 设置参数
fs = 1000; % 采样频率
t = 0:1/fs:1; % 时间向量，从0到1秒，以1/fs的步长增加

% 生成相位在0到2π均匀分布的正弦波
amplitude = 1; % 振幅
for i = 1:3
    % 生成正弦波
    phase = lower_bound(i) + (upper_bound(i)- lower_bound(i)) * rand(1, length(t));
    sinewave = amplitude * sin(2*10*pi*t+phase);
    figure(i);
    % 绘制波形图
    subplot(3,1,1);
    plot(t, sinewave);
    xlabel('时间 (秒)');
    ylabel('振幅');
    title('相位在0到2π均匀分布的正弦波');
    grid on;
    
    [r,lags]=xcorr(sinewave, 'coeff'); % 使用 'coeff' 标准化自相关函数
    subplot(3,1,2);
    plot(lags,r);grid on;
    title('自相关函数');
    xlabel('样本延迟');
    ylabel('自相关系数');
    str1=['均值=',num2str(mean(sinewave))];
    str2=['方差=',num2str(std(sinewave)^2)];
    text(-600,0.5,str1)
    text(200,0.5,str2)
    
    subplot(3,1,3); 
    f=fftshift(fft(r));%频谱校正
    x=(0:length(f)-1)*200/length(f)-100; %x轴
    y=abs(f);
    plot(x,y,'g');grid on;
    title('功率谱密度');
   
    xlabel('角频率');
    ylabel('功率谱密度 ');
end

%%
clear all;
%带有白噪声的多个正弦信号
% 设置参数
fs = 1000; % 采样频率
t = 0:1/fs:1; % 时间向量，从0到1秒，以1/fs的步长增加
% 生成白噪声
white_noise = randn(size(t));
% 设置正弦信号的参数
frequencies = [50, 150, 300;10, 20, 30;20, 40, 60]; % 正弦信号的频率
amplitudes = [1, 0.5, 0.2;1,1.5, 2;1, 0.5, 0.6]; % 正弦信号的振幅
phases = [pi/4, pi/2, 3*pi/4;pi/4, pi/2, 3*pi/4;pi/4, pi/2, 3*pi/4]; % 正弦信号的相位
for j=1:3
    % 生成多个正弦信号并加入白噪声
    sinewaves = zeros(size(t));
    for i = 1:3
        sinewaves = sinewaves + amplitudes(j,i) * sin(2*pi*frequencies(j,i)*t + phases(j,i));
    end
    % 加入白噪声
    noisy_signal = sinewaves + white_noise;
    % 绘制结果
    figure(j);
    subplot(3,1,1);
    plot(t, noisy_signal);
    title('带有白噪声的多个正弦信号');
    xlabel('时间 (秒)');
    [r,lags]=xcorr(noisy_signal, 'coeff'); % 使用 'coeff' 标准化自相关函数
    subplot(3,1,2);
    plot(lags,r);grid on;
    title('自相关函数');
    xlabel('样本延迟');
    ylabel('自相关系数');
    str1=['均值=',num2str(mean(noisy_signal))];
    str2=['方差=',num2str(std(noisy_signal)^2)];
    text(-600,0.5,str1)
    text(200,0.5,str2)
    subplot(3,1,3); 
    f=fftshift(fft(r));%频谱校正
    x=(0:length(f)-1)*200/length(f)-100; %x轴
    y=abs(f);
    plot(x,y,'g');grid on;
    title('功率谱密度');
    xlabel('角频率');
    ylabel('功率谱密度');

end


%%
clear all;
%%二元随机过程
num_samples = 1000; % 信号的样本数
% 生成跳变时间
jump_times = sort(randi([1, num_samples], 1, round(num_samples/10))); % 10% 的样本是跳变点

% 生成-1/1二元随机信号
random_signal = zeros(1, num_samples);
current_amplitude = 2 * randi([0, 1]) - 1;

for i = 1:length(jump_times)
    jump_time = jump_times(i);
    random_signal(jump_time:end) = current_amplitude;
    current_amplitude = 2 * randi([0, 1]) - 1;
end
% 绘制结果
figure(1);
subplot(3,1,1);
stairs(random_signal);
title('离散平稳零均值二元随机信号');
xlabel('样本数');
ylabel('信号值');
ylim([-1.5, 1.5]);
grid on;
[r,lags]=xcorr(random_signal, 'coeff'); % 使用 'coeff' 标准化自相关函数
subplot(3,1,2);
plot(lags,r);grid on;
title('自相关函数');
xlabel('样本延迟');
ylabel('自相关系数');
str1=['均值=',num2str(mean(random_signal))];
str2=['方差=',num2str(std(random_signal)^2)];
text(-600,0.5,str1)
text(200,0.5,str2)

subplot(3,1,3); 
f=fftshift(fft(r));%频谱校正
x=(0:length(f)-1)*200/length(f)-100; %x轴
y=abs(f);
plot(x,y,'g');grid on;
title('功率谱密度');
xlabel('角频率');
ylabel('功率谱密度 ');


% 生成0/1二元随机信号
random_signal = zeros(1, num_samples);
current_amplitude =  randi([0, 1]) ;

for i = 1:length(jump_times)
    jump_time = jump_times(i);
    random_signal(jump_time:end) = current_amplitude;
    current_amplitude = randi([0, 1]);
end
% 绘制结果
figure(2);
subplot(3,1,1);
stairs(random_signal);
title('二元随机信号');
xlabel('样本数');
ylabel('信号值');
ylim([0, 1.5]);
grid on;
[r,lags]=xcorr(random_signal, 'coeff'); % 使用 'coeff' 标准化自相关函数
subplot(3,1,2);
plot(lags,r);grid on;
title('自相关函数');
xlabel('样本延迟');
ylabel('自相关系数');
str1=['均值=',num2str(mean(random_signal))];
str2=['方差=',num2str(std(random_signal)^2)];
text(-600,0.5,str1)
text(200,0.5,str2)
subplot(3,1,3);
f=fftshift(fft(r));%频谱校正
x=(0:length(f)-1)*200/length(f)-100; %x轴
y=abs(f);
plot(x,y,'g');grid on;
title('功率谱密度');
xlabel('角频率');
ylabel('功率谱密度 ');


% 生成0/-1二元随机信号
random_signal = zeros(1, num_samples);
current_amplitude =  randi([-1, 0]) ;

for i = 1:length(jump_times)
    jump_time = jump_times(i);
    random_signal(jump_time:end) = current_amplitude;
    current_amplitude = randi([-1, 0]);
end
% 绘制结果
figure(3);
subplot(3,1,1);
stairs(random_signal);
title('二元随机信号');
xlabel('样本数');
ylabel('信号值');
ylim([-1.5, 0]);
grid on;
[r,lags]=xcorr(random_signal, 'coeff'); % 使用 'coeff' 标准化自相关函数
subplot(3,1,2);
plot(lags,r);grid on;
title('自相关函数');
xlabel('样本延迟');
ylabel('自相关系数');
str1=['均值=',num2str(mean(random_signal))];
str2=['方差=',num2str(std(random_signal)^2)];
text(-600,0.5,str1)
text(200,0.5,str2)
subplot(3,1,3);
f=fftshift(fft(r));%频谱校正
x=(0:length(f)-1)*200/length(f)-100; %x轴
y=abs(f);
plot(x,y,'g');grid on;
title('功率谱密度');
xlabel('角频率');
ylabel('功率谱密度 ');

%%
clear all;
fs = 44100; % 采样率，以赫兹为单位
duration = 5; % 采样时长，以秒为单位
% 创建音频记录器对象
recorder = audiorecorder(fs, 16, 1); % 16位量化，单声道
% 开始录音
disp('开始录音...');
record(recorder, duration);
pause(duration);
% 停止录音
stop(recorder);
disp('录音完成');
% 获取录音数据
audio_data = getaudiodata(recorder);
% 绘制波形图
t = (0:length(audio_data)-1) / fs;
figure;
subplot(3,1,1);
plot(t, audio_data);
xlabel('时间 (秒)');
ylabel('幅度');
title('录音波形图');
[r,lags]=xcorr(audio_data, 'coeff'); % 使用 'coeff' 标准化自相关函数
subplot(3,1,2);
plot(lags,r);grid on;
title('自相关函数');
xlabel('样本延迟');
ylabel('自相关系数');
str1=['均值=',num2str(mean(audio_data))];
str2=['方差=',num2str(std(audio_data)^2)];
text(-200000,0.5,str1)
text(20000,0.5,str2)
subplot(3,1,3);
f=fftshift(fft(r));%频谱校正
x=(0:length(f)-1)*200/length(f)-100; %x轴
y=abs(f);
plot(x,y,'g');grid on;
title('功率谱密度');
xlabel('角频率');
ylabel('功率谱密度 ');
% 播放录音
soundsc(audio_data, fs);


%%