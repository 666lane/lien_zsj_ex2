
%%
clear all;
% 生成高斯分布的白噪声
rng(42);
mu = [0, 0, 0]; % 均值
sigma = [1, 0.5, 2]; % 标准差
num_samples = 1000; % 样本数量

for i = 1:3
    figure(i);
    white_noise = mu(i) + sigma(i) * randn(1, num_samples);
    subplot(1,2,1)
    plot(white_noise);
    title('高斯分布的白噪声图');
    xlabel('样本');
    ylabel('值');
    axis square;
    % 绘制直方图
    subplot(1,2,2)
    histogram(white_noise, 'Normalization', 'probability');
    title('高斯分布的白噪声直方图');
    xlabel('值');
    ylabel('概率');
    axis square;
    hold on;
    x=-10:0.1:10;
    y=gaussmf(x,[sigma(i),mu(i)]);
    plot(x,y/8,LineWidth=2);
    str1=['均值=',num2str(mu(i))];
    str2=['方差=',num2str(i)];
    text(-4,0.1,str1)
    text(-4,0.08,str2)
    legend('生成的随机数', '目标概率密度函数');
end

%%
clear all;
% 生成均匀分布的白噪声
rng(42);
lower_bound = [-1,-2,-0.5];   % 均匀分布的下界
upper_bound = [1,2,0.5];    % 均匀分布的上界
num_samples = 1000; % 样本数量
for i = 1:3
    figure(i);
    uniform_noise = lower_bound(i) + (upper_bound(i) - lower_bound(i)) * rand(1, num_samples);
    % 绘制噪声图
    subplot(1,2,1)
    plot(uniform_noise);
    title('均匀分布的白噪声图');
    xlabel('样本');
    ylabel('值');
    axis square;
    % 绘制直方图
    subplot(1,2,2)
    histogram(uniform_noise, 'Normalization', 'probability');
    title('均匀分布的白噪声直方图');
    xlabel('值');
    ylabel('概率');
    axis square;
    legend('生成的随机数');
end

%%
clear all;
%任意分布分布
% 设定生成随机数的数量
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
subplot(1,2,1)
plot(generated_samples);
title('任意分布的白噪声图');
xlabel('样本'); 
ylabel('值');
axis square;
subplot(1,2,2)
% 绘制生成的随机数的直方图
histogram(generated_samples, 'Normalization', 'pdf');
hold on;
% 绘制目标概率密度函数
x_values = linspace(-1, 1, 100);
y_values = target_pdf(x_values) / integral(target_pdf, -1, 1);
plot(x_values, y_values, 'LineWidth', 2);
xlabel('随机数');
ylabel('概率密度');
title('使用舍选抽样法生成y=|x|随机数');
legend('生成的随机数', '目标概率密度函数');
axis square;
hold off;





%%
clear all;
%随相正弦信号
rng(42);
lower_bound = [0,0,pi];   % 均匀分布的下界
upper_bound = [2*pi,pi,2*pi];    % 均匀分布的上界
% 设置参数
fs = 1000; % 采样频率
t = 0:1/fs:1; % 时间向量，从0到1秒，以1/fs的步长增加

% 生成相位在0到2π均匀分布的正弦波
amplitude = 1; % 振幅
for i = 1:3
    % 生成正弦波
    phase = lower_bound(i) + (upper_bound(i)- lower_bound(i)) * rand(1, length(t));
    sinewave = amplitude * sin(2*pi*10*t+phase);
    figure(i);
    % 绘制波形图

    plot(t, sinewave);
    xlabel('时间 (秒)');
    ylabel('振幅');
    title('相位在0到2π均匀分布的正弦波');
    grid on;
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
    plot(t, sinewaves);
    title('多个正弦信号');
    subplot(3,1,2);
    plot(t, white_noise);
    title('白噪声');
    subplot(3,1,3);
    plot(t, noisy_signal);
    title('带有白噪声的多个正弦信号');
    xlabel('时间 (秒)');

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
stairs(random_signal);
title('离散平稳零均值二元随机信号');
xlabel('样本数');
ylabel('信号值');
ylim([-1.5, 1.5]);
grid on;


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
stairs(random_signal);
title('二元随机信号');
xlabel('样本数');
ylabel('信号值');
ylim([0, 1.5]);
grid on;


% 生成0/1二元随机信号
random_signal = zeros(1, num_samples);
current_amplitude =  randi([-1, 0]) ;

for i = 1:length(jump_times)
    jump_time = jump_times(i);
    random_signal(jump_time:end) = current_amplitude;
    current_amplitude = randi([-1, 0]);
end
% 绘制结果
figure(3);
stairs(random_signal);
title('二元随机信号');
xlabel('样本数');
ylabel('信号值');
ylim([-1.5, 0]);
grid on;

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
plot(t, audio_data);
xlabel('时间 (秒)');
ylabel('幅度');
title('录音波形图');
% 播放录音
soundsc(audio_data, fs);


%%