clear all;
Fs=1000;
t=0:1/Fs:1;
for k=1:100
    frequence1=10:1:990;
    frequence2=10:1:990;
    frequence1=frequence1(randperm(length(frequence1)));
    frequence2=frequence2(randperm(length(frequence2)));
    A1=1:0.1:99;
    A2=1:0.1:99;
    A1=A1(randperm(length(A1)));
    A2=A2(randperm(length(A2)));
    savePath_In= 'C:\Users\48828\Desktop\my\experiment2\Dataset\In\';
    savePath_Out='C:\Users\48828\Desktop\my\experiment2\Dataset\Out\';
    for i=1:980
        xn=A1(i)*sin(2*pi*frequence1(i)*t)+A2(i)*sin(2*pi*frequence2(i)*t)+randn(size(t));
    %     figure(i);
    %     subplot(5,1,1);
        Px=abs(fft(xn,1024)).^2/length(xn);
        y1=10*log10(Px);
    %     plot((0:1023)/1024*Fs,10*log10(Px));
    %     title('周期图谱估计法');
    %     xlabel('频率（Hz）');
    %     ylabel('功率谱(dB)');

    %     subplot(5,1,2);
        w=hanning(256)';
        Pxxx=(abs(fft(w.*xn(1:256))).^2+abs(fft(w.*xn(129:384))).^2+abs(fft(w.*xn(257:512))).^2+abs(fft(w.*xn(385:640))).^2+abs(fft(w.*xn(513:768))).^2+abs(fft(w.*xn(641:896))).^2)/(norm(w)^2*6);
    %     plot((0:255)/256*Fs,10*log10(Pxxx));
    %     title('Welch谱估计法');
    %     xlabel('频率（Hz）');
    %     ylabel('功率谱(dB)');

    %     subplot(5,1,3);
        new_points = linspace(1, 256, 1024);
        seq1024 = interp1(10*log10(Pxxx), new_points, 'linear');
        y2=seq1024;
    %     plot((0:1023)/1024*Fs,seq1024);
    %     title('Welch谱估计法（插值后）');


    %     subplot(5,1,4);
        a=9;
        start_index =[floor((frequence1(i)-a)/Fs*1024),floor((frequence2(i)-a)/Fs*1024),floor((1000-frequence2(i)-a)/Fs*1024),floor((1000-frequence1(i)-a)/Fs*1024)];
        end_index = [floor((frequence1(i)+a)/Fs*1024),floor((frequence2(i)+a)/Fs*1024),floor((1000-frequence2(i)+a)/Fs*1024),floor((1000-frequence1(i)+a)/Fs*1024)];

        for j=1:4
            subsequence =y1(start_index(j):end_index(j));
            y2(start_index(j):end_index(j)) = subsequence;
        end
    %     plot((0:1023)/1024*Fs,y2);
    %     title('结合谱估计法');
    %     subplot(5,1,5);
    %     plot(xn);
        fileName = sprintf('%sIn%d.txt', savePath_In, 8000+981*(k-1)+i);
        fid = fopen(fileName, 'w');
        fprintf(fid, '%f\n', xn); % '%f\n' 表示每个数字占一行
        fclose(fid);

        fileName = sprintf('%sOut%d.txt', savePath_Out, 8000+981*(k-1)+i);
        fid = fopen(fileName, 'w');
        fprintf(fid, '%f\n', y2); % '%f\n' 表示每个数字占一行
        fclose(fid);
    end
end