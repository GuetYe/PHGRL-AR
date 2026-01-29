clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

% 数据数量
data_num = 500

% 上层奖励
data = readmatrix('../plot_data/High_eva_value_data_0.0015_0.005_0.0005_0.005_0.00014_train_main_thread_690.csv');

x = data(1:data_num,1);
y = data(1:data_num,2);
% xlim([0,460])
mov_y = moving_average(y,100);

figure()
plot(x,y)
hold on
plot(x,mov_y)
hold off


% 下层奖励
data = readmatrix('../plot_data/Low_eva_value_data_0.0015_0.005_0.0005_0.005_0.00014_train_main_thread_690.csv')

xL = data(1:data_num,5);
yL1 = data(1:data_num,1);
yL2 = data(1:data_num,2);
yL3 = data(1:data_num,3);
yL4 = data(1:data_num,4);

% % xlim([0,460])
mov_yL1 = moving_average(yL1,100);
mov_yL2 = moving_average(yL2,100);
mov_yL3 = moving_average(yL3,100);
mov_yL4 = moving_average(yL4,100);
% 
figure()
hold on
plot(xL,yL1)
%plot(xL,mov_yL1)
plot(xL,yL2)
%plot(xL,mov_yL2)
plot(xL,yL3)
%plot(xL,mov_yL3)
plot(xL,yL4)
%plot(xL,mov_yL4)
hold off

