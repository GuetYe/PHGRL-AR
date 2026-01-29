clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

data = readmatrix('../plot_data/High_eva_value_data_train_main_thread_800_7_29_1.csv');

x = data(:,1);
y = data(:,2);
% xlim([0,460])
mov_y = moving_average(y,100);

figure()
plot(x,y)
hold on
plot(x,mov_y)
hold off

