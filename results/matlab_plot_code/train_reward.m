clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

data = readmatrix('../plot_data/High_eva_value_data_train_main_thread_460_7_2.csv');

x = data(:,1);
y = data(:,2);
xlim([0,460])

plot(x,y)