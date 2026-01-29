clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹
window_size = 100
sample_number = 2000

% 随机目的的数据
data1 = readmatrix('../plot_data/DDPG_5100_eva_value_data_2025-07-10-15-47-07_SP_15_4000.csv')
x1 = data1(1:sample_number,1)
y1 = data1(1:sample_number,2)
mov_y1 = moving_average(y1,window_size)



% 最短最短路径
data2 = readmatrix('../plot_data/DDPG_4850_eva_value_data_2025-07-10-19-08-51_random_4000.csv')
x2 = data2(1:sample_number,1)
y2 = data2(1:sample_number,2)
mov_y2 = moving_average(y2,window_size)

% % 最短最短路径
% data3 = readmatrix('../plot_data/DDPG_2600_eva_value_data_2025-07-09-16-01-45_SP_17.csv')
% x3 = data3(1:sample_number,1)
% y3 = data3(1:sample_number,2)
% mov_y3 = moving_average(y3,window_size)
% 
% % 最短最短路径
% data4 = readmatrix('../plot_data/DDPG_8650_eva_value_data_2025-07-09-17-18-03_SP_18.csv')
% x4 = data4(1:sample_number,1)
% y4 = data4(1:sample_number,2)
% mov_y4 = moving_average(y4,window_size)


figure
hold on 
plot(x1,y1)
plot(x1,mov_y1)
plot(x2,y2)
plot(x2,mov_y2)
% plot(x3,y3)
% plot(x3,mov_y3)
% plot(x4,y4)
% plot(x4,mov_y4)
hold off

%% 结论: 4000 的情况下固定流表随机目标和固定流表选15的情况近似
