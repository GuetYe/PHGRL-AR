clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹
window_size = 100
sample_number = 2000

% 目标节点15
data1 = readmatrix('../plot_data/DDPG_2650_eva_value_data_2025-07-09-09-38-20_SP_15.csv')
x1 = data1(1:sample_number,1)
y1 = data1(1:sample_number,2)
mov_y1 = moving_average(y1,window_size)


% 目标节点16
data2 = readmatrix('../plot_data/DDPG_8200_eva_value_data_2025-07-09-10-56-37_SP_16.csv')
x2 = data2(1:sample_number,1)
y2 = data2(1:sample_number,2)
mov_y2 = moving_average(y2,window_size)


% 目标节点17
data3 = readmatrix('../plot_data/DDPG_2600_eva_value_data_2025-07-09-16-01-45_SP_17.csv')
x3 = data3(1:sample_number,1)
y3 = data3(1:sample_number,2)
mov_y3 = moving_average(y3,window_size)


% 目标节点18
data4 = readmatrix('../plot_data/DDPG_8650_eva_value_data_2025-07-09-17-18-03_SP_18.csv')
x4 = data4(1:sample_number,1)
y4 = data4(1:sample_number,2)
mov_y4 = moving_average(y4,window_size)

% 随机目标
data5 = readmatrix('../plot_data/DDPG_5650_eva_value_data_2025-07-07-09-27-29_random_7_7.csv')
x5 = data5(1:sample_number,1)
y5 = data5(1:sample_number,2)
mov_y5 = moving_average(y5,window_size)

figure
hold on 
plot(x1,y1)
plot(x1,mov_y1)
plot(x2,y2)
plot(x2,mov_y2)
plot(x3,y3)
plot(x3,mov_y3)
plot(x4,y4)
plot(x4,mov_y4)
plot(x5,y5)
plot(x5,mov_y5)
hold off


