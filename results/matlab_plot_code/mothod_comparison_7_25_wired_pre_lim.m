clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

opts = detectImportOptions('../plot_data/DDPG_9950_eva_value_data_2025-07-20-15-50-25_test_wired_dp_lim_pre_control.csv');
data = readtable('../plot_data/DDPG_9950_eva_value_data_2025-07-20-15-50-25_test_wired_dp_lim_pre_control.csv',opts);

% 取出第三列所有标签分类
control_tag = cellfun(@(c) c(1), data.control_tag_list,'UniformOutput',false);

% 找到"C"的行索引
idx_C = strcmp(control_tag,'T');

% 提取相应的数据行
sample_number = 1000;
window_size = 100;

% 模型控制的数据
xc = 1:sample_number;
ycf = data.eva_value_list(idx_C);
% yc = ycf(1:sample_number);
yc = ycf((length(ycf)-sample_number+1):length(ycf));

mov_yc = moving_average(yc,window_size)

% 随机目的的数据
data1 = readmatrix('../plot_data/DDPG_7950_eva_value_data_2025-07-14-09-52-47_7_14_random.csv')
xr = data1(1:sample_number,1)
yr = data1(1:sample_number,2)
mov_yr = moving_average(yr,window_size)



% 最短最短路径
data2 = readmatrix('../plot_data/DDPG_7800_eva_value_data_2025-07-15-22-42-18_test_wired_dp_7_15_5.csv')
xs = data2(1:sample_number,1)
ys = data2(1:sample_number,2)
mov_ys = moving_average(ys,window_size)

% 最短最短路径
data3 = readmatrix('../plot_data/DDPG_9900_eva_value_data_2025-07-16-09-50-06_test_wired_dp_7_16_6.csv')
xs1 = data3(1:sample_number,1)
ys1 = data3(1:sample_number,2)
mov_ys1 = moving_average(ys1,window_size)

figure
hold on 
plot(xc,yc)
plot(xc,mov_yc)
plot(xr,yr)
plot(xr,mov_yr)
plot(xs,ys)
plot(xs,mov_ys)
plot(xs1,ys1)
plot(xs1,mov_ys1)
hold off


