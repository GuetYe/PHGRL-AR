clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

opts = detectImportOptions('../plot_data/DDPG_4550_eva_value_data_2025-07-05-12-09-37_control.csv');
data = readtable('../plot_data/DDPG_4550_eva_value_data_2025-07-05-12-09-37_control.csv',opts);

% 取出第三列所有标签分类
control_tag = cellfun(@(c) c(1), data.control_tag_list,'UniformOutput',false);

% 找到"C"的行索引
idx_C = strcmp(control_tag,'C');

% 提取相应的数据行
sample_number = 2000;
window_size = 100;

% 模型控制的数据
xc = 1:sample_number;
ycf = data.eva_value_list(idx_C);
yc = ycf(1:sample_number);

mov_yc = moving_average(yc,window_size)

% 随机目的的数据
data1 = readmatrix('../plot_data/DDPG_3900_eva_value_data_2025-07-04-15-04-25_random.csv')
xr = data1(1:sample_number,1)
yr = data1(1:sample_number,2)
mov_yr = moving_average(yr,window_size)



% 最短最短路径
data2 = readmatrix('../plot_data/DDPG_2450_eva_value_data_2025-07-05-10-43-25_SSP.csv')
xs = data2(1:sample_number,1)
ys = data2(1:sample_number,2)
mov_ys = moving_average(ys,window_size)


figure
hold on 
plot(xc,yc)
plot(xc,mov_yc)
plot(xr,yr)
plot(xr,mov_yr)
plot(xs,ys)
plot(xs,mov_ys)
hold off


