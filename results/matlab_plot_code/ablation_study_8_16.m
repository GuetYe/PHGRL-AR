clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

opts = detectImportOptions('../plot_data/SAC_8000_eva_value_data_2025-08-13-20-54-36.csv');
data = readtable('../plot_data/SAC_8000_eva_value_data_2025-08-13-20-54-36.csv',opts);

% 取出第三列所有标签分类
control_tag = cellfun(@(c) c(1), data.control_tag_list,'UniformOutput',false);

% 找到"C"的行索引
idx_C = strcmp(control_tag,'C');

% 提取相应的数据行
sample_number = 1500;
window_size = 50;
start_index = 500;
data_index = (start_index+1):(start_index+sample_number);


% 模型控制的数据
xc = 1:sample_number;
ycf = data.eva_value_list(idx_C);
yc = ycf(data_index+1100);

mov_yc = moving_average(yc,window_size);


% 随机-路由
data1 = readmatrix('../plot_data/SAC_6000_eva_value_data_2025-08-15-09-39-09_random_routing.csv');
x1 = data1(end-sample_number+1:end,1);
y1 = data1(end-sample_number+1:end,2);
mov_y1 = moving_average(y1,window_size);



% 决策-最短
data2 = readmatrix('../plot_data/SAC_5400_eva_value_data_2025-08-15-11-50-53_control_sp.csv');
x2 = data2(end-sample_number+1:end,1);
y2 = data2(end-sample_number+1:end,2);
mov_y2 = moving_average(y2,window_size);

% 随机-最短
data3 = readmatrix('../plot_data/SAC_4200_eva_value_data_2025-08-14-10-14-18_random_sp.csv');
x3 = data3(end-sample_number+1:end,1);
y3 = data3(end-sample_number+1:end,2);
mov_y3 = moving_average(y3,window_size);

rate = 95;
figure
hold on 
plot_smoothed_band(xc, yc, 'Window', window_size, 'Band', 0.9, 'ShowRaw', false, 'YLim', [0 max(yc)+1]);
mean(yc)
std(yc)
plot_smoothed_band(xc, y1, 'Window', window_size, 'Band', 0.9, 'ShowRaw', false, 'YLim', [0 max(yc)+1]);
mean(y1)
std(y2)
plot_smoothed_band(xc, y2, 'Window', window_size, 'Band', 0.9, 'ShowRaw', false, 'YLim', [0 max(yc)+1]);
mean(y2)
std(y2)
plot_smoothed_band(xc, y3, 'Window', window_size, 'Band', 0.9, 'ShowRaw', false, 'YLim', [0 max(yc)+1]);
mean(y3)
std(y3)
hold off


