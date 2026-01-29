clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

% opts = detectImportOptions('../plot_data/DDPG_2000_eva_value_data_2025-06-19-20-48-37_control.csv');
% data = readtable('../plot_data/DDPG_2000_eva_value_data_2025-06-19-20-48-37_control.csv',opts);


% disp(data)
% 
% x = data.eva_count_list;
% y = data.eva_value_list;

data = readmatrix('../plot_data/DDPG_2000_eva_value_data_2025-06-21-20-58-20_3s_control.csv')
x = data(:,1)
y = data(:,2)

data1 = readmatrix('../plot_data/DDPG_2000_eva_value_data_2025-06-21-23-05-23_3s_random.csv')
x1 = data1(:,1)
y1 = data1(:,2)


data2 = readmatrix('../plot_data/DDPG_2000_eva_value_data_2025-06-22-15-02-38_3s_15.csv')
x2 = data2(:,1)
y2 = data2(:,2)


% data3 = readmatrix('../plot_data/DDPG_2000_eva_value_data_2025-06-20-15-17-09_comparison_16.csv')
% x3 = data3(:,1)
% y3 = data3(:,2)


plot(x,y)

hold on 
plot(x1,y1)
plot(x2,y2)
% plot(x3,y3)

xline(28)
xline(712)
ax = gca
% ax.XTick = sort([ax.XTick 28 712])
% ax.XTickLabel{ax.XTick==28} = '📍28'
% ax.XTickLabel{ax.XTick==712} = '📍712'



hold off
