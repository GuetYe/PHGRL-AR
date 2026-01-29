clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

opts = detectImportOptions('../plot_data/DDPG_2000_eva_value_data_2025-06-19-20-48-37_control.csv');
data = readtable('../plot_data/DDPG_2000_eva_value_data_2025-06-19-20-48-37_control.csv',opts);


disp(data)

x = data.eva_count_list;
y = data.eva_value_list;

plot(x,y)

hold on 
xline(28)
xline(712)
ax = gca
ax.XTick = sort([ax.XTick 28 712])
ax.XTickLabel{ax.XTick==28} = '📍28'
ax.XTickLabel{ax.XTick==712} = '📍712'



hold off
