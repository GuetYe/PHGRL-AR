clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

data = [1, 2, 3, 4, 5, 6, 7];
window = 4;
smoothed = moving_average(data, window);
disp(smoothed);