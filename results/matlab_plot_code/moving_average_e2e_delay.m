clc,clear
clear all;
crt = pwd;
addpath(genpath(crt));    % 添加当前的文件夹

opts = detectImportOptions('../plot_data/SAC_8000_eva_value_data_2025-08-13-20-54-36.csv');
data = readtable('../plot_data/SAC_8000_eva_value_data_2025-08-13-20-54-36.csv',opts);


disp(data)
choice_num = 5000

x = data.eva_count_list(1:choice_num);
y = data.eva_value_list(1:choice_num);

% 提取第3列分类
category = cellfun(@(c) c(1),data.control_tag_list(1:choice_num),'UniformOutput',false)

% 提取第4列标签
shape_label = cellfun(@(c) c(13:14),data.servers_list(1:choice_num),'UniformOutput',false)

% 找出所有类别和形状的唯一值
unique_cat = unique(category)
unique_shape = unique(shape_label)

% 定义颜色映射（可根据类别数量调整）
colors = lines(length(unique_cat)); % 颜色数组
% 定义形状映射符号（可根据形状类别数量调整）
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};

figure; hold on;

for i = 1:length(unique_cat)
    for j = 1:length(unique_shape)
        % 找出对应类别和形状的索引
        idx = strcmp(category, unique_cat{i}) & strcmp(shape_label, unique_shape{j});
        if any(idx)
            scatter(x(idx), y(idx), 50, 'Marker', markers{mod(j-1,length(markers))+1}, ...
                'MarkerEdgeColor', colors(i,:), 'MarkerFaceColor', colors(i,:));
        end
    end
end
xlabel('Index (第1列)');
ylabel('Value (第2列)');
title('基于类别颜色与形状的散点图');
legend_entries = {};
for i=1:length(unique_cat)
    for j=1:length(unique_shape)
        legend_entries{end+1} = sprintf('%s - %s', unique_cat{i}, unique_shape{j});
    end
end
legend(legend_entries, 'Location', 'bestoutside');
plot(x,y)
mov_y = moving_average(y,100)
plot(x,mov_y)

grid on;
hold off;
