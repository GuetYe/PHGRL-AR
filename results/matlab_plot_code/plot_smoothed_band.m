function plot_smoothed_band(x, y, varargin)
% plot_smoothed_band(x, y, ...) 
% 对单列数据y沿x进行滑动窗口平滑并画置信带，支持负值数据
%
% 参数（同前）：
%   'Window'     滑动窗口宽度（默认11）
%   'Band'       置信带宽度系数（默认1）
%   'ShowRaw'    是否显示原始数据（默认true）
%   'Color'      线条颜色（默认[0 0.4470 0.7410]）
%   'YLim'       y轴范围（如[-2 2]，默认自动）

% 解析参数
p = inputParser;
addParameter(p, 'Window', 11);
addParameter(p, 'Band', 1);
addParameter(p, 'ShowRaw', true);
addParameter(p, 'Color', [0 0.4470 0.7410]);
addParameter(p, 'YLim', []);  % 默认自动
parse(p, varargin{:});
w = p.Results.Window;
b = p.Results.Band;
show_raw = p.Results.ShowRaw;
c = p.Results.Color;
yl = p.Results.YLim;

outputStr = sprintf('均值: %.2f, 方差: %.2f', mean(y), std(y));
disp(outputStr);

x = x(:)'; y = y(:)'; % 保证行向量

mean_y = movmean(y, w, 'Endpoints','shrink');
std_y  = movstd(y, w, 0, 'Endpoints','shrink');

upper = mean_y + b*std_y;
lower = mean_y - b*std_y;

% figure; hold on;

% 为避免颜色越界，确保阴影颜色在[0,1]
fill_color = min(max(c+0.5*(1-c),0),1);

h1 = fill([x fliplr(x)], [lower fliplr(upper)], fill_color, ...
    'EdgeColor','none', 'FaceAlpha',0.4, 'HandleVisibility','off');
h2 = plot(x, mean_y, '-', 'LineWidth',2, 'Color',c, 'DisplayName','滑动均值');
if show_raw
    h3 = plot(x, y, ':', 'LineWidth',1, 'Color',[0.4 0.4 0.4], 'DisplayName','原始数据');
end

xlabel('x'); ylabel('y');
title('滑动窗口平滑及置信带');
legend('show');

% 自动或手动设定y轴范围
if isempty(yl)
    margin = 0.05 * (max([upper y])-min([lower y]));
    ylim([min([lower y])-margin, max([upper y])+margin]);
else
    ylim(yl);
end