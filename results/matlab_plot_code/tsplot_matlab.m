function tsplot_matlab(x, y_data, group, varargin)
% x: x坐标 (T*1)
% y_data: (T*N) 数据，每列一条曲线
% group: 1*N 向量，表示每列y_data所属组（如1,2,3...）
% varargin: 可选参数设置
% 支持多组自动配色，画均值及置信带

if nargin<3
    group = ones(1, size(y_data,2));
end
groups = unique(group);
cols = lines(numel(groups)); % 自动配色
hold on;
for i = 1:numel(groups)
    idx = group==groups(i);
    dat = y_data(:,idx);   % 这一组所有曲线
    mean_y = nanmean(dat,2);
    std_y = nanstd(dat,0,2);
    n = sum(~isnan(dat),2);
    alpha = 0.05;
    tval = tinv(1-alpha/2, n-1);
    ci = tval .* std_y ./ sqrt(n);
    
    % fill置信带，自动透明
    fill([x;flipud(x)], [mean_y-ci; flipud(mean_y+ci)], cols(i,:), ...
         'EdgeColor','none', 'FaceAlpha',0.2);
    plot(x, mean_y, '-', 'LineWidth',2, 'Color', cols(i,:), ...
         'DisplayName', sprintf('Group %d', groups(i)));
    scatter(x,y_data)
end
legend show; grid on;
xlabel('x'); ylabel('y');
title('MATLAB仿tsplot多组示例');
hold off;
end