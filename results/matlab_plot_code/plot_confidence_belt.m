function [lower,upper,medianFit] = plot_confidence_belt(x,y,rate)
% 绘制置信带
% 输入 原始数据
% 绘制 原始数据图像
% 输出 

xc = x(:);       % 保证是列向量
yc = y(:);
nBoot = 500;                              % bootstrap次数
yFits = zeros(length(xc), nBoot);
for i = 1:nBoot
    idx = randi(length(xc), length(xc), 1);      % 有放回采样
    xBoot = xc(idx);
    yBoot = yc(idx);
    fBoot = fit(xBoot, yBoot, 'smoothingspline');
    yFits(:, i) = fBoot(xc);                     % 记录拟合值
end
% 计算置信区间
lower = prctile(yFits, 100-rate, 2);   % 下置信带
upper = prctile(yFits, rate, 2);  % 上置信带
medianFit = median(yFits, 2);
hold on;
fill([xc; flipud(xc)], [lower; flipud(upper)], [0.8 0.85 1], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.4, 'DisplayName', '90%置信带');
% plot(xc, medianFit, 'b-', 'LineWidth', 2, 'DisplayName', '拟合中位数');
% plot(xc, yc, 'ko', 'MarkerFaceColor', 'w', 'DisplayName', '原始数据');

end

