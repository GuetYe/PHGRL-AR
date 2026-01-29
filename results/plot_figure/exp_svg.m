% exp_svg.m  (R2022 适用)
inDir  = pwd;
outDir = fullfile(inDir, 'SVG_out');
if ~exist(outDir, 'dir'); mkdir(outDir); end

files = dir(fullfile(inDir, '*.fig'));
assert(~isempty(files), '没找到 *.fig 文件');

for k = 1:numel(files)
    figPath = fullfile(files(k).folder, files(k).name);

    % 不显示打开
    f = openfig(figPath, 'invisible');
    set(f, 'Color', 'w');

    % 强制使用矢量渲染器（SVG 一般用 painters 更稳定）
    set(f, 'Renderer', 'painters');

    % （可选）A4 横向的 paper 设置：对 print 有用
    set(f, 'PaperUnits', 'centimeters');
    set(f, 'PaperSize', [29.7 21.0]);              % A4 横向
    set(f, 'PaperPosition', [0 0 29.7 21.0]);
    set(f, 'PaperOrientation', 'landscape');

    % 强制使用矢量渲染器（SVG 一般用 painters 更稳定）
    set(f, 'Renderer', 'painters');

    % 取消所有文字加粗 + 固定字体（推荐）
    set(findall(f, '-property', 'FontWeight'), 'FontWeight', 'normal');
    set(findall(f, '-property', 'FontName'), 'FontName', 'Times New Roman'); % 或 Arial
    set(findall(f, '-property', 'Interpreter'), 'Interpreter', 'none');      % 可选

    drawnow;

    % 输出路径：print 建议用“不带扩展名”的文件名
    outBase = fullfile(outDir, erase(files(k).name, '.fig'));

    % 导出 SVG
    print(f, outBase, '-dsvg', '-painters');   % 会生成 outBase.svg

    close(f);
    fprintf('[%d/%d] Saved: %s.svg\n', k, numel(files), outBase);
end
