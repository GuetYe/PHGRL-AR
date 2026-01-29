% export_ablation_pdf.m
inDir  = pwd;                         % 输入目录：当前目录
outDir = fullfile(inDir, 'pdf_out');  % 输出目录
if ~exist(outDir, 'dir'); mkdir(outDir); end

files = dir(fullfile(inDir, '*.fig'));
assert(~isempty(files), '没找到 ablation_study*.fig 文件');

for k = 1:numel(files)
    figPath = fullfile(files(k).folder, files(k).name);

    % 不显示打开 .fig
    f = openfig(figPath, 'invisible');
    set(f, 'Color', 'w');

    % ===== 1) 取消文字加粗 + 固定字体（强烈建议）=====
    set(findall(f, '-property', 'FontWeight'), 'FontWeight', 'normal');
    set(findall(f, '-property', 'FontName'), 'FontName', 'Times New Roman'); % 或 Arial
    % 可选：避免 LaTeX interpreter 造成字体替换看起来更粗
    % set(findall(f, '-property', 'Interpreter'), 'Interpreter', 'none');
    
    % ------- A4 横向设置（关键）-------
    set(f, 'PaperUnits', 'centimeters');
    set(f, 'PaperSize', [29.7 21.0]);          % A4 横向：宽29.7 高21.0
    set(f, 'PaperOrientation', 'landscape');


    % 让画布尺寸也接近 A4 横向（导出更铺满）
    set(f, 'Units', 'centimeters');
    set(f, 'Position', [1 1 29.7 21.0]);       % 可按需调边距
    drawnow;

    % 输出 pdf（矢量）
    pdfName = replace(files(k).name, '.fig', '.pdf');
    outPath = fullfile(outDir, pdfName);
    exportgraphics(f, outPath, 'ContentType', 'vector', 'BackgroundColor', 'white');

    close(f);
    fprintf('[%d/%d] Saved: %s\n', k, numel(files), outPath);
end
