%% ========================================================================
%  BATCH EXPORT ALL FIGURES AT 300 DPI
%  ========================================================================

clear; clc; close all;

fprintf('\n╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║     BATCH EXPORTING ALL FIGURES AT 300 DPI                    ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

% Create output directory
if ~exist('Figures/HighDPI', 'dir')
    mkdir('Figures/HighDPI');
end

% Find all PNG files in Figures folder
figFiles = dir('Figures/*.png');
fprintf('Found %d PNG files to export...\n\n', length(figFiles));

exportCount = 0;
dpiValue = 300;

for i = 1:length(figFiles)
    inputPath = fullfile(figFiles(i).folder, figFiles(i).name);
    [~, filename, ~] = fileparts(figFiles(i).name);
    outputPath = sprintf('Figures/HighDPI/%s_300dpi.png', filename);
    
    try
        % Read image
        img = imread(inputPath);
        
        % Get image size
        [height, width, ~] = size(img);
        
        % Calculate new size for 300 DPI
        % Assuming original was at ~96 DPI, scale accordingly
        scaleFactor = dpiValue / 96;
        newHeight = round(height * scaleFactor);
        newWidth = round(width * scaleFactor);
        
        % Resize using high-quality interpolation
        imgResized = imresize(img, [newHeight newWidth], 'bicubic');
        
        % Write with imwrite (preserves quality)
        imwrite(imgResized, outputPath);
        
        fprintf('✓ Exported: %s (%d x %d pixels)\n', ...
            filename, newWidth, newHeight);
        exportCount = exportCount + 1;
        
    catch e
        fprintf('✗ Error exporting %s: %s\n', filename, e.message);
    end
end

fprintf('\n');

if exportCount > 0
    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    EXPORT SUCCESSFUL                           ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
    fprintf('Exported: %d figures at 300 DPI\n', exportCount);
    fprintf('Location: Figures/HighDPI/\n\n');
    fprintf('Files ready for IEEE Access / MDPI submission! 📊\n\n');
else
    fprintf('No figures were exported. Check Figures/ folder for PNG files.\n\n');
end