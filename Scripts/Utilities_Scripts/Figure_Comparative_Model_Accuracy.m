clc;
clear;
close all;

%% Comparative Model Accuracy

models = {
    'Spatial Attention'
    'Channel Attention'
    'EfficientNet-B0'
    'EfficientNet-CBAM'
    'Adaptive Attention'
    'ResNet50'
    'GoogLeNet'
    'VGG16'
};

meanAccuracy = [
    96.40
    96.07
    96.02
    95.78
    96.56
    90.89
    89.64
    71.78
];

stdAccuracy = [
    0.79
    0.88
    0.50
    0.82
    0.42
    0.85
    2.34
    1.39
];

%% Figure Configuration

fig = figure('Color','w', ...
    'Units','inches', ...
    'Position',[1 1 11 6.5]);

ax = axes(fig);
hold(ax,'on');

x = 1:length(models);

%% Plot Accuracy Bars

b = bar(ax, x, meanAccuracy, 0.65);

b.FaceColor = [0.25 0.48 0.70];
b.EdgeColor = [0.15 0.15 0.15];
b.LineWidth = 0.8;

%% Highlight Adaptive Attention

adaptiveIndex = 5;

b.CData(adaptiveIndex,:) = [0.85 0.40 0.20];

%% Standard Deviation Error Bars

errorbar(ax, x, meanAccuracy, stdAccuracy, ...
    'k.', ...
    'LineWidth',1.3, ...
    'CapSize',9);

%% Value Labels

for i = 1:length(models)

    text(ax, x(i), meanAccuracy(i)+stdAccuracy(i)+0.5, ...
        sprintf('%.2f',meanAccuracy(i)), ...
        'HorizontalAlignment','center', ...
        'FontSize',10, ...
        'FontWeight','bold');

end

%% Axis Formatting

xticks(ax,x);
xticklabels(ax,models);

xtickangle(ax,25);

ylabel(ax,'Mean Test Accuracy (%)', ...
    'FontSize',13, ...
    'FontWeight','bold');

xlabel(ax,'Classification Model', ...
    'FontSize',13, ...
    'FontWeight','bold');

title(ax,'Comparative Performance of Brain Tumour Classification Models', ...
    'FontSize',15, ...
    'FontWeight','bold');

ylim(ax,[65 100]);

yticks(ax,65:5:100);

grid(ax,'on');
ax.GridAlpha = 0.18;
ax.LineWidth = 1;
ax.FontSize = 10;
ax.FontName = 'Arial';

box(ax,'on');

%% Export High-Resolution Figures

exportgraphics(fig, ...
    'Figure_Comparative_Model_Accuracy.tiff', ...
    'Resolution',600);

exportgraphics(fig, ...
    'Figure_Comparative_Model_Accuracy.pdf', ...
    'ContentType','vector');

disp('Figure 1 generated and exported successfully.');