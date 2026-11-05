clc;
clear;
close all;

%% Adaptive Attention Confusion Matrix
% Values taken from Table 15 of the manuscript.
% Rows = Actual classes
% Columns = Predicted classes

classNames = {
    'Glioma'
    'Meningioma'
    'No Tumour'
    'Pituitary'
};

confMat = [
    131  7  1  0
      9 128  0  3
      2  0 72  1
      0  1  0 134
];

%% Calculate Performance

totalSamples = sum(confMat(:));
correctPredictions = sum(diag(confMat));
accuracy = (correctPredictions / totalSamples) * 100;

fprintf('Total test samples: %d\n',totalSamples);
fprintf('Correct predictions: %d\n',correctPredictions);
fprintf('Accuracy: %.2f%%\n',accuracy);

%% Create Figure

fig = figure( ...
    'Color','w', ...
    'Units','inches', ...
    'Position',[1 1 9 7]);

ax = axes(fig);

imagesc(ax,confMat);

%% Colormap

colormap(ax,parula(256));

cb = colorbar(ax);
cb.Label.String = 'Number of Images';
cb.Label.FontSize = 12;
cb.Label.FontWeight = 'bold';

%% Axis Formatting

axis(ax,'equal');
axis(ax,'tight');

xticks(ax,1:4);
yticks(ax,1:4);

xticklabels(ax,classNames);
yticklabels(ax,classNames);

xlabel(ax,'Predicted Class', ...
    'FontSize',13, ...
    'FontWeight','bold');

ylabel(ax,'Actual Class', ...
    'FontSize',13, ...
    'FontWeight','bold');

title(ax,'Adaptive Attention Confusion Matrix', ...
    'FontSize',16, ...
    'FontWeight','bold');

ax.FontName = 'Arial';
ax.FontSize = 12;
ax.LineWidth = 1.2;

ax.XAxisLocation = 'bottom';

%% Add Cell Annotations

maxValue = max(confMat(:));

for row = 1:4

    for col = 1:4

        value = confMat(row,col);

        if value > maxValue*0.55
            txtColor = 'w';
        else
            txtColor = 'k';
        end

        text(ax,col,row,num2str(value), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',14, ...
            'FontWeight','bold', ...
            'Color',txtColor);

    end

end

%% Add Grid Lines

hold(ax,'on');

for i = 0.5:1:4.5

    plot(ax,[0.5 4.5],[i i], ...
        'k-', ...
        'LineWidth',1.0);

    plot(ax,[i i],[0.5 4.5], ...
        'k-', ...
        'LineWidth',1.0);

end

hold(ax,'off');

%% Add Performance Information

subtitle(ax, ...
    sprintf('Correct Predictions: %d/%d | Accuracy: %.2f%%', ...
    correctPredictions,totalSamples,accuracy), ...
    'FontSize',11);

%% Export Figure

exportgraphics(fig, ...
    'Figure_Adaptive_Attention_Confusion_Matrix.tiff', ...
    'Resolution',600);

exportgraphics(fig, ...
    'Figure_Adaptive_Attention_Confusion_Matrix.pdf', ...
    'ContentType','vector');

disp('Adaptive Attention confusion matrix generated successfully.');