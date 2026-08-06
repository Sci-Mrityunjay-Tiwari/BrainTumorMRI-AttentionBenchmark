%% Load trained model
load('TrainedModels/efficientnet_sp_only_braintumor.mat');

%% Create normalized confusion matrix
figure('Color','w','Position',[100 100 850 700]);

cm = confusionchart(C,...
    classNames,...
    'Normalization','row-normalized');

cm.Title = 'Normalized Confusion Matrix';
cm.FontSize = 16;
cm.FontName = 'Times New Roman';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

exportgraphics(gcf,...
    'Figures/Fig5_ConfusionMatrix_EfficientNetSA.png',...
    'Resolution',600);