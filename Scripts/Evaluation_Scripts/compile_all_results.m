%% ========================================================================
% COMPILE ALL RESULTS
%
% Reads all experimental runs from Results/all_results.csv and computes
% statistical summaries for publication.
%
% This script assumes that each model has been trained multiple times
% (e.g., five independent runs).
% ========================================================================

clear;
clc;
close all;

fprintf('\n==============================================================\n');
fprintf('        FINAL RESULTS COMPILATION\n');
fprintf('==============================================================\n\n');

%% ------------------------------------------------------------------------
% Check Results Folder
% -------------------------------------------------------------------------

resultsFolder = 'Results';
resultsFile   = fullfile(resultsFolder,'all_results.csv');

if ~exist(resultsFolder,'dir')
    error('Results folder not found.');
end

if ~isfile(resultsFile)
    error('Results file not found:\n%s',resultsFile);
end

fprintf('Loading experimental results...\n');

%% ------------------------------------------------------------------------
% Read CSV
% -------------------------------------------------------------------------

T = readtable(resultsFile);

fprintf('✓ %d experimental runs loaded.\n\n',height(T));

%% ------------------------------------------------------------------------
% Validate Required Columns
% -------------------------------------------------------------------------

requiredColumns = { ...
    'Model',...
    'Run',...
    'Accuracy___',...
    'Precision___',...
    'Recall___',...
    'F1_Score___',...
    'Specificity___',...
    'Training_Time__min_',...
    'Inference_Time_Image__ms_'};

for i = 1:numel(requiredColumns)

    if ~ismember(requiredColumns{i},T.Properties.VariableNames)

        error('Missing required column: %s',requiredColumns{i});

    end

end

fprintf('✓ CSV format verified.\n\n');

%% ------------------------------------------------------------------------
% Unique Models
% -------------------------------------------------------------------------

modelNames = unique(T.Model,'stable');

numModels = numel(modelNames);

fprintf('Models detected:\n');

for i = 1:numModels
    fprintf('  %d. %s\n',i,string(modelNames(i)));
end

fprintf('\n');

%% ------------------------------------------------------------------------
% Allocate Summary Table
% -------------------------------------------------------------------------

summary = table();

summary.Model = modelNames;

summary.MeanAccuracy      = zeros(numModels,1);
summary.StdAccuracy       = zeros(numModels,1);

summary.MeanPrecision     = zeros(numModels,1);
summary.StdPrecision      = zeros(numModels,1);

summary.MeanRecall        = zeros(numModels,1);
summary.StdRecall         = zeros(numModels,1);

summary.MeanF1            = zeros(numModels,1);
summary.StdF1             = zeros(numModels,1);

summary.MeanSpecificity   = zeros(numModels,1);
summary.StdSpecificity    = zeros(numModels,1);

summary.MeanTrainingTime  = zeros(numModels,1);
summary.StdTrainingTime   = zeros(numModels,1);

summary.MeanInferenceTime = zeros(numModels,1);
summary.StdInferenceTime  = zeros(numModels,1);

fprintf('Computing statistics...\n');

%% ------------------------------------------------------------------------
% Compute Mean and Standard Deviation
% -------------------------------------------------------------------------

for i = 1:numModels

    idx = strcmp(T.Model,modelNames{i});

    summary.MeanAccuracy(i) = mean(T.Accuracy___(idx));
    summary.StdAccuracy(i)  = std(T.Accuracy___(idx));

    summary.MeanPrecision(i) = mean(T.Precision___(idx));
    summary.StdPrecision(i)  = std(T.Precision___(idx));

    summary.MeanRecall(i) = mean(T.Recall___(idx));
    summary.StdRecall(i)  = std(T.Recall___(idx));

    summary.MeanF1(i) = mean(T.F1_Score___(idx));
    summary.StdF1(i)  = std(T.F1_Score___(idx));

    summary.MeanSpecificity(i) = mean(T.Specificity___(idx));
    summary.StdSpecificity(i)  = std(T.Specificity___(idx));

    summary.MeanTrainingTime(i) = ...
        mean(T.Training_Time__min_(idx));

    summary.StdTrainingTime(i) = ...
        std(T.Training_Time__min_(idx));

    summary.MeanInferenceTime(i) = ...
        mean(T.Inference_Time_Image__ms_(idx));

    summary.StdInferenceTime(i) = ...
        std(T.Inference_Time_Image__ms_(idx));

end

fprintf('✓ Statistical analysis complete.\n\n');
%% ------------------------------------------------------------------------
% Create Publication Table
% -------------------------------------------------------------------------

fprintf('Creating publication-ready summary table...\n');

publicationTable = table();

publicationTable.Model = summary.Model;

publicationTable.Accuracy = strings(numModels,1);
publicationTable.Precision = strings(numModels,1);
publicationTable.Recall = strings(numModels,1);
publicationTable.F1Score = strings(numModels,1);
publicationTable.Specificity = strings(numModels,1);
publicationTable.TrainingTime = strings(numModels,1);
publicationTable.InferenceTime = strings(numModels,1);

for i = 1:numModels

    publicationTable.Accuracy(i) = sprintf( ...
        '%.2f ± %.2f', ...
        summary.MeanAccuracy(i), ...
        summary.StdAccuracy(i));

    publicationTable.Precision(i) = sprintf( ...
        '%.2f ± %.2f', ...
        summary.MeanPrecision(i), ...
        summary.StdPrecision(i));

    publicationTable.Recall(i) = sprintf( ...
        '%.2f ± %.2f', ...
        summary.MeanRecall(i), ...
        summary.StdRecall(i));

    publicationTable.F1Score(i) = sprintf( ...
        '%.2f ± %.2f', ...
        summary.MeanF1(i), ...
        summary.StdF1(i));

    publicationTable.Specificity(i) = sprintf( ...
        '%.2f ± %.2f', ...
        summary.MeanSpecificity(i), ...
        summary.StdSpecificity(i));

    publicationTable.TrainingTime(i) = sprintf( ...
        '%.2f ± %.2f', ...
        summary.MeanTrainingTime(i), ...
        summary.StdTrainingTime(i));

    publicationTable.InferenceTime(i) = sprintf( ...
        '%.3f ± %.3f', ...
        summary.MeanInferenceTime(i), ...
        summary.StdInferenceTime(i));

end

%% ------------------------------------------------------------------------
% Sort by Accuracy
% -------------------------------------------------------------------------

summary = sortrows(summary,"MeanAccuracy","descend");

publicationTable = publicationTable( ...
    ismember(publicationTable.Model,summary.Model),:);

publicationTable = publicationTable( ...
    match(publicationTable.Model,summary.Model),:);

%% ------------------------------------------------------------------------
% Save Results
% -------------------------------------------------------------------------

summaryFile = fullfile(resultsFolder,...
    'publication_summary.csv');

writetable(publicationTable,summaryFile);

fprintf('✓ Publication summary saved:\n');
fprintf('  %s\n\n',summaryFile);

disp(publicationTable);
%% ------------------------------------------------------------------------
% Create Figures Folder
% -------------------------------------------------------------------------

figFolder = 'Figures';

if ~exist(figFolder,'dir')
    mkdir(figFolder);
end

%% ------------------------------------------------------------------------
% Sort Models by Mean Accuracy
% -------------------------------------------------------------------------

[~,sortIdx] = sort(summary.MeanAccuracy,'descend');

summary = summary(sortIdx,:);

%% ------------------------------------------------------------------------
% Accuracy Bar Chart
% -------------------------------------------------------------------------

fprintf('Creating accuracy comparison chart...\n');

figure( ...
    'Color','white', ...
    'Units','inches', ...
    'Position',[1 1 12 6]);

barHandle = bar(summary.MeanAccuracy,'FaceColor','flat');

colors = repmat([0.25 0.55 0.85],height(summary),1);

for i = 1:height(summary)

    model = string(summary.Model(i));

    if contains(model,"CBAM")

        colors(i,:) = [0.85 0.20 0.20];

    elseif contains(model,"CA")

        colors(i,:) = [0.20 0.75 0.20];

    end

end

barHandle.CData = colors;

hold on;

errorbar( ...
    1:height(summary), ...
    summary.MeanAccuracy, ...
    summary.StdAccuracy, ...
    '.k', ...
    'LineWidth',1.5);

set(gca,...
    'FontSize',12,...
    'XTick',1:height(summary),...
    'XTickLabel',summary.Model,...
    'XTickLabelRotation',35);

ylabel('Accuracy (%)','FontWeight','bold');

title('Mean Accuracy Across Independent Runs');

ylim([60 100]);

grid on;
box on;

for i = 1:height(summary)

    text( ...
        i,...
        summary.MeanAccuracy(i)+1,...
        sprintf('%.2f',summary.MeanAccuracy(i)),...
        'HorizontalAlignment','center',...
        'FontWeight','bold');

end

exportgraphics( ...
    gcf,...
    fullfile(figFolder,'Accuracy_Comparison.png'),...
    'Resolution',300);

fprintf('✓ Accuracy chart saved.\n');

%% ------------------------------------------------------------------------
% Training Time Comparison
% -------------------------------------------------------------------------

fprintf('Creating training-time chart...\n');

figure( ...
    'Color','white', ...
    'Units','inches', ...
    'Position',[1 1 11 6]);

bar(summary.MeanTrainingTime);

hold on;

errorbar( ...
    1:height(summary), ...
    summary.MeanTrainingTime, ...
    summary.StdTrainingTime, ...
    '.k', ...
    'LineWidth',1.5);

set(gca,...
    'FontSize',12,...
    'XTick',1:height(summary),...
    'XTickLabel',summary.Model,...
    'XTickLabelRotation',35);

ylabel('Training Time (minutes)');

title('Training Time Comparison');

grid on;
box on;

exportgraphics( ...
    gcf,...
    fullfile(figFolder,'Training_Time_Comparison.png'),...
    'Resolution',300);

fprintf('✓ Training-time chart saved.\n');

%% ------------------------------------------------------------------------
% Inference Time Comparison
% -------------------------------------------------------------------------

fprintf('Creating inference-time chart...\n');

figure( ...
    'Color','white', ...
    'Units','inches', ...
    'Position',[1 1 11 6]);

bar(summary.MeanInferenceTime);

hold on;

errorbar( ...
    1:height(summary), ...
    summary.MeanInferenceTime, ...
    summary.StdInferenceTime, ...
    '.k', ...
    'LineWidth',1.5);

set(gca,...
    'FontSize',12,...
    'XTick',1:height(summary),...
    'XTickLabel',summary.Model,...
    'XTickLabelRotation',35);

ylabel('Inference Time (ms/image)');

title('Inference Time Comparison');

grid on;
box on;

exportgraphics( ...
    gcf,...
    fullfile(figFolder,'Inference_Time_Comparison.png'),...
    'Resolution',300);

fprintf('✓ Inference-time chart saved.\n\n');
%% ------------------------------------------------------------------------
% Radar Chart (Top 4 Models)
% -------------------------------------------------------------------------

fprintf('Creating radar chart...\n');

topModels = summary(1:min(4,height(summary)),:);

figure( ...
    'Color','white', ...
    'Units','inches', ...
    'Position',[1 1 9 8]);

theta = linspace(0,2*pi,5);

hold on;

radarColors = [
    0.85 0.20 0.20
    0.20 0.75 0.20
    0.25 0.55 0.85
    0.85 0.60 0.20];

for i = 1:height(topModels)

    values = [
        topModels.MeanAccuracy(i)
        topModels.MeanPrecision(i)
        topModels.MeanRecall(i)
        topModels.MeanF1(i)];

    values(end+1) = values(1);

    polarplot( ...
        theta,...
        values,...
        '-o',...
        'LineWidth',2,...
        'MarkerSize',7,...
        'Color',radarColors(i,:),...
        'DisplayName',string(topModels.Model(i)));

end

ax = gca;

ax.ThetaTick = rad2deg(theta(1:end-1));
ax.ThetaTickLabel = ...
    {'Accuracy','Precision','Recall','F1'};

ax.RLim = [80 100];

title('Top Four Models');

legend('Location','bestoutside');

exportgraphics( ...
    gcf,...
    fullfile(figFolder,'Radar_Chart.png'),...
    'Resolution',300);

fprintf('✓ Radar chart saved.\n\n');

%% ------------------------------------------------------------------------
% Rank Models
% -------------------------------------------------------------------------

summary.Rank = (1:height(summary))';

bestModel = summary.Model(1);

fprintf('==============================================================\n');
fprintf('                 FINAL PERFORMANCE SUMMARY\n');
fprintf('==============================================================\n\n');

fprintf('Ranking:\n\n');

for i = 1:height(summary)

    fprintf('%d. %-25s %.2f ± %.2f %%\n',...
        i,...
        string(summary.Model(i)),...
        summary.MeanAccuracy(i),...
        summary.StdAccuracy(i));

end

fprintf('\n');

%% ------------------------------------------------------------------------
% Baseline vs Proposed
% -------------------------------------------------------------------------

baselineIdx = find(strcmp(summary.Model,'EfficientNetB0'));

cbamIdx = find(strcmp(summary.Model,'EfficientNet-CBAM'));

caIdx = find(strcmp(summary.Model,'EfficientNet-CA'));

saIdx = find(strcmp(summary.Model,'EfficientNet-SA'));

if ~isempty(baselineIdx)

    baselineAcc = summary.MeanAccuracy(baselineIdx);

    fprintf('Baseline Accuracy : %.2f %%\n',baselineAcc);

    if ~isempty(cbamIdx)

        improvement = ...
            summary.MeanAccuracy(cbamIdx)-baselineAcc;

        fprintf('CBAM Improvement  : +%.2f %%\n',improvement);

    end

    if ~isempty(caIdx)

        improvement = ...
            summary.MeanAccuracy(caIdx)-baselineAcc;

        fprintf('CA Improvement    : +%.2f %%\n',improvement);

    end

    if ~isempty(saIdx)

        improvement = ...
            summary.MeanAccuracy(saIdx)-baselineAcc;

        fprintf('SA Improvement    : +%.2f %%\n',improvement);

    end

end

fprintf('\n');

fprintf('Best Overall Model : %s\n',string(bestModel));

fprintf('Mean Accuracy      : %.2f %%\n',...
    summary.MeanAccuracy(1));

fprintf('Std Deviation      : %.2f %%\n',...
    summary.StdAccuracy(1));

fprintf('\n');

%% ------------------------------------------------------------------------
% Save Summary Table
% -------------------------------------------------------------------------

summaryOutput = fullfile(resultsFolder,...
    'statistical_summary.csv');

writetable(summary,summaryOutput);

fprintf('✓ Statistical summary saved.\n');

fprintf('  %s\n\n',summaryOutput);

fprintf('==============================================================\n');
fprintf('Compilation completed successfully.\n');
fprintf('==============================================================\n\n');