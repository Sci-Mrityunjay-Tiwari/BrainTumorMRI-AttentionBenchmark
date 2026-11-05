%% BOOTSTRAP CONFIDENCE INTERVALS FOR EXTERNAL VALIDATION
% 95% bootstrap confidence intervals for:
% Accuracy, Macro Precision, Macro Recall, Macro F1,
% Weighted F1, and class-wise Precision, Recall and F1.
%
% Stratified bootstrap: resampling is performed independently
% within each true class to preserve class support.
%
% No model retraining or inference is performed.

clear;
clc;

rng(42);

%% 1. Configuration

projectRoot = pwd;

inputFolder = fullfile(projectRoot, ...
    'EvaluationResults', 'ExternalValidation');

controlledFolder = fullfile(inputFolder, ...
    'OverlapControlled');

outputFolder = fullfile(inputFolder, ...
    'StatisticalAnalysis');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

numBootstrap = 2000;

classOrder = { ...
    'glioma_tumor'
    'meningioma_tumor'
    'no_tumor'
    'pituitary_tumor'};

%% 2. Load all four prediction datasets

datasets = {
    'BRISC_Full', ...
    fullfile(inputFolder, 'BRISC_2025_Predictions.csv');

    'BRISC_OverlapControlled', ...
    fullfile(controlledFolder, ...
    'BRISC_2025_OverlapControlled_Predictions.csv');

    'BDNeuro_Full', ...
    fullfile(inputFolder, 'BDNeuro_MRI_Predictions.csv');

    'BDNeuro_OverlapControlled', ...
    fullfile(controlledFolder, ...
    'BDNeuro_MRI_OverlapControlled_Predictions.csv')
    };

allResults = struct();

%% 3. Bootstrap each dataset

for datasetIndex = 1:size(datasets,1)

    datasetName = datasets{datasetIndex,1};
    predictionFile = datasets{datasetIndex,2};

    fprintf('\n====================================\n');
    fprintf('Dataset: %s\n', datasetName);
    fprintf('====================================\n');

    predictionTable = readtable(predictionFile);

    trueLabels = string(predictionTable.TrueLabel);
    predictedLabels = string(predictionTable.PredictedLabel);

    N = numel(trueLabels);

    % Verify class labels
    assert(all(ismember(trueLabels, string(classOrder))), ...
        'Unexpected true class label detected.');

    assert(all(ismember(predictedLabels, string(classOrder))), ...
        'Unexpected predicted class label detected.');

    % Original point estimates
    originalMetrics = calculateMetrics( ...
        trueLabels, predictedLabels, classOrder);

    fprintf('Images: %d\n', N);
    fprintf('Accuracy: %.4f%%\n', ...
        originalMetrics.Accuracy * 100);

    %% Bootstrap storage

    overallBootstrap = zeros(numBootstrap,5);

    classPrecision = zeros(numBootstrap,4);
    classRecall = zeros(numBootstrap,4);
    classF1 = zeros(numBootstrap,4);

    %% Stratified bootstrap resampling

    for b = 1:numBootstrap

        sampledTrue = strings(0,1);
        sampledPredicted = strings(0,1);

        for c = 1:4

            classIndices = find(trueLabels == classOrder{c});

            nClass = numel(classIndices);

            sampledIndices = classIndices( ...
                randi(nClass,nClass,1));

            sampledTrue = [sampledTrue; ...
                trueLabels(sampledIndices)];

            sampledPredicted = [sampledPredicted; ...
                predictedLabels(sampledIndices)];

        end

        bootstrapMetrics = calculateMetrics( ...
            sampledTrue, sampledPredicted, classOrder);

        overallBootstrap(b,:) = [ ...
            bootstrapMetrics.Accuracy, ...
            bootstrapMetrics.MacroPrecision, ...
            bootstrapMetrics.MacroRecall, ...
            bootstrapMetrics.MacroF1, ...
            bootstrapMetrics.WeightedF1];

        classPrecision(b,:) = bootstrapMetrics.Precision';
        classRecall(b,:) = bootstrapMetrics.Recall';
        classF1(b,:) = bootstrapMetrics.F1Score';

    end

    %% Calculate percentile confidence intervals

    overallNames = { ...
        'Accuracy'
        'MacroPrecision'
        'MacroRecall'
        'MacroF1'
        'WeightedF1'};

    pointValues = [ ...
        originalMetrics.Accuracy
        originalMetrics.MacroPrecision
        originalMetrics.MacroRecall
        originalMetrics.MacroF1
        originalMetrics.WeightedF1];

    lowerCI = zeros(5,1);
    upperCI = zeros(5,1);

    for m = 1:5

        sortedValues = sort(overallBootstrap(:,m));

        lowerCI(m) = sortedValues( ...
            max(1,ceil(0.025*numBootstrap)));

        upperCI(m) = sortedValues( ...
            min(numBootstrap,ceil(0.975*numBootstrap)));

    end

    overallCITable = table( ...
        string(overallNames), ...
        pointValues, lowerCI, upperCI, ...
        'VariableNames', { ...
        'Metric','Estimate','Lower95CI','Upper95CI'});

    %% Class-wise confidence intervals

    classMetricNames = { ...
        'Precision'
        'Recall'
        'F1Score'};

    classPointValues = [ ...
        originalMetrics.Precision, ...
        originalMetrics.Recall, ...
        originalMetrics.F1Score];

    classBootstrapData = { ...
        classPrecision, classRecall, classF1};

    classNamesColumn = strings(12,1);
    metricNamesColumn = strings(12,1);
    estimates = zeros(12,1);
    classLowerCI = zeros(12,1);
    classUpperCI = zeros(12,1);

    row = 0;

    for metricIndex = 1:3

        currentBootstrap = classBootstrapData{metricIndex};

        for c = 1:4

            row = row + 1;

            sortedValues = sort(currentBootstrap(:,c));

            lower = sortedValues( ...
                max(1,ceil(0.025*numBootstrap)));

            upper = sortedValues( ...
                min(numBootstrap,ceil(0.975*numBootstrap)));

            classNamesColumn(row) = string(classOrder{c});

            metricNamesColumn(row) = ...
                string(classMetricNames{metricIndex});

            estimates(row) = classPointValues(c,metricIndex);

            classLowerCI(row) = lower;
            classUpperCI(row) = upper;

        end

    end

    classCITable = table( ...
        classNamesColumn, metricNamesColumn, ...
        estimates, classLowerCI, classUpperCI, ...
        'VariableNames', { ...
        'Class','Metric','Estimate','Lower95CI','Upper95CI'});

    %% Display results

    fprintf('\nOverall metrics with 95%% CI:\n');
    disp(overallCITable);

    fprintf('\nClass-wise metrics with 95%% CI:\n');
    disp(classCITable);

    %% Store results

    allResults.(datasetName).SampleCount = N;

    allResults.(datasetName).OverallMetrics = ...
        overallCITable;

    allResults.(datasetName).ClassMetrics = ...
        classCITable;

    allResults.(datasetName).BootstrapOverall = ...
        overallBootstrap;

    allResults.(datasetName).BootstrapPrecision = ...
        classPrecision;

    allResults.(datasetName).BootstrapRecall = ...
        classRecall;

    allResults.(datasetName).BootstrapF1 = ...
        classF1;

    %% Export CSV files

    writetable(overallCITable, fullfile(outputFolder, ...
        [datasetName '_Overall_95CI.csv']));

    writetable(classCITable, fullfile(outputFolder, ...
        [datasetName '_Classwise_95CI.csv']));

end

%% 4. Save complete statistical analysis

save(fullfile(outputFolder, ...
    'external_validation_bootstrap_results.mat'), ...
    'allResults', 'numBootstrap', 'classOrder', '-v7.3');

fprintf('\n====================================\n');
fprintf('BOOTSTRAP ANALYSIS COMPLETED\n');
fprintf('Bootstrap iterations: %d\n', numBootstrap);
fprintf('Confidence level: 95%%\n');
fprintf('Results folder:\n%s\n', outputFolder);
fprintf('====================================\n');


%% LOCAL FUNCTION

function metrics = calculateMetrics( ...
    trueLabels, predictedLabels, classOrder)

    CM = zeros(4,4);

    for i = 1:4
        for j = 1:4

            CM(i,j) = sum( ...
                trueLabels == classOrder{i} & ...
                predictedLabels == classOrder{j});

        end
    end

    TP = diag(CM);
    FP = sum(CM,1)' - TP;
    FN = sum(CM,2) - TP;

    TN = sum(CM(:)) - TP - FP - FN;

    precision = TP ./ (TP + FP);
    recall = TP ./ (TP + FN);
    specificity = TN ./ (TN + FP);

    f1 = 2 .* precision .* recall ./ ...
        (precision + recall);

    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    specificity(isnan(specificity)) = 0;
    f1(isnan(f1)) = 0;

    support = sum(CM,2);

    metrics.Accuracy = sum(TP) / sum(CM(:));

    metrics.MacroPrecision = mean(precision);
    metrics.MacroRecall = mean(recall);
    metrics.MacroF1 = mean(f1);

    metrics.WeightedF1 = ...
        sum(f1 .* support) / sum(support);

    metrics.Precision = precision;
    metrics.Recall = recall;
    metrics.Specificity = specificity;
    metrics.F1Score = f1;

end