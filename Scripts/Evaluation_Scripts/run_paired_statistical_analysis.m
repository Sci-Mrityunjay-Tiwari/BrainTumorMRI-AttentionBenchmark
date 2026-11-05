%% PAIRED STATISTICAL ANALYSIS: FIVE BRAIN TUMOUR MODELS
% Common test set: 489 images
% Bootstrap: stratified, paired, 10,000 iterations
% Significance: exact McNemar test
% Multiple comparisons: Holm correction

clearvars -except combinedResults

clc;

%% Load consolidated predictions

T = readtable( ...
    'EvaluationResults/Five_Model_ImageLevel_Predictions.csv');

modelNames = {
    'EfficientNetB0'
    'ChannelAttention'
    'SpatialAttention'
    'CBAM'
    'AdaptiveAttention'
};

trueLabels = categorical(T.TrueLabel);
Y = categorical(T{:,modelNames});

nModels = numel(modelNames);
n = height(T);

assert(n == 489, 'Unexpected test-set size.');
assert(size(Y,1) == n, 'Prediction dimensions mismatch.');

classNames = categories(trueLabels);
nClasses = numel(classNames);

assert(nClasses == 4, 'Expected four classes.');

%% Calculate individual model metrics

accuracy = zeros(nModels,1);
macroPrecision = zeros(nModels,1);
macroRecall = zeros(nModels,1);
macroF1 = zeros(nModels,1);
weightedF1 = zeros(nModels,1);

classF1 = zeros(nModels,nClasses);
confusionMatrices = cell(nModels,1);

for m = 1:nModels

    pred = Y(:,m);

    C = confusionmat(trueLabels,pred, ...
        'Order',categorical(classNames,classNames));

    confusionMatrices{m} = C;

    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2) - TP;

    precision = TP ./ max(TP + FP,1);
    recall = TP ./ max(TP + FN,1);

    f1 = 2 .* precision .* recall ./ ...
        max(precision + recall,eps);

    support = sum(C,2);

    accuracy(m) = sum(TP)/n;
    macroPrecision(m) = mean(precision);
    macroRecall(m) = mean(recall);
    macroF1(m) = mean(f1);
    weightedF1(m) = sum(f1 .* support)/n;

    classF1(m,:) = f1';

end

%% Display performance table

performanceTable = table( ...
    modelNames,accuracy,macroPrecision,macroRecall, ...
    macroF1,weightedF1, ...
    'VariableNames', ...
    {'Model','Accuracy','MacroPrecision', ...
    'MacroRecall','MacroF1','WeightedF1'});

disp('===== FIVE-MODEL PERFORMANCE =====');
disp(performanceTable);

%% Stratified paired bootstrap

rng(42);

nBootstrap = 10000;

bootstrapMetrics = zeros(nBootstrap,nModels,2);

% Metric 1: accuracy
% Metric 2: macro F1

classIndices = cell(nClasses,1);

for c = 1:nClasses
    classIndices{c} = find(trueLabels == classNames{c});
end

for b = 1:nBootstrap

    sampledIndices = [];

    for c = 1:nClasses

        idx = classIndices{c};

        sampledIndices = [sampledIndices; ...
            idx(randi(numel(idx),numel(idx),1))];

    end

    yTrue = trueLabels(sampledIndices);

    for m = 1:nModels

        yPred = Y(sampledIndices,m);

        C = confusionmat(yTrue,yPred, ...
            'Order',categorical(classNames,classNames));

        TP = diag(C);
        FP = sum(C,1)' - TP;
        FN = sum(C,2) - TP;

        precision = TP ./ max(TP + FP,1);
        recall = TP ./ max(TP + FN,1);

        f1 = 2 .* precision .* recall ./ ...
            max(precision + recall,eps);

        bootstrapMetrics(b,m,1) = ...
            mean(yTrue == yPred);

        bootstrapMetrics(b,m,2) = mean(f1);

    end

end

%% Bootstrap confidence intervals for individual models

ciLower = zeros(nModels,2);
ciUpper = zeros(nModels,2);

for m = 1:nModels
    for metric = 1:2

        values = bootstrapMetrics(:,m,metric);

        ci = prctile(values,[2.5 97.5]);

        ciLower(m,metric) = ci(1);
        ciUpper(m,metric) = ci(2);

    end
end

individualCITable = table( ...
    modelNames, ...
    ciLower(:,1),ciUpper(:,1), ...
    ciLower(:,2),ciUpper(:,2), ...
    'VariableNames', ...
    {'Model','Accuracy_CI_Lower','Accuracy_CI_Upper', ...
    'MacroF1_CI_Lower','MacroF1_CI_Upper'});

disp('===== INDIVIDUAL MODEL 95% CONFIDENCE INTERVALS =====');
disp(individualCITable);

%% Paired bootstrap comparisons

pairIndices = nchoosek(1:nModels,2);

nPairs = size(pairIndices,1);

comparisonNames = strings(nPairs,1);

accuracyDifference = zeros(nPairs,1);
accuracyCILower = zeros(nPairs,1);
accuracyCIUpper = zeros(nPairs,1);

macroF1Difference = zeros(nPairs,1);
macroF1CILower = zeros(nPairs,1);
macroF1CIUpper = zeros(nPairs,1);

for p = 1:nPairs

    i = pairIndices(p,1);
    j = pairIndices(p,2);

    comparisonNames(p) = ...
        modelNames{i} + " vs " + modelNames{j};

    % Differences: first model minus second model

    accDiff = bootstrapMetrics(:,i,1) - ...
              bootstrapMetrics(:,j,1);

    f1Diff = bootstrapMetrics(:,i,2) - ...
             bootstrapMetrics(:,j,2);

    accuracyDifference(p) = accuracy(i)-accuracy(j);

    ci = prctile(accDiff,[2.5 97.5]);

    accuracyCILower(p) = ci(1);
    accuracyCIUpper(p) = ci(2);

    macroF1Difference(p) = macroF1(i)-macroF1(j);

    ci = prctile(f1Diff,[2.5 97.5]);

    macroF1CILower(p) = ci(1);
    macroF1CIUpper(p) = ci(2);

end

pairedBootstrapTable = table( ...
    comparisonNames, ...
    accuracyDifference,accuracyCILower,accuracyCIUpper, ...
    macroF1Difference,macroF1CILower,macroF1CIUpper, ...
    'VariableNames', ...
    {'Comparison','AccuracyDifference', ...
    'AccuracyCI_Lower','AccuracyCI_Upper', ...
    'MacroF1Difference','MacroF1CI_Lower','MacroF1CI_Upper'});

disp('===== PAIRED BOOTSTRAP COMPARISONS =====');
disp(pairedBootstrapTable);

%% Exact McNemar tests

mcnemarP = zeros(nPairs,1);
discordantB = zeros(nPairs,1);
discordantC = zeros(nPairs,1);

for p = 1:nPairs

    i = pairIndices(p,1);
    j = pairIndices(p,2);

    correctI = Y(:,i) == trueLabels;
    correctJ = Y(:,j) == trueLabels;

    % b: first model correct, second incorrect
    % c: first model incorrect, second correct

    b = sum(correctI & ~correctJ);
    c = sum(~correctI & correctJ);

    discordantB(p) = b;
    discordantC(p) = c;

    d = b+c;

    if d == 0

        mcnemarP(p) = 1;

    else

        mcnemarP(p) = min(1, ...
            2 * binocdf(min(b,c),d,0.5));

    end

end

%% Holm correction

[sortedP,sortOrder] = sort(mcnemarP);

holmSorted = zeros(nPairs,1);

for k = 1:nPairs

    holmSorted(k) = min(1, ...
        (nPairs-k+1)*sortedP(k));

end

% Enforce monotonicity
for k = 2:nPairs
    holmSorted(k) = max(holmSorted(k),holmSorted(k-1));
end

holmP = zeros(nPairs,1);
holmP(sortOrder) = holmSorted;

significant = holmP < 0.05;

mcnemarTable = table( ...
    comparisonNames,discordantB,discordantC, ...
    mcnemarP,holmP,significant, ...
    'VariableNames', ...
    {'Comparison','FirstCorrectSecondWrong', ...
    'FirstWrongSecondCorrect','ExactMcNemarP', ...
    'HolmAdjustedP','SignificantAfterHolm'});

disp('===== MCNEMAR TESTS WITH HOLM CORRECTION =====');
disp(mcnemarTable);

%% Per-class F1 comparison

classF1Table = array2table(classF1, ...
    'VariableNames',matlab.lang.makeValidName(classNames), ...
    'RowNames',modelNames);

disp('===== PER-CLASS F1 SCORES =====');
disp(classF1Table);

%% Export results

outputFolder = 'EvaluationResults/PairedStatisticalAnalysis';

if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

writetable(performanceTable, ...
    fullfile(outputFolder,'Model_Performance.csv'));

writetable(individualCITable, ...
    fullfile(outputFolder,'Individual_Model_Confidence_Intervals.csv'));

writetable(pairedBootstrapTable, ...
    fullfile(outputFolder,'Paired_Bootstrap_Comparisons.csv'));

writetable(mcnemarTable, ...
    fullfile(outputFolder,'McNemar_Holm_Results.csv'));

writetable(classF1Table, ...
    fullfile(outputFolder,'Per_Class_F1.csv'), ...
    'WriteRowNames',true);

save(fullfile(outputFolder,'Complete_Statistical_Results.mat'));

fprintf('\nSTATISTICAL ANALYSIS COMPLETED.\n');
fprintf('Bootstrap iterations: %d\n',nBootstrap);
fprintf('Pairwise comparisons: %d\n',nPairs);
fprintf('Significance threshold: 0.05\n');