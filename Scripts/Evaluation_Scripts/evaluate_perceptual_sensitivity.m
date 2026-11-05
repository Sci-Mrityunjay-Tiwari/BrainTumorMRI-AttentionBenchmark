%% Perceptual Overlap Sensitivity Analysis
% Uses existing predictions from the frozen model.
% Excludes exact SHA-256 matches and perceptual candidates (pHash <= 5).
% No retraining or inference is performed.

clc;

%% 1. Define paths

rootDir = pwd;

predictionDir = fullfile(rootDir, ...
    'EvaluationResults', 'ExternalValidation');

outputDir = fullfile(predictionDir, ...
    'PerceptualSensitivity');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

briscFile = fullfile(predictionDir, ...
    'BRISC_2025_Predictions.csv');

bdneuroFile = fullfile(predictionDir, ...
    'BDNeuro_MRI_Predictions.csv');

assert(isfile(briscFile), 'BRISC prediction CSV not found.');
assert(isfile(bdneuroFile), 'BDNeuro prediction CSV not found.');

%% 2. Verify workspace variables

assert(exist('briscCombined','var') == 1, ...
    'briscCombined is missing from the workspace.');

assert(exist('bdneuroCombined','var') == 1, ...
    'bdneuroCombined is missing from the workspace.');

briscExcludedPaths = string(briscCombined(:));
bdneuroExcludedPaths = string(bdneuroCombined(:));

%% 3. Define class order

classOrder = ["glioma_tumor", ...
              "meningioma_tumor", ...
              "no_tumor", ...
              "pituitary_tumor"];

%% 4. Load original predictions

briscTable = readtable(briscFile);
bdneuroTable = readtable(bdneuroFile);

requiredColumns = ["ImagePath", "TrueLabel", "PredictedLabel"];

assert(all(ismember(requiredColumns, ...
    string(briscTable.Properties.VariableNames))), ...
    'Required columns missing from BRISC CSV.');

assert(all(ismember(requiredColumns, ...
    string(bdneuroTable.Properties.VariableNames))), ...
    'Required columns missing from BDNeuro CSV.');

%% 5. Exclude perceptual and exact-match candidates

briscPaths = string(briscTable.ImagePath);
bdneuroPaths = string(bdneuroTable.ImagePath);

briscKeep = true(height(briscTable),1);

for i = 1:numel(briscPaths)
    if any(strcmpi(briscPaths(i), briscExcludedPaths))
        briscKeep(i) = false;
    end
end

bdneuroKeep = true(height(bdneuroTable),1);

for i = 1:numel(bdneuroPaths)
    if any(strcmpi(bdneuroPaths(i), bdneuroExcludedPaths))
        bdneuroKeep(i) = false;
    end
end

briscFiltered = briscTable(briscKeep,:);
bdneuroFiltered = bdneuroTable(bdneuroKeep,:);

fprintf('\n========== EXCLUSION SUMMARY ==========\n');

fprintf('BRISC original images: %d\n', height(briscTable));
fprintf('BRISC excluded: %d\n', sum(~briscKeep));
fprintf('BRISC remaining: %d\n', height(briscFiltered));

fprintf('\nBDNeuro original images: %d\n', height(bdneuroTable));
fprintf('BDNeuro excluded: %d\n', sum(~bdneuroKeep));
fprintf('BDNeuro remaining: %d\n', height(bdneuroFiltered));

%% 6. Calculate metrics

datasets = {briscFiltered, bdneuroFiltered};
datasetNames = ["BRISC_2025_PerceptualSensitivity"; ...
                "BDNeuro_PerceptualSensitivity"];

overallResults = table();
classResults = table();

for d = 1:numel(datasets)

    T = datasets{d};

    trueLabels = lower(strtrim(string(T.TrueLabel)));
    predictedLabels = lower(strtrim(string(T.PredictedLabel)));

    trueCat = categorical(trueLabels, classOrder);
    predCat = categorical(predictedLabels, classOrder);

    assert(~any(isundefined(trueCat)), ...
        'Unexpected true labels detected.');

    assert(~any(isundefined(predCat)), ...
        'Unexpected predicted labels detected.');

    C = confusionmat(trueCat, predCat, ...
        'Order', categorical(classOrder, classOrder));

    TP = diag(C);
    FN = sum(C,2) - TP;
    FP = sum(C,1)' - TP;
    TN = sum(C(:)) - TP - FN - FP;

    precision = TP ./ (TP + FP);
    recall = TP ./ (TP + FN);
    specificity = TN ./ (TN + FP);
    f1 = 2 .* precision .* recall ./ (precision + recall);

    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    specificity(isnan(specificity)) = 0;
    f1(isnan(f1)) = 0;

    support = sum(C,2);

    accuracy = sum(TP) / sum(C(:));
    macroPrecision = mean(precision);
    macroRecall = mean(recall);
    macroF1 = mean(f1);

    weightedF1 = sum(f1 .* support) / sum(support);

    %% Overall metrics

    overallRow = table( ...
        datasetNames(d), ...
        sum(C(:)), ...
        accuracy, ...
        macroPrecision, ...
        macroRecall, ...
        macroF1, ...
        weightedF1, ...
        'VariableNames', { ...
        'Dataset','SampleCount','Accuracy', ...
        'MacroPrecision','MacroRecall', ...
        'MacroF1','WeightedF1'});

    overallResults = [overallResults; overallRow];

    %% Class-wise metrics

    classRow = table( ...
        repmat(datasetNames(d),numel(classOrder),1), ...
        classOrder(:), ...
        precision, recall, specificity, f1, support, ...
        'VariableNames', { ...
        'Dataset','Class','Precision','Recall', ...
        'Specificity','F1Score','Support'});

    classResults = [classResults; classRow];

    %% Save confusion matrix

    fig = figure('Visible','off');

cm = confusionchart(C, cellstr(classOrder));
cm.Title = char(datasetNames(d));

    confusionPath = fullfile(outputDir, ...
        char(datasetNames(d) + "_Confusion_Matrix.png"));

    exportgraphics(gcf, confusionPath, 'Resolution',300);
    close(gcf);

    %% Display results

    fprintf('\n========== %s ==========\n', datasetNames(d));

    disp(overallRow);
    disp(classRow);
    disp(C);

end

%% 7. Save filtered predictions and metrics

writetable(briscFiltered, fullfile(outputDir, ...
    'BRISC_PerceptualSensitivity_Predictions.csv'));

writetable(bdneuroFiltered, fullfile(outputDir, ...
    'BDNeuro_PerceptualSensitivity_Predictions.csv'));

writetable(overallResults, fullfile(outputDir, ...
    'PerceptualSensitivity_Overall_Metrics.csv'));

writetable(classResults, fullfile(outputDir, ...
    'PerceptualSensitivity_Classwise_Metrics.csv'));

save(fullfile(outputDir, ...
    'perceptual_sensitivity_results.mat'), ...
    'overallResults','classResults', ...
    'briscFiltered','bdneuroFiltered', ...
    'briscExcludedPaths','bdneuroExcludedPaths');

fprintf('\n========================================\n');
fprintf('PERCEPTUAL SENSITIVITY ANALYSIS COMPLETED\n');
fprintf('Results saved to:\n%s\n', outputDir);
fprintf('========================================\n');