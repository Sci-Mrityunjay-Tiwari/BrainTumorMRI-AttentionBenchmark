%% OVERLAP-CONTROLLED EXTERNAL VALIDATION
% Uses saved predictions and SHA-256 audit records.
% Does NOT retrain or modify the trained model.
% Does NOT modify original datasets or full-set results.

clearvars;
clc;
close all;

%% 1. Project paths

projectRoot = pwd;

inputFolder = fullfile(projectRoot, ...
    'EvaluationResults', 'ExternalValidation');

auditFile = fullfile(projectRoot, ...
    'EvaluationResults', 'external_duplicate_audit.mat');

outputFolder = fullfile(inputFolder, 'OverlapControlled');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% 2. Load SHA-256 audit records

load(auditFile, 'auditTable');

bhuvajiRows = auditTable( ...
    strcmp(auditTable.Dataset, 'Bhuvaji'), :);

bhuvajiHashes = unique(string(bhuvajiRows.SHA256));

briscRows = auditTable( ...
    strcmp(auditTable.Dataset, 'BRISC_Test'), :);

bdneuroRows = auditTable( ...
    strcmp(auditTable.Dataset, 'BDNeuro_Test'), :);

briscMatches = briscRows( ...
    ismember(string(briscRows.SHA256), bhuvajiHashes), :);

bdneuroMatches = bdneuroRows( ...
    ismember(string(bdneuroRows.SHA256), bhuvajiHashes), :);

briscExcludedPaths = unique(string(briscMatches.FilePath));
bdneuroExcludedPaths = unique(string(bdneuroMatches.FilePath));

fprintf('\nSHA-256 overlap audit:\n');
fprintf('BRISC exact-match files: %d\n', ...
    numel(briscExcludedPaths));
fprintf('BDNeuro exact-match files: %d\n', ...
    numel(bdneuroExcludedPaths));

%% 3. Load original prediction results

briscPred = readtable(fullfile(inputFolder, ...
    'BRISC_2025_Predictions.csv'));

bdneuroPred = readtable(fullfile(inputFolder, ...
    'BDNeuro_MRI_Predictions.csv'));

%% 4. Exclude exact SHA-256 matches

briscPaths = string(briscPred.ImagePath);
bdneuroPaths = string(bdneuroPred.ImagePath);

briscKeep = ~ismember(lower(briscPaths), ...
    lower(briscExcludedPaths));

bdneuroKeep = ~ismember(lower(bdneuroPaths), ...
    lower(bdneuroExcludedPaths));

briscClean = briscPred(briscKeep, :);
bdneuroClean = bdneuroPred(bdneuroKeep, :);

fprintf('\nOverlap-controlled dataset sizes:\n');
fprintf('BRISC: %d original - %d excluded = %d remaining\n', ...
    height(briscPred), sum(~briscKeep), height(briscClean));

fprintf('BDNeuro: %d original - %d excluded = %d remaining\n', ...
    height(bdneuroPred), sum(~bdneuroKeep), height(bdneuroClean));

%% 5. Verify no exact matches remain

assert(~any(ismember(lower(string(briscClean.ImagePath)), ...
    lower(briscExcludedPaths))), ...
    'BRISC exact duplicates remain after exclusion.');

assert(~any(ismember(lower(string(bdneuroClean.ImagePath)), ...
    lower(bdneuroExcludedPaths))), ...
    'BDNeuro exact duplicates remain after exclusion.');

assert(height(briscClean) == 722, ...
    'Unexpected BRISC remaining image count.');

assert(height(bdneuroClean) == 639, ...
    'Unexpected BDNeuro remaining image count.');

%% 6. Define class order

classOrder = { ...
    'glioma_tumor'
    'meningioma_tumor'
    'no_tumor'
    'pituitary_tumor'};

%% 7. Calculate metrics

[briscMetrics, briscCM, briscAccuracy, ...
    briscMacroPrecision, briscMacroRecall, ...
    briscMacroF1, briscWeightedF1] = ...
    calculateMetrics(briscClean, classOrder);

[bdneuroMetrics, bdneuroCM, bdneuroAccuracy, ...
    bdneuroMacroPrecision, bdneuroMacroRecall, ...
    bdneuroMacroF1, bdneuroWeightedF1] = ...
    calculateMetrics(bdneuroClean, classOrder);

%% 8. Display BRISC results

fprintf('\n========================================\n');
fprintf('BRISC 2025 - OVERLAP-CONTROLLED RESULTS\n');
fprintf('========================================\n');

fprintf('Images evaluated: %d\n', height(briscClean));
fprintf('Accuracy: %.4f%%\n', briscAccuracy * 100);
fprintf('Macro Precision: %.4f\n', briscMacroPrecision);
fprintf('Macro Recall: %.4f\n', briscMacroRecall);
fprintf('Macro F1: %.4f\n', briscMacroF1);
fprintf('Weighted F1: %.4f\n', briscWeightedF1);

disp('BRISC confusion matrix:');
disp(array2table(briscCM, ...
    'VariableNames', classOrder, ...
    'RowNames', classOrder));

disp('BRISC class-wise metrics:');
disp(briscMetrics);

%% 9. Display BDNeuro results

fprintf('\n========================================\n');
fprintf('BDNeuro-MRI - OVERLAP-CONTROLLED RESULTS\n');
fprintf('========================================\n');

fprintf('Images evaluated: %d\n', height(bdneuroClean));
fprintf('Accuracy: %.4f%%\n', bdneuroAccuracy * 100);
fprintf('Macro Precision: %.4f\n', bdneuroMacroPrecision);
fprintf('Macro Recall: %.4f\n', bdneuroMacroRecall);
fprintf('Macro F1: %.4f\n', bdneuroMacroF1);
fprintf('Weighted F1: %.4f\n', bdneuroWeightedF1);

disp('BDNeuro confusion matrix:');
disp(array2table(bdneuroCM, ...
    'VariableNames', classOrder, ...
    'RowNames', classOrder));

disp('BDNeuro class-wise metrics:');
disp(bdneuroMetrics);

%% 10. Save overlap-controlled predictions

writetable(briscClean, fullfile(outputFolder, ...
    'BRISC_2025_OverlapControlled_Predictions.csv'));

writetable(bdneuroClean, fullfile(outputFolder, ...
    'BDNeuro_MRI_OverlapControlled_Predictions.csv'));

%% 11. Save class-wise metrics

writetable(briscMetrics, fullfile(outputFolder, ...
    'BRISC_2025_OverlapControlled_Metrics.csv'));

writetable(bdneuroMetrics, fullfile(outputFolder, ...
    'BDNeuro_MRI_OverlapControlled_Metrics.csv'));

%% 12. Generate confusion matrix figures

fig1 = figure('Name', 'BRISC Overlap-Controlled Confusion Matrix');

confusionchart(briscCM, classOrder);
title('BRISC 2025 - Overlap-Controlled Evaluation');

saveas(fig1, fullfile(outputFolder, ...
    'BRISC_2025_OverlapControlled_Confusion_Matrix.png'));

fig2 = figure('Name', 'BDNeuro Overlap-Controlled Confusion Matrix');

confusionchart(bdneuroCM, classOrder);
title('BDNeuro-MRI - Overlap-Controlled Evaluation');

saveas(fig2, fullfile(outputFolder, ...
    'BDNeuro_MRI_OverlapControlled_Confusion_Matrix.png'));

%% 13. Save complete results

overlapControlledResults.BRISC.Predictions = briscClean;
overlapControlledResults.BRISC.ExcludedPaths = briscExcludedPaths;
overlapControlledResults.BRISC.ExcludedCount = sum(~briscKeep);
overlapControlledResults.BRISC.Metrics = briscMetrics;
overlapControlledResults.BRISC.ConfusionMatrix = briscCM;
overlapControlledResults.BRISC.Accuracy = briscAccuracy;
overlapControlledResults.BRISC.MacroPrecision = briscMacroPrecision;
overlapControlledResults.BRISC.MacroRecall = briscMacroRecall;
overlapControlledResults.BRISC.MacroF1 = briscMacroF1;
overlapControlledResults.BRISC.WeightedF1 = briscWeightedF1;

overlapControlledResults.BDNeuro.Predictions = bdneuroClean;
overlapControlledResults.BDNeuro.ExcludedPaths = bdneuroExcludedPaths;
overlapControlledResults.BDNeuro.ExcludedCount = sum(~bdneuroKeep);
overlapControlledResults.BDNeuro.Metrics = bdneuroMetrics;
overlapControlledResults.BDNeuro.ConfusionMatrix = bdneuroCM;
overlapControlledResults.BDNeuro.Accuracy = bdneuroAccuracy;
overlapControlledResults.BDNeuro.MacroPrecision = bdneuroMacroPrecision;
overlapControlledResults.BDNeuro.MacroRecall = bdneuroMacroRecall;
overlapControlledResults.BDNeuro.MacroF1 = bdneuroMacroF1;
overlapControlledResults.BDNeuro.WeightedF1 = bdneuroWeightedF1;

overlapControlledResults.ClassOrder = classOrder;

save(fullfile(outputFolder, ...
    'overlap_controlled_external_results.mat'), ...
    'overlapControlledResults', '-v7.3');

fprintf('\n========================================\n');
fprintf('OVERLAP-CONTROLLED EVALUATION COMPLETED\n');
fprintf('========================================\n');
fprintf('Results saved in:\n%s\n', outputFolder);


%% LOCAL FUNCTION

function [metricsTable, CM, accuracy, macroPrecision, ...
    macroRecall, macroF1, weightedF1] = ...
    calculateMetrics(predictionTable, classOrder)

    trueLabels = string(predictionTable.TrueLabel);
    predictedLabels = string(predictionTable.PredictedLabel);

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

    f1Score = 2 .* precision .* recall ./ ...
        (precision + recall);

    % Handle undefined metrics safely
    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    specificity(isnan(specificity)) = 0;
    f1Score(isnan(f1Score)) = 0;

    support = sum(CM,2);

    accuracy = sum(TP) / sum(CM(:));

    macroPrecision = mean(precision);
    macroRecall = mean(recall);
    macroF1 = mean(f1Score);

    weightedF1 = sum(f1Score .* support) / sum(support);

    metricsTable = table( ...
        string(classOrder(:)), ...
        precision, recall, specificity, ...
        f1Score, support, ...
        'VariableNames', { ...
        'Class', 'Precision', 'Recall', ...
        'Specificity', 'F1Score', 'Support'});
end