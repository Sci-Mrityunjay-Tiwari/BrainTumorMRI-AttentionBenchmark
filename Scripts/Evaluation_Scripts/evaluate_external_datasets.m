%% EXTERNAL VALIDATION: ADAPTIVE ATTENTION MODEL
% BRISC 2025 and BDNeuro-MRI
% MATLAB R2026a
%
% IMPORTANT:
% - Uses the previously trained model.
% - Does not retrain or modify model weights.
% - Evaluates ONLY external test partitions.
% - Preserves the model's built-in Z-score normalization.

clc;

fprintf('\n============================================\n');
fprintf('ADAPTIVE ATTENTION: EXTERNAL VALIDATION\n');
fprintf('============================================\n');

%% 1. Load trained model

modelFile = fullfile(pwd, ...
    'TrainedModels', ...
    'efficientnet_adaptive_attention_softmax.mat');

assert(isfile(modelFile), ...
    'Trained model file not found.');

S = load(modelFile);

netAdaptiveTrained = S.netAdaptiveTrained;
classNames = S.classNames;

inputSize = [224 224 3];

fprintf('\nTrained model loaded successfully.\n');
fprintf('Experiment: %s\n', S.experimentID);

%% 2. Define dataset locations

briscTestPath = fullfile(pwd, ...
    'brisc2025', ...
    'classification_task', ...
    'test');

bdneuroTestPath = fullfile(pwd, ...
    'BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI', ...
    'test');

assert(isfolder(briscTestPath), ...
    'BRISC test folder not found. Check briscTestPath.');

assert(isfolder(bdneuroTestPath), ...
    'BDNeuro test folder not found. Check bdneuroTestPath.');

%% 3. Define evaluation datasets

datasetPaths = {
    briscTestPath
    bdneuroTestPath
};

datasetNames = {
    'BRISC_2025'
    'BDNeuro_MRI'
};

%% 4. Create output directory

outputPath = fullfile(pwd, ...
    'EvaluationResults', ...
    'ExternalValidation');

if ~exist(outputPath,'dir')
    mkdir(outputPath);
end

%% 5. Evaluate both datasets

allExternalResults = struct();

for d = 1:numel(datasetPaths)

    fprintf('\n============================================\n');
    fprintf('DATASET: %s\n',datasetNames{d});
    fprintf('============================================\n');

    %% Load test images

    imds = imageDatastore(datasetPaths{d}, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

    fprintf('Images detected: %d\n',numel(imds.Files));

    %% Map external class names to model class names

    externalLabels = string(imds.Labels);

    mappedLabels = strings(size(externalLabels));

    mappedLabels(externalLabels == "glioma") = "glioma_tumor";

    mappedLabels(externalLabels == "meningioma") = ...
        "meningioma_tumor";

    mappedLabels(externalLabels == "no_tumor") = "no_tumor";

    mappedLabels(externalLabels == "pituitary") = ...
        "pituitary_tumor";

    assert(all(mappedLabels ~= ""), ...
        'Unrecognized class folder detected.');

    imds.Labels = categorical( ...
        mappedLabels,classNames,classNames);

    %% Prepare images

    augExternal = augmentedImageDatastore( ...
        inputSize, ...
        imds, ...
        'ColorPreprocessing','gray2rgb');

    %% Initialize prediction variables

    trueLabels = categorical();
    predictedLabels = categorical();

    allScores = [];

    %% Run frozen model inference

    reset(augExternal);

    fprintf('\nRunning model inference...\n');

    batchNumber = 0;

    while hasdata(augExternal)

        batch = read(augExternal);

        X = cat(4,batch.input{:});

        dlX = dlarray(single(X),'SSCB');

        if canUseGPU
            dlX = gpuArray(dlX);
        end

        Y = predict(netAdaptiveTrained,dlX);

        scores = gather(extractdata(Y));

        [~,predIdx] = max(scores,[],1);

        batchPredictions = categorical( ...
            predIdx(:), ...
            1:numel(classNames), ...
            classNames);

        trueLabels = [trueLabels; batch.response(:)];

        predictedLabels = [predictedLabels; batchPredictions];

        allScores = [allScores,scores];

        batchNumber = batchNumber + 1;

        if mod(batchNumber,10) == 0
            fprintf('Processed batches: %d | Images: %d\n', ...
                batchNumber,numel(trueLabels));
        end

    end

    %% Verify evaluation count

    assert(numel(trueLabels) == numel(imds.Files), ...
        'Some images were not evaluated.');

    %% Confusion matrix

    classOrder = categorical(classNames,classNames);

    C = confusionmat( ...
        trueLabels, ...
        predictedLabels, ...
        'Order',classOrder);

    %% Classification metrics

    TP = diag(C);

    FP = sum(C,1)' - TP;

    FN = sum(C,2) - TP;

    TN = sum(C,'all') - TP - FP - FN;

    precision = TP ./ max(TP + FP,1);

    recall = TP ./ max(TP + FN,1);

    specificity = TN ./ max(TN + FP,1);

    f1Score = 2 .* precision .* recall ./ ...
        max(precision + recall,eps);

    support = sum(C,2);

    accuracy = sum(TP) / sum(C,'all');

    macroPrecision = mean(precision);

    macroRecall = mean(recall);

    macroF1 = mean(f1Score);

    weightedF1 = sum(f1Score .* support) / sum(support);

    %% Display results

    fprintf('\n---------- RESULTS ----------\n');

    fprintf('Dataset: %s\n',datasetNames{d});

    fprintf('Total images: %d\n',sum(C,'all'));

    fprintf('Accuracy: %.4f%%\n',accuracy*100);

    fprintf('Macro Precision: %.4f\n',macroPrecision);

    fprintf('Macro Recall: %.4f\n',macroRecall);

    fprintf('Macro F1: %.4f\n',macroF1);

    fprintf('Weighted F1: %.4f\n',weightedF1);

    %% Per-class metrics table

    metricsTable = table( ...
        string(classNames(:)), ...
        precision, ...
        recall, ...
        specificity, ...
        f1Score, ...
        support, ...
        'VariableNames',{ ...
        'Class','Precision','Recall', ...
        'Specificity','F1Score','Support'});

    disp(metricsTable);

    %% Confusion matrix table

    confusionTable = array2table(C, ...
        'VariableNames',matlab.lang.makeValidName(classNames), ...
        'RowNames',matlab.lang.makeValidName(classNames));

    disp(confusionTable);

    %% Save confusion matrix figure

    fig = figure('Visible','off');

    confusionchart(C,classNames);

    title(sprintf('%s - Adaptive Attention',datasetNames{d}));

    saveas(fig,fullfile(outputPath, ...
        [datasetNames{d} '_Confusion_Matrix.png']));

    close(fig);

    %% Save individual predictions

    predictionTable = table( ...
        string(imds.Files), ...
        string(trueLabels), ...
        string(predictedLabels), ...
        'VariableNames',{ ...
        'ImagePath','TrueLabel','PredictedLabel'});

    writetable(predictionTable,fullfile(outputPath, ...
        [datasetNames{d} '_Predictions.csv']));

    %% Save individual metrics

    writetable(metricsTable,fullfile(outputPath, ...
        [datasetNames{d} '_Classification_Metrics.csv']));

    %% Store results

    result.datasetName = datasetNames{d};

    result.totalImages = sum(C,'all');

    result.accuracy = accuracy;

    result.macroPrecision = macroPrecision;

    result.macroRecall = macroRecall;

    result.macroF1 = macroF1;

    result.weightedF1 = weightedF1;

    result.confusionMatrix = C;

    result.metricsTable = metricsTable;

    result.predictionTable = predictionTable;

    result.scores = allScores;

    allExternalResults.(datasetNames{d}) = result;

    fprintf('\nDataset evaluation completed: %s\n', ...
        datasetNames{d});

end

%% 6. Save combined external evaluation results

save(fullfile(outputPath, ...
    'external_validation_results.mat'), ...
    'allExternalResults', ...
    'classNames', ...
    '-v7.3');

fprintf('\n============================================\n');
fprintf('EXTERNAL VALIDATION COMPLETED\n');
fprintf('============================================\n');

fprintf('\nResults saved in:\n%s\n',outputPath);