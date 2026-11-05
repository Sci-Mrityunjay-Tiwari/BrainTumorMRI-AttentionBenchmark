%% PHASE 1 - STEP 1.22
% Independent external-validation reconstruction and provenance inspection

clc;

baseDir = fullfile('EvaluationResults', 'ExternalValidation');

classNames = {
    'glioma_tumor'
    'meningioma_tumor'
    'no_tumor'
    'pituitary_tumor'
};

%% 1. Independently reconstruct full-validation results

predictionFiles = {
    'BRISC_2025_Predictions.csv'
    'BDNeuro_MRI_Predictions.csv'
};

fprintf('\n========== INDEPENDENT METRIC RECONSTRUCTION ==========\n');

for k = 1:numel(predictionFiles)

    fprintf('\n--- %s ---\n', predictionFiles{k});

    T = readtable(fullfile(baseDir, predictionFiles{k}));

    trueLabels = string(T.TrueLabel);
    predictedLabels = string(T.PredictedLabel);

    C = confusionmat(trueLabels, predictedLabels, ...
        'Order', string(classNames));

    totalImages = sum(C(:));
    correctImages = sum(diag(C));
    accuracy = correctImages / totalImages;

    fprintf('Total images: %d\n', totalImages);
    fprintf('Correct predictions: %d\n', correctImages);
    fprintf('Incorrect predictions: %d\n', totalImages-correctImages);
    fprintf('Reconstructed accuracy: %.4f%%\n', accuracy*100);

    fprintf('Confusion matrix (rows=true, columns=predicted):\n');
    disp(array2table(C, ...
        'VariableNames', classNames, ...
        'RowNames', classNames));

    fprintf('Ground-truth class supports:\n');

    for j = 1:numel(classNames)

        support = sum(trueLabels == classNames{j});

        fprintf('%s: %d\n', classNames{j}, support);

    end

    fprintf('Unique image paths: %d\n', ...
        numel(unique(string(T.ImagePath))));

end

%% 2. Inspect full-validation MAT structure

fprintf('\n========== FULL VALIDATION STRUCT FIELDS ==========\n');

S = load(fullfile(baseDir, 'external_validation_results.mat'));

disp(fieldnames(S.allExternalResults));

%% 3. Inspect nested fields

externalFields = fieldnames(S.allExternalResults);

for k = 1:numel(externalFields)

    fieldName = externalFields{k};

    fprintf('\nField: %s\n', fieldName);

    value = S.allExternalResults.(fieldName);

    fprintf('Class: %s\n', class(value));
    fprintf('Size: %s\n', mat2str(size(value)));

    if isstruct(value)

        disp(fieldnames(value));

    elseif istable(value)

        disp(value(1:min(3,height(value)),:));

    elseif iscell(value)

        disp(value(1:min(5,numel(value))));

    end

end

%% 4. Inspect overlap-controlled MAT structure

fprintf('\n========== OVERLAP-CONTROLLED STRUCT FIELDS ==========\n');

O = load(fullfile(baseDir, ...
    'OverlapControlled', ...
    'overlap_controlled_external_results.mat'));

disp(fieldnames(O.overlapControlledResults));

overlapFields = fieldnames(O.overlapControlledResults);

for k = 1:numel(overlapFields)

    fieldName = overlapFields{k};

    fprintf('\nField: %s\n', fieldName);

    value = O.overlapControlledResults.(fieldName);

    fprintf('Class: %s\n', class(value));
    fprintf('Size: %s\n', mat2str(size(value)));

    if isstruct(value)

        disp(fieldnames(value));

    elseif istable(value)

        disp(value(1:min(3,height(value)),:));

    end

end

%% 5. Inspect dataset path counts

fprintf('\n========== DATASET PATH AUDIT ==========\n');

for k = 1:numel(predictionFiles)

    T = readtable(fullfile(baseDir, predictionFiles{k}));

    paths = string(T.ImagePath);

    fprintf('\n%s\n', predictionFiles{k});
    fprintf('Total records: %d\n', numel(paths));
    fprintf('Unique paths: %d\n', numel(unique(paths)));
    fprintf('Duplicate path records: %d\n', ...
        numel(paths)-numel(unique(paths)));

end

fprintf('\n========== STEP 1.22 COMPLETE ==========\n');