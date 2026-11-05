%% REGENERATE ADAPTIVE ATTENTION PREDICTIONS
% Uses the saved trained network and original test image paths.
% Does not retrain or overwrite existing results.

clearvars -except ans;
clc;

%% 1. Load original test image ordering

load('EvaluationResults/AdaptiveAttention_ImageLevel_Predictions.mat');

imagePaths = adaptiveResults.ImagePath;
trueLabels = string(adaptiveResults.TrueLabel);

fprintf('Test images: %d\n', numel(imagePaths));

%% 2. Load trained Adaptive Attention network

S = load('TrainedModels/efficientnet_adaptive_attention_softmax.mat');

net = S.netAdaptiveTrained;
classNames = string(S.classNames);

fprintf('Network loaded successfully.\n');
disp(classNames);

%% 3. Create resized image datastore

imds = imageDatastore(cellstr(imagePaths));

assert(isequal(string(imds.Files), imagePaths), ...
    'Image ordering mismatch. Stop execution.');

% Resize every image to the network input dimensions
ads = augmentedImageDatastore([224 224], imds);

%% 4. Generate predictions

fprintf('\nGenerating predictions...\n');

scores = minibatchpredict(net, ads);

scores = gather(extractdata(scores));

% Ensure predictions have one row per image
scores = squeeze(scores);

if size(scores,1) == 4 && size(scores,2) == 489
    scores = scores';
end

if size(scores,1) ~= numel(imagePaths)
    error('Prediction count does not match test image count.');
end

if size(scores,2) ~= numel(classNames)
    error('Score dimensions do not match class count.');
end

[~, predictedIndices] = max(scores,[],2);

predictedLabels = classNames(predictedIndices);
%% 5. Calculate accuracy

accuracy = mean(predictedLabels == trueLabels)*100;

fprintf('\nRegenerated Accuracy: %.4f%%\n', accuracy);

%% 6. Verify against previously saved predictions

saved = load('adaptive_attention_test_results.mat');

savedPredictions = string(saved.predictedLabels);

fprintf('Saved MAT accuracy: %.4f%%\n', ...
    mean(savedPredictions == string(saved.trueLabels))*100);

fprintf('Matching predictions: %d / %d\n', ...
    sum(predictedLabels == savedPredictions), ...
    numel(predictedLabels));

%% 7. Create image-aligned prediction table

regeneratedResults = table( ...
    imagePaths, ...
    trueLabels, ...
    predictedLabels, ...
    'VariableNames', ...
    {'ImagePath','TrueLabel','PredictedLabel'});

%% 8. Save separate output files

outputFolder = 'EvaluationResults/UpdatedAdaptiveAttention';

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

save(fullfile(outputFolder, ...
    'AdaptiveAttention_Aligned_Predictions.mat'), ...
    'regeneratedResults','scores','classNames','accuracy');

writetable(regeneratedResults, ...
    fullfile(outputFolder, ...
    'AdaptiveAttention_Aligned_Predictions.csv'));

fprintf('\nAligned predictions saved successfully.\n');