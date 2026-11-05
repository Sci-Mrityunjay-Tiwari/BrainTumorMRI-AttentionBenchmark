%% EfficientNet-B0 + CBAM Brain Tumour MRI Classification
% MATLAB R2026a
%
% Requires these variables from the initialization script:
% classNames, imdsTrain, imdsVal, imdsTest, augTrain, augVal, augTest
%
% New outputs:
%   TrainedModels/efficientnet_cbam_braintumor_run2.mat
%   Results/research_results_v2.csv
%
% This script does not write to the legacy Results/all_results.csv.
%
% IMPORTANT:
% Verify that the custom attention parameters update during training.
% The custom layers currently implement predict but not forward.

clc;

fprintf('\n=============================================================\n');
fprintf('       EFFICIENTNET-B0 + CBAM BRAIN TUMOUR CLASSIFICATION\n');
fprintf('=============================================================\n\n');

%% 1. Validate the initialized workspace

requiredVars = {'classNames','imdsTrain','imdsVal','imdsTest', ...
                'augTrain','augVal','augTest'};

for k = 1:numel(requiredVars)
    assert(exist(requiredVars{k},'var') == 1, ...
        'Required variable "%s" is missing. Run the initialization script first.', ...
        requiredVars{k});
end

assert(isfile('ChannelAttentionLayer.m'), ...
    'ChannelAttentionLayer.m was not found.');
assert(isfile('SpatialAttentionLayer.m'), ...
    'SpatialAttentionLayer.m was not found.');
assert(isfile('computeMetrics.m'), ...
    'computeMetrics.m was not found.');

numClasses = numel(classNames);

assert(numClasses == 4, ...
    'Expected four tumour classes; found %d.', numClasses);

assert(numel(imdsTrain.Files) > 0, 'Training dataset is empty.');
assert(numel(imdsVal.Files) > 0, 'Validation dataset is empty.');
assert(numel(imdsTest.Files) > 0, 'Test dataset is empty.');

fprintf('Number of classes : %d\n', numClasses);
fprintf('Training images   : %d\n', numel(imdsTrain.Files));
fprintf('Validation images : %d\n', numel(imdsVal.Files));
fprintf('Test images       : %d\n', numel(imdsTest.Files));

%% 2. Configure experiment identifier

% Use a new run number for every independent experiment.
runNumber = 2;

experimentID = sprintf( ...
    'EfficientNet_CBAM_run%02d_%s', ...
    runNumber, ...
    char(datetime('now','Format','yyyyMMdd_HHmmss')));

fprintf('\nExperiment ID: %s\n', experimentID);

%% 3. Load pretrained EfficientNet-B0

fprintf('\nLoading pretrained EfficientNet-B0...\n');

netBase = imagePretrainedNetwork("efficientnetb0");
lgraph = layerGraph(netBase);

%% 4. Remove the original classification head

allLayers = lgraph.Layers;
layersToRemove = {};
lastFeatLayer = '';

for i = numel(allLayers):-1:1

    layer = allLayers(i);

    if isa(layer,'nnet.cnn.layer.ClassificationOutputLayer') || ...
       isa(layer,'nnet.cnn.layer.SoftmaxLayer') || ...
       isa(layer,'nnet.cnn.layer.FullyConnectedLayer') || ...
       isa(layer,'nnet.cnn.layer.GlobalAveragePooling2DLayer')

        layersToRemove{end+1} = layer.Name; %#ok<SAGROW>

    else

        lastFeatLayer = layer.Name;
        break;

    end

end

assert(~isempty(lastFeatLayer), ...
    'Could not identify the EfficientNet-B0 feature output layer.');

for i = 1:numel(layersToRemove)
    lgraph = removeLayers(lgraph,layersToRemove{i});
end

fprintf('Original classification head removed.\n');
fprintf('Feature output layer: %s\n',lastFeatLayer);

%% 5. Add CBAM channel attention

fprintf('\nAdding channel attention...\n');

chAtt = ChannelAttentionLayer();
chAtt.Name = 'channel_attention';

lgraph = addLayers(lgraph,chAtt);

lgraph = connectLayers( ...
    lgraph,lastFeatLayer,'channel_attention');

%% 6. Add CBAM spatial attention

fprintf('Adding spatial attention...\n');

spAtt = SpatialAttentionLayer();
spAtt.Name = 'spatial_attention';

lgraph = addLayers(lgraph,spAtt);

lgraph = connectLayers( ...
    lgraph,'channel_attention','spatial_attention');

lastCBAMLayer = 'spatial_attention';

fprintf('CBAM attention module added.\n');

%% 7. Add the new classification head

newHead = [

    globalAveragePooling2dLayer('Name','gap_new')

    fullyConnectedLayer(256, ...
        'Name','fc_new', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10)

    batchNormalizationLayer('Name','bn_new')

    reluLayer('Name','relu_new')

    dropoutLayer(0.5,'Name','drop_new')

    fullyConnectedLayer(numClasses, ...
        'Name','fc_out', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10)

    softmaxLayer('Name','softmax_new')

    classificationLayer('Name','output_new')

];

lgraph = addLayers(lgraph,newHead);

lgraph = connectLayers( ...
    lgraph,lastCBAMLayer,'gap_new');

fprintf('Classification head added.\n');

%% 8. Configure training

maxEpochs = 10;
miniBatchSize = 32;

% Approximate number of iterations per epoch.
iterationsPerEpoch = ceil( ...
    numel(imdsTrain.Files)/miniBatchSize);

validationFrequency = max(1,iterationsPerEpoch);

fprintf('\nTraining configuration:\n');
fprintf('Epochs                : %d\n',maxEpochs);
fprintf('Mini-batch size       : %d\n',miniBatchSize);
fprintf('Initial learning rate : %.6f\n',1e-3);
fprintf('Validation frequency  : %d iterations\n', ...
    validationFrequency);

opts = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',1e-4, ...
    'ValidationData',augVal, ...
    'ValidationFrequency',validationFrequency, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'OutputNetwork','best-validation-loss', ...
    'Verbose',true);

%% 9. Train the network

fprintf('\n=============================================================\n');
fprintf('TRAINING STARTED\n');
fprintf('=============================================================\n');

trainingTimer = tic;

net_cbam = trainNetwork(augTrain,lgraph,opts);

trainingTime = toc(trainingTimer);

fprintf('\nTraining finished.\n');
fprintf('Training time: %.2f minutes\n',trainingTime/60);

%% 10. Evaluate on the held-out test set

fprintf('\n=============================================================\n');
fprintf('TEST-SET EVALUATION\n');
fprintf('=============================================================\n');

trueLabels = imdsTest.Labels;

inferenceTimer = tic;

predLabels = classify(net_cbam,augTest);

inferenceTime = toc(inferenceTimer);

numTestImages = numel(imdsTest.Files);

timePerImage = inferenceTime/numTestImages;

%% 11. Establish consistent class ordering

% Use the same category order for true labels, predictions,
% and the confusion matrix.

classNames = string(classNames);

orderedClasses = categorical(classNames,classNames);

trueLabels = categorical(string(trueLabels),classNames);
predLabels = categorical(string(predLabels),classNames);

assert(~any(isundefined(trueLabels)), ...
    'Some true labels do not match classNames.');

assert(~any(isundefined(predLabels)), ...
    'Some predictions do not match classNames.');

C = confusionmat( ...
    trueLabels,predLabels, ...
    'Order',orderedClasses);

%% 12. Calculate evaluation metrics

% computeMetrics must calculate macro-F1 as the mean of the
% individual class F1 scores, not the harmonic mean of macro
% precision and macro recall.

[acc,prec,rec,f1,spec] = computeMetrics(C);

fprintf('\nConfusion matrix:\n');
disp(array2table(C, ...
    'VariableNames',cellstr(matlab.lang.makeValidName(classNames)), ...
    'RowNames',cellstr(matlab.lang.makeValidName(classNames))));

fprintf('\nOverall accuracy : %.2f%%\n',acc*100);
fprintf('Macro precision  : %.2f%%\n',prec*100);
fprintf('Macro recall     : %.2f%%\n',rec*100);
fprintf('Macro F1         : %.2f%%\n',f1*100);
fprintf('Macro specificity: %.2f%%\n',spec*100);

fprintf('\nTotal inference time: %.4f seconds\n',inferenceTime);
fprintf('Inference per image : %.6f seconds\n',timePerImage);

%% 13. Prepare output directories

modelDir = 'TrainedModels';
resultsDir = 'Results';

if ~isfolder(modelDir)
    mkdir(modelDir);
end

if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

%% 14. Save model without overwriting an existing checkpoint

modelFile = fullfile( ...
    modelDir, ...
    sprintf('efficientnet_cbam_braintumor_run%02d.mat',runNumber));

assert(~isfile(modelFile), ...
    ['Model file already exists: %s\n' ...
     'Change runNumber before running to avoid overwriting it.'], ...
    modelFile);

save(modelFile, ...
    'net_cbam', ...
    'classNames', ...
    'orderedClasses', ...
    'trueLabels', ...
    'predLabels', ...
    'C', ...
    'acc', ...
    'prec', ...
    'rec', ...
    'f1', ...
    'spec', ...
    'trainingTime', ...
    'inferenceTime', ...
    'timePerImage', ...
    'runNumber', ...
    'experimentID', ...
    'maxEpochs', ...
    'miniBatchSize');

fprintf('\nModel saved to:\n%s\n',modelFile);

%% 15. Append metrics to the new research-results CSV

% The legacy Results/all_results.csv is intentionally not modified.

resultsFile = fullfile(resultsDir,'research_results_v2.csv');

headers = { ...
    'ExperimentID', ...
    'Model', ...
    'Run', ...
    'Accuracy', ...
    'MacroPrecision', ...
    'MacroRecall', ...
    'MacroF1', ...
    'MacroSpecificity', ...
    'TrainingTimeSeconds', ...
    'InferenceTimeSeconds', ...
    'InferenceTimePerImageSeconds'};

if ~isfile(resultsFile) || dir(resultsFile).bytes == 0

    writecell(headers,resultsFile);

else

    existingHeaders = readcell(resultsFile,'Range','1:1');

    existingHeaders = string(existingHeaders);
    expectedHeaders = string(headers);

    assert(isequal(existingHeaders(:)',expectedHeaders), ...
        ['The existing results CSV has an unexpected header. ' ...
         'No results row was appended. File: %s'], ...
        resultsFile);

end

newRow = { ...
    experimentID, ...
    'EfficientNet_CBAM', ...
    runNumber, ...
    acc, ...
    prec, ...
    rec, ...
    f1, ...
    spec, ...
    trainingTime, ...
    inferenceTime, ...
    timePerImage};

writecell(newRow,resultsFile,'WriteMode','append');

fprintf('Results appended to:\n%s\n',resultsFile);

%% 16. Final summary

fprintf('\n=============================================================\n');
fprintf('       EFFICIENTNET-CBAM EXPERIMENT FINISHED\n');
fprintf('=============================================================\n');

fprintf('Experiment ID       : %s\n',experimentID);
fprintf('Accuracy            : %.2f%%\n',acc*100);
fprintf('Macro precision     : %.2f%%\n',prec*100);
fprintf('Macro recall        : %.2f%%\n',rec*100);
fprintf('Macro F1            : %.2f%%\n',f1*100);
fprintf('Macro specificity   : %.2f%%\n',spec*100);
fprintf('Training time       : %.2f minutes\n',trainingTime/60);
fprintf('Inference time      : %.4f seconds\n',inferenceTime);
fprintf('Inference per image : %.6f seconds\n',timePerImage);
fprintf('Model file          : %s\n',modelFile);
fprintf('Results CSV         : %s\n\n',resultsFile);