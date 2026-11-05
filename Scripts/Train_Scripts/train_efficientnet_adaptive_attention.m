%% EfficientNet-B0 + Adaptive Attention Training
% MATLAB R2026a
% Dataset: Bhuvaji Brain Tumour MRI
% Four-class classification

clc;

fprintf('\n============================================\n');
fprintf(' EFFICIENTNET-B0 + ADAPTIVE ATTENTION\n');
fprintf('============================================\n');

%% 1. Validate workspace

assert(exist('imdsTrain','var') == 1, ...
    'Run the dataset initialization script first.');

assert(exist('imdsVal','var') == 1);
assert(exist('classNames','var') == 1);

assert(exist('AdaptiveAttentionLayer','class') == 8, ...
    'AdaptiveAttentionLayer.m not found.');

numClasses = numel(classNames);

fprintf('Training images: %d\n',numel(imdsTrain.Files));
fprintf('Validation images: %d\n',numel(imdsVal.Files));
fprintf('Number of classes: %d\n',numClasses);

%% 2. Configure experiment

rng(42);

experimentID = sprintf('AdaptiveAttention_Softmax_%s', ...
    char(datetime('now','Format','yyyyMMdd_HHmmss')));

fprintf('\nExperiment ID: %s\n',experimentID);

%% 3. Load pretrained EfficientNet-B0

fprintf('\nLoading EfficientNet-B0...\n');

netBase = imagePretrainedNetwork("efficientnetb0");

lgraph = layerGraph(netBase);

%% 4. Remove original classification head

layersToRemove = {
    'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool'
    'efficientnet-b0|model|head|dense|MatMul'
    'Softmax'
};

for i = 1:numel(layersToRemove)

    currentNames = string({lgraph.Layers.Name});

    if any(currentNames == layersToRemove{i})
        lgraph = removeLayers(lgraph,layersToRemove{i});
    end

end

featureEndpoint = 'efficientnet-b0|model|head|MulLayer';

assert(any(string({lgraph.Layers.Name}) == featureEndpoint), ...
    'EfficientNet feature endpoint not found.');

%% 5. Add Adaptive Attention

adaptiveLayer = AdaptiveAttentionLayer( ...
    16,7,'adaptive_attention');

lgraph = addLayers(lgraph,adaptiveLayer);

lgraph = connectLayers( ...
    lgraph,featureEndpoint,'adaptive_attention');

%% 6. Add classification head

head = [

    globalAveragePooling2dLayer( ...
        'Name','gap_adaptive')

    fullyConnectedLayer(256, ...
        'Name','fc_adaptive', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10)

    batchNormalizationLayer( ...
        'Name','bn_adaptive')

    reluLayer( ...
        'Name','relu_adaptive')

    dropoutLayer(0.5, ...
        'Name','drop_adaptive')

    fullyConnectedLayer(numClasses, ...
        'Name','fc_adaptive_out', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10)

    softmaxLayer( ...
        'Name','softmax_adaptive')

];

lgraph = addLayers(lgraph,head);

lgraph = connectLayers( ...
    lgraph,'adaptive_attention','gap_adaptive');

%% 7. Construct fresh dlnetwork

fprintf('\nConstructing Adaptive Attention network...\n');

netAdaptive = dlnetwork(lgraph);

netAdaptive = initialize(netAdaptive);

fprintf('Network initialized successfully.\n');

%% 8. Configure training

options = trainingOptions("adam", ...
    'MaxEpochs',10, ...
    'MiniBatchSize',16, ...
    'InitialLearnRate',1e-4, ...
    'L2Regularization',1e-4, ...
    'ValidationData',augVal, ...
    'ValidationFrequency',40, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',true);

%% 9. Train network

fprintf('\n============================================\n');
fprintf(' STARTING ADAPTIVE ATTENTION TRAINING\n');
fprintf('============================================\n');

reset(augTrain);
reset(augVal);

trainingStart = tic;

[netAdaptiveTrained,trainingInfo] = trainnet( ...
    augTrain, ...
    netAdaptive, ...
    "crossentropy", ...
    options);

trainingTime = toc(trainingStart);

fprintf('\nTraining completed.\n');
fprintf('Training time: %.2f minutes\n',trainingTime/60);

%% 10. Save trained model

if ~exist('TrainedModels','dir')
    mkdir('TrainedModels');
end

save(fullfile('TrainedModels', ...
    'efficientnet_adaptive_attention_softmax.mat'), ...
    'netAdaptiveTrained', ...
    'trainingInfo', ...
    'trainingTime', ...
    'experimentID', ...
    'classNames', ...
    '-v7.3');

fprintf('\nCorrected model saved successfully.\n');

fprintf('Experiment ID: %s\n',experimentID);

fprintf('\n============================================\n');
fprintf(' ADAPTIVE ATTENTION TRAINING FINISHED\n');
fprintf('============================================\n');