%% EfficientNet-CBAM Architecture Validation
% Purpose:
% 1. Load pretrained EfficientNet-B0.
% 2. Reproduce the intended CBAM architecture.
% 3. Construct a dlnetwork.
% 4. Verify attention parameters and perform a forward pass.
%
% This is a diagnostic only.
% It does not train a model or overwrite existing results.

clear;
clc;

fprintf('====================================================\n');
fprintf(' EFFICIENTNET-CBAM ARCHITECTURE VALIDATION\n');
fprintf('====================================================\n\n');

%% 1. Confirm custom layer resolution

fprintf('ChannelAttentionLayer:\n');
disp(which('ChannelAttentionLayer'));

fprintf('SpatialAttentionLayer:\n');
disp(which('SpatialAttentionLayer'));

assert(~isempty(which('ChannelAttentionLayer')), ...
    'ChannelAttentionLayer.m is not on the MATLAB path.');

assert(~isempty(which('SpatialAttentionLayer')), ...
    'SpatialAttentionLayer.m is not on the MATLAB path.');

%% 2. Load pretrained EfficientNet-B0

fprintf('\nLoading pretrained EfficientNet-B0...\n');

try
    netBase = imagePretrainedNetwork("efficientnetb0");
    lgraph = layerGraph(netBase);

    fprintf('Pretrained network loaded.\n');

catch ME
    fprintf('FAILED TO LOAD BASE NETWORK:\n%s\n', ME.message);
    return;
end

%% 3. Remove the original classification head
% Follow the layer-removal approach in the existing research script.

allLayers = lgraph.Layers;
layersToRemove = {};

lastFeatLayer = '';

for i = numel(allLayers):-1:1

    currentLayer = allLayers(i);

    if isa(currentLayer, 'nnet.cnn.layer.ClassificationOutputLayer') || ...
       isa(currentLayer, 'nnet.cnn.layer.SoftmaxLayer') || ...
       isa(currentLayer, 'nnet.cnn.layer.FullyConnectedLayer') || ...
       isa(currentLayer, 'nnet.cnn.layer.GlobalAveragePooling2DLayer')

        layersToRemove{end+1} = currentLayer.Name; %#ok<SAGROW>

    else
        lastFeatLayer = currentLayer.Name;
        break;
    end
end

assert(~isempty(lastFeatLayer), ...
    'Could not identify the final feature-extraction layer.');

for i = 1:numel(layersToRemove)
    lgraph = removeLayers(lgraph, layersToRemove{i});
end

fprintf('Original classification head removed.\n');
fprintf('Feature connection point: %s\n', lastFeatLayer);

%% 4. Add channel attention

fprintf('\nAdding channel attention...\n');

channelLayer = ChannelAttentionLayer(16, 'channel_attention');

lgraph = addLayers(lgraph, channelLayer);
lgraph = connectLayers( ...
    lgraph, lastFeatLayer, 'channel_attention');

%% 5. Add spatial attention

fprintf('Adding spatial attention...\n');

spatialLayer = SpatialAttentionLayer(7, 'spatial_attention');

lgraph = addLayers(lgraph, spatialLayer);
lgraph = connectLayers( ...
    lgraph, 'channel_attention', 'spatial_attention');

%% 6. Add a classification head
% Softmax and classification output are omitted because trainnet
% will use the crossentropy loss directly on the output logits.

numClasses = 4;

newHead = [
    globalAveragePooling2dLayer('Name', 'gap_new')

    fullyConnectedLayer(256, ...
        'Name', 'fc_new', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)

    batchNormalizationLayer('Name', 'bn_new')

    reluLayer('Name', 'relu_new')

    dropoutLayer(0.5, 'Name', 'drop_new')

    fullyConnectedLayer(numClasses, ...
        'Name', 'fc_out', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
];

lgraph = addLayers(lgraph, newHead);

lgraph = connectLayers( ...
    lgraph, 'spatial_attention', 'gap_new');

fprintf('Classification head added.\n');

%% 7. Validate the layer graph

fprintf('\nChecking network connections...\n');

try
    analyzeNetwork(lgraph);
    fprintf('Network analysis completed.\n');

catch ME
    fprintf('Network analysis reported an issue:\n%s\n', ME.message);
end

%% 8. Convert to dlnetwork

fprintf('\nConstructing dlnetwork...\n');

try
    netCBAM = dlnetwork(lgraph);

    fprintf('dlnetwork construction successful.\n');

catch ME
    fprintf('\ndlnetwork construction FAILED:\n%s\n', ME.message);

    for k = 1:numel(ME.stack)
        fprintf('  %s, line %d\n', ...
            ME.stack(k).name, ME.stack(k).line);
    end

    return;
end

%% 9. Inspect registered attention parameters

learnables = netCBAM.Learnables;

fprintf('\nRegistered channel-attention parameters:\n');

channelMask = learnables.Layer == "channel_attention";

disp(learnables(channelMask, {'Layer', 'Parameter'}));

fprintf('\nRegistered spatial-attention parameters:\n');

spatialMask = learnables.Layer == "spatial_attention";

disp(learnables(spatialMask, {'Layer', 'Parameter'}));

assert(sum(channelMask) == 4, ...
    'Expected four channel-attention learnable parameters.');

assert(sum(spatialMask) == 2, ...
    'Expected two spatial-attention learnable parameters.');

%% 10. Forward-pass validation

fprintf('\nRunning a synthetic forward pass...\n');

try
    inputSize = netCBAM.Layers(1).InputSize;

    % One synthetic image in SSCB format.
    X = dlarray( ...
        rand(inputSize(1), inputSize(2), inputSize(3), 1, 'single'), ...
        'SSCB');

    scores = predict(netCBAM, X);

    scoresData = extractdata(scores);

    fprintf('Forward pass successful.\n');

    fprintf('Output size:\n');
    disp(size(scoresData));

    fprintf('Output contains NaN: %d\n', ...
        any(isnan(scoresData(:))));

    fprintf('Output contains Inf: %d\n', ...
        any(isinf(scoresData(:))));

    assert(all(isfinite(scoresData(:))), ...
        'Forward pass produced non-finite values.');

catch ME
    fprintf('\nFORWARD PASS FAILED:\n%s\n', ME.message);

    for k = 1:numel(ME.stack)
        fprintf('  %s, line %d\n', ...
            ME.stack(k).name, ME.stack(k).line);
    end

    return;
end

%% 11. Summary

fprintf('\n====================================================\n');
fprintf(' ARCHITECTURE VALIDATION PASSED\n');
fprintf('====================================================\n');

fprintf('Channel-attention parameters: %d\n', sum(channelMask));
fprintf('Spatial-attention parameters: %d\n', sum(spatialMask));
fprintf('Output classes: %d\n', numClasses);

fprintf('\nNo training was performed.\n');
fprintf('No existing model or result file was overwritten.\n');