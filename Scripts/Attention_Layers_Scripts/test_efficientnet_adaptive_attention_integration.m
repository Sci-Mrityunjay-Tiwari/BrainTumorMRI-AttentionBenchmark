%% EfficientNet-B0 + Adaptive Attention: Integration Test
% MATLAB R2026a
% Tests:
%   1. Network construction
%   2. Forward propagation
%   3. Full-network gradient propagation
%   4. Adaptive-attention parameter gradients
%   5. New classification-head gradients
%
% No model training or saving is performed.

clearvars -except imdsTrain imdsVal imdsTest ...
    augTrain augVal augTest classNames
clc;

assert(exist('AdaptiveAttentionLayer','class') == 8, ...
    'AdaptiveAttentionLayer.m is not on the MATLAB path.');

fprintf('\n============================================\n');
fprintf('ADAPTIVE ATTENTION INTEGRATION TEST\n');
fprintf('============================================\n');

%% 1. Load pretrained EfficientNet-B0

fprintf('\nLoading EfficientNet-B0...\n');

netBase = imagePretrainedNetwork("efficientnetb0");
lgraph = layerGraph(netBase);

%% 2. Identify the spatial feature endpoint

% Confirmed from the network inspection:
% The final spatial feature map is produced by this multiplication layer.
% Its output feeds the original GlobalAveragePooling2D layer.

featureEndpoint = 'efficientnet-b0|model|head|MulLayer';

allNames = string({lgraph.Layers.Name});

assert(any(allNames == featureEndpoint), ...
    'Expected spatial feature endpoint was not found.');

fprintf('Feature output layer: %s\n', featureEndpoint);

%% 3. Remove the original classification tail

% Remove the original output layers explicitly.
% Preserve all preceding EfficientNet feature-extraction layers.

layersToRemove = {
    'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool'
    'efficientnet-b0|model|head|dense|MatMul'
    'Softmax'
};

for i = 1:numel(layersToRemove)

    layerName = layersToRemove{i};

    currentNames = string({lgraph.Layers.Name});

    if any(currentNames == layerName)
        lgraph = removeLayers(lgraph, layerName);
        fprintf('Removed original head layer: %s\n', layerName);
    else
        warning('Original head layer not found: %s', layerName);
    end
end

%% 4. Add adaptive attention

adaptiveName = 'adaptive_attention';

currentNames = string({lgraph.Layers.Name});

assert(~any(currentNames == adaptiveName), ...
    'An adaptive-attention layer with this name already exists.');

adaptiveLayer = AdaptiveAttentionLayer( ...
    16, 7, adaptiveName);

lgraph = addLayers(lgraph, adaptiveLayer);

lgraph = connectLayers( ...
    lgraph, featureEndpoint, adaptiveName);

fprintf('Adaptive attention connected.\n');

%% 5. Add the four-class classification head

head = [
    globalAveragePooling2dLayer( ...
        'Name', 'gap_adaptive')

    fullyConnectedLayer(256, ...
        'Name', 'fc_adaptive', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)

    batchNormalizationLayer( ...
        'Name', 'bn_adaptive')

    reluLayer( ...
        'Name', 'relu_adaptive')

    dropoutLayer(0.5, ...
        'Name', 'drop_adaptive')

    fullyConnectedLayer(4, ...
        'Name', 'fc_adaptive_out', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
];

lgraph = addLayers(lgraph, head);

lgraph = connectLayers( ...
    lgraph, adaptiveName, 'gap_adaptive');

fprintf('Four-class classification head connected.\n');

%% 6. Construct the dlnetwork

fprintf('\nConstructing dlnetwork...\n');

netAdaptive = dlnetwork(lgraph);

fprintf('Network construction: PASS\n');

%% 7. Test forward propagation

fprintf('\nTesting forward propagation...\n');

% Use multiple samples so batch normalization has meaningful
% batch statistics during the gradient diagnostic.
batchSize = 4;

X = dlarray( ...
    rand(224, 224, 3, batchSize, 'single'), 'SSCB');

Y = forward(netAdaptive, X);

fprintf('Output dimensions: %s\n', mat2str(size(Y)));

assert(size(Y,1) == 4 && size(Y,2) == batchSize, ...
    'Expected four class logits for each input image.');

Ydata = extractdata(Y);

assert(all(isfinite(Ydata), 'all'), ...
    'Network output contains NaN or Inf.');

fprintf('Logit range: %.8g to %.8g\n', ...
    double(min(Ydata, [], 'all')), ...
    double(max(Ydata, [], 'all')));

fprintf('Forward propagation: PASS\n');

%% 8. Test full-network gradient propagation

fprintf('\nTesting full-network gradients...\n');

% One target distribution per input image.
% Shape: [4 classes x batchSize].
target = repmat( ...
    single([0.70; 0.10; 0.10; 0.10]), ...
    1, batchSize);

target = dlarray(target);

[loss, gradients] = dlfeval( ...
    @networkGradientTest, netAdaptive, X, target);

lossValue = double(extractdata(loss));

fprintf('Synthetic cross-entropy loss: %.8f\n', lossValue);

assert(isfinite(lossValue), ...
    'Loss is not finite.');

%% 9. Inspect adaptive-attention gradients

learnables = netAdaptive.Learnables;

layerNames = string(learnables.Layer);
parameterNames = string(learnables.Parameter);

adaptiveRows = find(layerNames == adaptiveName);

assert(~isempty(adaptiveRows), ...
    'Adaptive-attention parameters were not found.');

fprintf('\nAdaptive-attention gradient norms:\n');

adaptiveNorms = zeros(numel(adaptiveRows), 1);

for j = 1:numel(adaptiveRows)

    i = adaptiveRows(j);
    g = gradients.Value{i};

    if isempty(g)

        adaptiveNorms(j) = NaN;

    else

        adaptiveNorms(j) = double(gather(extractdata( ...
            sqrt(sum(g.^2, 'all')))));

    end

    fprintf('%s: %.10g\n', ...
        parameterNames(i), adaptiveNorms(j));

    assert(~isempty(g), ...
        'Gradient is empty for parameter %s.', ...
        parameterNames(i));

    assert(all(isfinite(extractdata(g)), 'all'), ...
        'Non-finite gradient detected for parameter %s.', ...
        parameterNames(i));

end

%% 10. Inspect classification-head gradients

headNames = [
    "fc_adaptive"
    "bn_adaptive"
    "fc_adaptive_out"
];

headRows = find(ismember(layerNames, headNames));

fprintf('\nClassification-head gradient norms:\n');

headNorms = zeros(numel(headRows), 1);

for j = 1:numel(headRows)

    i = headRows(j);
    g = gradients.Value{i};

    if isempty(g)

        headNorms(j) = NaN;

    else

        headNorms(j) = double(gather(extractdata( ...
            sqrt(sum(g.^2, 'all')))));

    end

    fprintf('%s / %s: %.10g\n', ...
        layerNames(i), parameterNames(i), headNorms(j));

end

%% 11. Summarize gradient diagnostics

finiteAdaptiveNorms = adaptiveNorms( ...
    isfinite(adaptiveNorms));

finiteHeadNorms = headNorms( ...
    isfinite(headNorms));

adaptiveHasNonzeroGradient = ...
    any(finiteAdaptiveNorms > 1e-12);

headHasNonzeroGradient = ...
    any(finiteHeadNorms > 1e-12);

fprintf('\n============================================\n');
fprintf('GRADIENT DIAGNOSTIC SUMMARY\n');
fprintf('============================================\n');

if adaptiveHasNonzeroGradient
    fprintf('Adaptive-attention gradients: PASS\n');
else
    fprintf('Adaptive-attention gradients: FAIL\n');
end

if headHasNonzeroGradient
    fprintf('Classification-head gradients: PASS\n');
else
    fprintf('Classification-head gradients: FAIL\n');
end

if adaptiveHasNonzeroGradient && headHasNonzeroGradient

    fprintf('\nGradient checks indicate differentiable paths.\n');
    fprintf('This does not yet establish successful training.\n');

else

    fprintf('\nGradient checks are incomplete or unsuccessful.\n');
    fprintf('Do not start full training yet.\n');
    fprintf('Inspect the custom layer and gradient path.\n');

end

fprintf('\n============================================\n');
fprintf('INTEGRATION TEST FINISHED\n');
fprintf('No model training performed.\n');
fprintf('No files saved by this script.\n');
fprintf('============================================\n');

%%
function [loss, gradients] = ...
    networkGradientTest(net, X, target)

    % Forward pass: expected output [4 classes x batchSize].
    logits = forward(net, X);

    assert(size(logits,1) == 4, ...
        'Network must produce four class logits.');

    assert(size(logits,2) == size(target,2), ...
        'Logit and target batch sizes do not match.');

    % Numerically stable softmax cross-entropy.
    % Calculate probabilities independently for each image.
    shifted = logits - max(logits, [], 1);

    logProb = shifted - log(sum(exp(shifted), 1));

    % Mean cross-entropy across the batch.
    loss = -mean(sum(target .* logProb, 1));

    % Calculate gradients for all learnable parameters.
    gradients = dlgradient(loss, net.Learnables);

end