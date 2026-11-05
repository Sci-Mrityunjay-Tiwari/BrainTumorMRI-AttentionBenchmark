%% TEST: EfficientNet-B0 Adaptive Attention Optimization
% Purpose:
% 1. Verify that EfficientNet-B0 backbone remains frozen.
% 2. Verify that adaptive attention parameters can update.
% 3. Verify that new classification head parameters can update.
% 4. Verify that updated parameters remain finite.
%
% No model or training results are saved.

clear;
clc;
close all;

fprintf('\n============================================\n');
fprintf('ADAPTIVE ATTENTION OPTIMIZATION TEST\n');
fprintf('============================================\n');

%% STEP 1: Load pretrained EfficientNet-B0

fprintf('\n[1/8] Loading pretrained EfficientNet-B0...\n');

net = imagePretrainedNetwork("efficientnetb0");

fprintf('Pretrained network loaded successfully.\n');

%% STEP 2: Convert network into layer graph

fprintf('\n[2/8] Creating layer graph...\n');

lgraph = layerGraph(net);

%% STEP 3: Identify feature extraction endpoint

fprintf('\n[3/8] Identifying EfficientNet feature endpoint...\n');

endpoint = "efficientnet-b0|model|head|MulLayer";

layerNames = string({lgraph.Layers.Name});

if ~any(layerNames == endpoint)
    error('Expected EfficientNet feature endpoint was not found.');
end

fprintf('Feature endpoint confirmed: %s\n', endpoint);

%% STEP 4: Inspect and remove original classification head

fprintf('\n[4/8] Inspecting original classification head...\n');

allLayerNames = string({lgraph.Layers.Name});

fprintf('\nRemaining layers after endpoint:\n');

for i = 1:numel(allLayerNames)
    fprintf('%s\n', allLayerNames(i));
end

layersToRemove = [
    "efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool"
    "efficientnet-b0|model|head|dense|MatMul"
    "efficientnet-b0|model|head|Softmax"
    ];

for i = 1:numel(layersToRemove)

    layerName = layersToRemove(i);

    currentNames = string({lgraph.Layers.Name});

    if any(currentNames == layerName)

        lgraph = removeLayers(lgraph, layerName);

        fprintf('Removed: %s\n', layerName);

    else

        fprintf('Not present (skipped): %s\n', layerName);

    end

end

fprintf('\nClassification head inspection completed.\n');

%% STEP 5: Remove remaining Softmax layer

fprintf('\n[5/8] Removing remaining Softmax layer...\n');

currentNames = string({lgraph.Layers.Name});

if any(currentNames == "Softmax")

    lgraph = removeLayers(lgraph, "Softmax");

    fprintf('Original Softmax layer removed successfully.\n');

else

    fprintf('Softmax layer already removed.\n');

end

currentNames = string({lgraph.Layers.Name});

if any(contains(currentNames, "global_average_pooling2d")) || ...
        any(contains(currentNames, "dense")) || ...
        any(currentNames == "Softmax")

    error('Original classification components still remain.');

else

    fprintf('Original classification head removal verified.\n');

end

%% STEP 6: Add Adaptive Attention Layer

fprintf('\n[6/8] Adding adaptive attention layer...\n');

adaptiveLayer = AdaptiveAttentionLayer(16, 7, 'adaptive_attention');

lgraph = addLayers(lgraph, adaptiveLayer);

lgraph = connectLayers(lgraph, endpoint, 'adaptive_attention');

fprintf('Adaptive attention layer added successfully.\n');

currentNames = string({lgraph.Layers.Name});

if any(currentNames == "adaptive_attention")

    fprintf('Adaptive attention layer presence verified.\n');

else

    error('Adaptive attention layer was not added successfully.');

end

connections = lgraph.Connections;

connectionExists = any( ...
    string(connections.Source) == endpoint & ...
    string(connections.Destination) == "adaptive_attention");

if connectionExists

    fprintf('EfficientNet-to-adaptive-attention connection verified.\n');

else

    error('Connection to adaptive attention layer was not established.');

end

fprintf('\nSTEP 6 completed successfully.\n');

%% STEP 7: Add New Classification Head

fprintf('\n[7/8] Adding new classification head...\n');

newHead = [
    globalAveragePooling2dLayer('Name', 'new_global_pool')
    fullyConnectedLayer(256, ...
    'Name', 'new_fc_256', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10)
    batchNormalizationLayer('Name', 'new_batch_norm')
    reluLayer('Name', 'new_relu')
    dropoutLayer(0.5, 'Name', 'new_dropout')
    fullyConnectedLayer(4, ...
    'Name', 'new_fc_output', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10)
    ];

lgraph = addLayers(lgraph, newHead);

lgraph = connectLayers( ...
    lgraph, ...
    'adaptive_attention', ...
    'new_global_pool');

fprintf('New classification head added successfully.\n');

requiredLayers = [
    "new_global_pool"
    "new_fc_256"
    "new_batch_norm"
    "new_relu"
    "new_dropout"
    "new_fc_output"
    ];

currentNames = string({lgraph.Layers.Name});

for i = 1:numel(requiredLayers)

    if any(currentNames == requiredLayers(i))

        fprintf('Verified: %s\n', requiredLayers(i));

    else

        error('Missing classification layer: %s', requiredLayers(i));

    end

end

fprintf('\nSTEP 7 completed successfully.\n');

%% STEP 8: Convert to dlnetwork and Inspect Learnables

fprintf('\n[8/8] Converting network and inspecting learnable parameters...\n');

dlnet = dlnetwork(lgraph);

fprintf('Network converted to dlnetwork successfully.\n');

learnables = dlnet.Learnables;

fprintf('\nTotal learnable parameter entries: %d\n', height(learnables));

adaptiveMask = string(learnables.Layer) == "adaptive_attention";

adaptiveLearnables = learnables(adaptiveMask, :);

fprintf('\nAdaptive attention learnables:\n');
disp(adaptiveLearnables(:, {'Layer', 'Parameter'}));

headNames = [
    "new_fc_256"
    "new_batch_norm"
    "new_fc_output"
    ];

headMask = ismember(string(learnables.Layer), headNames);

headLearnables = learnables(headMask, :);

fprintf('\nNew classification head learnables:\n');
disp(headLearnables(:, {'Layer', 'Parameter'}));

backboneMask = ...
    string(learnables.Layer) ~= "adaptive_attention" & ...
    ~ismember(string(learnables.Layer), headNames);

backboneLearnables = learnables(backboneMask, :);

fprintf('\nBackbone learnable parameter entries: %d\n', ...
    height(backboneLearnables));

expectedAdaptiveParameters = [
    "Wc1"
    "bc1"
    "Wc2"
    "bc2"
    "Ws"
    "bs"
    "Wg1"
    "bg1"
    "Wg2"
    "bg2"
    ];

actualAdaptiveParameters = string(adaptiveLearnables.Parameter);

for i = 1:numel(expectedAdaptiveParameters)

    if any(actualAdaptiveParameters == expectedAdaptiveParameters(i))

        fprintf('Verified adaptive parameter: %s\n', ...
            expectedAdaptiveParameters(i));

    else

        error('Missing adaptive parameter: %s', ...
            expectedAdaptiveParameters(i));

    end

end

fprintf('\nSTEP 8 construction and learnable inspection completed.\n');

%% STEP 9: Record Initial Parameter Values

fprintf('\n[9/9] Recording initial parameter values...\n');

initialLearnables = dlnet.Learnables;

adaptiveMask = ...
    string(initialLearnables.Layer) == "adaptive_attention";

initialAdaptiveLearnables = ...
    initialLearnables(adaptiveMask, :);

headNames = [
    "new_fc_256"
    "new_batch_norm"
    "new_fc_output"
    ];

headMask = ismember( ...
    string(initialLearnables.Layer), headNames);

initialHeadLearnables = ...
    initialLearnables(headMask, :);

backboneMask = ...
    ~adaptiveMask & ~headMask;

initialBackboneLearnables = ...
    initialLearnables(backboneMask, :);

initialAdaptiveValues = initialAdaptiveLearnables.Value;
initialHeadValues = initialHeadLearnables.Value;
initialBackboneValues = initialBackboneLearnables.Value;

fprintf('\nInitial parameter groups recorded:\n');

fprintf('Backbone entries: %d\n', ...
    numel(initialBackboneValues));

fprintf('Adaptive attention entries: %d\n', ...
    numel(initialAdaptiveValues));

fprintf('Classification head entries: %d\n', ...
    numel(initialHeadValues));

for i = 1:numel(initialAdaptiveValues)

    if any(~isfinite(extractdata(initialAdaptiveValues{i})), 'all')
        error('Non-finite initial adaptive parameter detected.');
    end

end

for i = 1:numel(initialHeadValues)

    if any(~isfinite(extractdata(initialHeadValues{i})), 'all')
        error('Non-finite initial classification head parameter detected.');
    end

end

for i = 1:numel(initialBackboneValues)

    if any(~isfinite(extractdata(initialBackboneValues{i})), 'all')
        error('Non-finite initial backbone parameter detected.');
    end

end

fprintf('\nAll initial parameter values are finite.\n');
fprintf('STEP 9 completed successfully.\n');

%% STEP 10: Inspect Backbone Layer Types

fprintf('\n[10/10] Inspecting backbone layer configuration...\n');

backboneLayerNames = unique(string(initialBackboneLearnables.Layer));

fprintf('\nTotal backbone layers with learnable parameters: %d\n', ...
    numel(backboneLayerNames));

fprintf('\nRepresentative backbone layer types and properties:\n');

layersToInspect = [
    backboneLayerNames(1)
    backboneLayerNames(round(numel(backboneLayerNames)/2))
    backboneLayerNames(end)
    ];

for i = 1:numel(layersToInspect)

    layerName = layersToInspect(i);

    layerIndex = find( ...
        string({dlnet.Layers.Name}) == layerName, 1);

    if isempty(layerIndex)
        fprintf('\nLayer not found in dlnet.Layers: %s\n', layerName);
        continue;
    end

    currentLayer = dlnet.Layers(layerIndex);

    fprintf('\nLayer name: %s\n', layerName);
    fprintf('Layer class: %s\n', class(currentLayer));

    fprintf('Available learn-rate related properties:\n');

    propertyNames = properties(currentLayer);

    matchingProperties = propertyNames( ...
        contains(lower(string(propertyNames)), 'learnrate'));

    if isempty(matchingProperties)
        fprintf('No learn-rate properties exposed.\n');
    else
        disp(matchingProperties);
    end

end

fprintf('\nSTEP 10 inspection completed.\n');

%% STEP 11: Audit Learn-Rate Properties Across Backbone

fprintf('\n[11/11] Auditing all backbone learn-rate properties...\n');

backboneLayerNames = unique(string(initialBackboneLearnables.Layer));

missingProperties = strings(0,1);
auditedLayers = 0;

for i = 1:numel(backboneLayerNames)

    layerName = backboneLayerNames(i);

    layerIndex = find( ...
        string({dlnet.Layers.Name}) == layerName, 1);

    if isempty(layerIndex)
        missingProperties(end+1) = layerName;
        continue;
    end

    currentLayer = dlnet.Layers(layerIndex);

    parameterNames = string( ...
        initialBackboneLearnables.Parameter( ...
        string(initialBackboneLearnables.Layer) == layerName));

    propertyNames = string(properties(currentLayer));

    for j = 1:numel(parameterNames)

        parameterName = parameterNames(j);

        expectedProperty = parameterName + "LearnRateFactor";

        if ~any(propertyNames == expectedProperty)
            missingProperties(end+1) = ...
                layerName + " -> " + expectedProperty;
        end

    end

    auditedLayers = auditedLayers + 1;

end

fprintf('\nBackbone layers audited: %d\n', auditedLayers);
fprintf('Missing learn-rate properties: %d\n', numel(missingProperties));

if isempty(missingProperties)

    fprintf('PASS: Every backbone learnable has a corresponding learn-rate property.\n');

else

    fprintf('WARNING: Some backbone learnables lack corresponding properties:\n');
    disp(missingProperties);

end

fprintf('\nSTEP 11 audit completed.\n');

%% STEP 12: Corrected Learn-Rate Property Audit

fprintf('\n[12/12] Running corrected backbone property audit...\n');

backboneLayerNames = unique(string(initialBackboneLearnables.Layer));

missingEntries = strings(0,1);
auditedEntries = 0;

for i = 1:height(initialBackboneLearnables)

    layerName = string(initialBackboneLearnables.Layer(i));
    parameterName = string(initialBackboneLearnables.Parameter(i));

    layerIndex = find( ...
        string({dlnet.Layers.Name}) == layerName, 1);

    if isempty(layerIndex)
        missingEntries(end+1) = ...
            layerName + " | " + parameterName + " | Layer not found";
        continue;
    end

    currentLayer = dlnet.Layers(layerIndex);
    propertyNames = string(properties(currentLayer));

    switch parameterName

        case "Weights"
            expectedProperty = "WeightLearnRateFactor";

        case "Bias"
            expectedProperty = "BiasLearnRateFactor";

        case "Offset"
            expectedProperty = "OffsetLearnRateFactor";

        case "Scale"
            expectedProperty = "ScaleLearnRateFactor";

        otherwise
            expectedProperty = parameterName + "LearnRateFactor";

    end

    if any(propertyNames == expectedProperty)

        auditedEntries = auditedEntries + 1;

    else

        missingEntries(end+1) = ...
            layerName + " | " + parameterName + ...
            " | " + class(currentLayer);

    end

end

fprintf('\nTotal backbone learnable entries: %d\n', ...
    height(initialBackboneLearnables));

fprintf('Entries with matching properties: %d\n', auditedEntries);

fprintf('Entries requiring investigation: %d\n', ...
    numel(missingEntries));

if ~isempty(missingEntries)

    fprintf('\nDetailed list of unmatched entries:\n');

    for i = 1:numel(missingEntries)
        fprintf('%d. %s\n', i, missingEntries(i));
    end

else

    fprintf('\nPASS: All backbone learnables have matching properties.\n');

end

fprintf('\nSTEP 12 diagnostic completed.\n');

%% STEP 13: Freezing EfficientNet-B0 Backbone

fprintf('\n[13/13] Freezing EfficientNet-B0 backbone...\n');

backboneLayerNames = unique(string(initialBackboneLearnables.Layer));

frozenLayerCount = 0;

for i = 1:numel(backboneLayerNames)

    layerName = backboneLayerNames(i);

    layerIndex = find( ...
        string({dlnet.Layers.Name}) == layerName, 1);

    if isempty(layerIndex)
        error('Backbone layer not found: %s', layerName);
    end

    currentLayer = dlnet.Layers(layerIndex);

    if isprop(currentLayer, 'WeightLearnRateFactor')
        currentLayer.WeightLearnRateFactor = 0;
    end

    if isprop(currentLayer, 'BiasLearnRateFactor')
        currentLayer.BiasLearnRateFactor = 0;
    end

    if isprop(currentLayer, 'OffsetLearnRateFactor')
        currentLayer.OffsetLearnRateFactor = 0;
    end

    if isprop(currentLayer, 'ScaleLearnRateFactor')
        currentLayer.ScaleLearnRateFactor = 0;
    end

    dlnet = replaceLayer(dlnet, layerName, currentLayer);

    frozenLayerCount = frozenLayerCount + 1;

end

fprintf('\nBackbone layers processed: %d\n', frozenLayerCount);

fprintf('Backbone learn-rate factors set to zero.\n');

fprintf('\nSTEP 13 completed.\n');

%% STEP 14: Verifying Frozen Backbone Configuration

fprintf('\n[14/14] Verifying backbone freeze configuration...\n');

backboneLayerNames = unique(string(initialBackboneLearnables.Layer));

nonZeroFactors = strings(0,1);
checkedFactors = 0;

for i = 1:numel(backboneLayerNames)

    layerName = backboneLayerNames(i);

    layerIndex = find( ...
        string({dlnet.Layers.Name}) == layerName, 1);

    currentLayer = dlnet.Layers(layerIndex);

    factorProperties = [
        "WeightLearnRateFactor"
        "BiasLearnRateFactor"
        "OffsetLearnRateFactor"
        "ScaleLearnRateFactor"
        ];

    for j = 1:numel(factorProperties)

        propertyName = factorProperties(j);

        if isprop(currentLayer, propertyName)

            factorValue = currentLayer.(char(propertyName));

            checkedFactors = checkedFactors + 1;

            if any(factorValue ~= 0, 'all')

                nonZeroFactors(end+1) = ...
                    layerName + " | " + propertyName;

            end

        end

    end

end

fprintf('\nBackbone layers checked: %d\n', ...
    numel(backboneLayerNames));

fprintf('Learn-rate factors checked: %d\n', checkedFactors);

fprintf('Non-zero backbone factors: %d\n', ...
    numel(nonZeroFactors));

if isempty(nonZeroFactors)

    fprintf('\nPASS: All inspected backbone learn-rate factors are zero.\n');

else

    fprintf('\nWARNING: Non-zero factors detected:\n');

    disp(nonZeroFactors);

end

fprintf('\nSTEP 14 completed.\n');

%% STEP 15: Verifying Trainable Parameter Configuration

fprintf('\n[15/15] Verifying trainable parameter configuration...\n');

adaptiveMask = string(dlnet.Learnables.Layer) == "adaptive_attention";

headNames = [
    "new_fc_256"
    "new_batch_norm"
    "new_fc_output"
    ];

headMask = ismember(string(dlnet.Learnables.Layer), headNames);

backboneMask = ~adaptiveMask & ~headMask;

adaptiveLearnables = dlnet.Learnables(adaptiveMask, :);
headLearnables = dlnet.Learnables(headMask, :);
backboneLearnables = dlnet.Learnables(backboneMask, :);

fprintf('\nParameter group summary:\n');

fprintf('Adaptive attention parameters: %d\n', ...
    height(adaptiveLearnables));

fprintf('Classification head parameters: %d\n', ...
    height(headLearnables));

fprintf('Backbone parameters: %d\n', ...
    height(backboneLearnables));

assert(height(adaptiveLearnables) == 10, ...
    'Unexpected adaptive attention parameter count.');

assert(height(headLearnables) == 6, ...
    'Unexpected classification head parameter count.');

assert(height(backboneLearnables) == 260, ...
    'Unexpected backbone parameter count.');

fprintf('\nAdaptive attention parameters:\n');
disp(adaptiveLearnables(:, {'Layer', 'Parameter'}));

fprintf('\nClassification head parameters:\n');
disp(headLearnables(:, {'Layer', 'Parameter'}));

allTrainableLearnables = [adaptiveLearnables; headLearnables];

for i = 1:height(allTrainableLearnables)

    parameterValue = allTrainableLearnables.Value{i};

    assert(all(isfinite(extractdata(parameterValue)), 'all'), ...
        'Non-finite trainable parameter detected.');

end

fprintf('\nPASS: Expected trainable parameter groups identified.\n');
fprintf('PASS: All trainable parameter values are finite.\n');

fprintf('\nSTEP 15 completed successfully.\n');

%% STEP 16: Prepare Explicit Optimizer Parameter Masks

fprintf('\n[16/16] Preparing explicit optimizer parameter masks...\n');

learnablesTable = dlnet.Learnables;

adaptiveMask = ...
    string(learnablesTable.Layer) == "adaptive_attention";

headNames = [
    "new_fc_256"
    "new_batch_norm"
    "new_fc_output"
    ];

headMask = ismember( ...
    string(learnablesTable.Layer), headNames);

backboneMask = ~adaptiveMask & ~headMask;

trainableMask = adaptiveMask | headMask;

fprintf('\nOptimizer parameter groups:\n');

fprintf('Adaptive attention parameters: %d\n', ...
    nnz(adaptiveMask));

fprintf('Classification head parameters: %d\n', ...
    nnz(headMask));

fprintf('Frozen backbone parameters: %d\n', ...
    nnz(backboneMask));

fprintf('Total trainable parameters: %d\n', ...
    nnz(trainableMask));

assert(nnz(adaptiveMask) == 10, ...
    'Incorrect adaptive attention parameter count.');

assert(nnz(headMask) == 6, ...
    'Incorrect classification head parameter count.');

assert(nnz(backboneMask) == 260, ...
    'Incorrect backbone parameter count.');

assert(nnz(trainableMask) == 16, ...
    'Incorrect total trainable parameter count.');

assert(all((adaptiveMask + headMask + backboneMask) == 1), ...
    'Parameter groups overlap or contain unclassified parameters.');

fprintf('\nPASS: 16 parameters identified for optimization.\n');
fprintf('PASS: 260 backbone parameters identified for freezing.\n');
fprintf('PASS: Parameter groups are mutually exclusive.\n');

fprintf('\nSTEP 16 completed successfully.\n');

%% STEP 18: Isolated Adaptive-Layer Autodiff Diagnostic
%
% This diagnostic tests AdaptiveAttentionLayer independently from
% EfficientNet-B0 and the classification head.
%
% It is placed before STEP 17 because STEP 17 previously encountered
% an error during dlfeval on the complete network.

fprintf('\n[18] Running isolated adaptive-layer autodiff diagnostic...\n');

% Create an input layer with the expected EfficientNet feature dimensions
testInputLayer = imageInputLayer([7 7 1280], ...
    'Normalization', 'none', ...
    'Name', 'test_input');

% Create an independent adaptive attention layer
testAdaptiveLayer = AdaptiveAttentionLayer(16, 7, 'test_attention');

% Construct the isolated test graph
testGraph = layerGraph();

testGraph = addLayers(testGraph, testInputLayer);
testGraph = addLayers(testGraph, testAdaptiveLayer);

testGraph = connectLayers( ...
    testGraph, 'test_input', 'test_attention');

% Convert the isolated graph to dlnetwork
testNet = dlnetwork(testGraph);

% Generate deterministic synthetic feature input
rng(42);

testImages = single(rand(7, 7, 1280, 2));

dlTestX = dlarray(testImages, 'SSCB');

fprintf('Isolated input size: %s\n', mat2str(size(testImages)));

% Forward-pass check
testOutput = forward(testNet, dlTestX);

fprintf('Isolated output size: %s\n', mat2str(size(testOutput)));

assert(all(isfinite(extractdata(testOutput)), 'all'), ...
    'Isolated adaptive-layer output contains non-finite values.');

fprintf('PASS: Isolated forward pass produced finite output.\n');

% Compute isolated gradients
[testLoss, testGradients] = dlfeval( ...
    @isolatedAdaptiveGradientTest, testNet, dlTestX);

fprintf('Isolated diagnostic loss: %.8f\n', ...
    double(extractdata(testLoss)));

% Verify gradient table size
assert(height(testGradients) == height(testNet.Learnables), ...
    'Isolated gradient table does not match network learnables.');

% Inspect gradient values
isolatedGradientNorms = zeros(height(testGradients), 1);

for i = 1:height(testGradients)

    gradientData = extractdata(testGradients.Value{i});

    assert(all(isfinite(gradientData), 'all'), ...
        'Non-finite gradient detected in isolated adaptive layer.');

    isolatedGradientNorms(i) = double(norm(gradientData(:)));

end

fprintf('Isolated learnable entries: %d\n', height(testGradients));

fprintf('Isolated non-zero gradients: %d\n', ...
    nnz(isolatedGradientNorms > 0));

assert(all(isolatedGradientNorms > 0), ...
    'At least one isolated adaptive-layer learnable has a zero gradient.');

fprintf('\nPASS: Isolated adaptive layer completed forward and autodiff tests.\n');
fprintf('STEP 18 completed successfully.\n');

%% STEP 17: Compute Synthetic-Batch Gradients

fprintf('\n[17/17] Computing synthetic-batch gradients...\n');

rng(42);

syntheticImages = single(rand(224, 224, 3, 4));

syntheticTargets = single([
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
]');

dlX = dlarray(syntheticImages, 'SSCB');
dlT = dlarray(syntheticTargets, 'CB');

fprintf('Synthetic image batch: %s\n', ...
    mat2str(size(syntheticImages)));

fprintf('Synthetic target batch: %s\n', ...
    mat2str(size(syntheticTargets)));

[loss, gradients] = dlfeval( ...
    @computeAdaptiveGradients, dlnet, dlX, dlT);

fprintf('\nSynthetic loss: %.6f\n', ...
    double(extractdata(loss)));

assert(height(gradients) == height(dlnet.Learnables), ...
    'Gradient table does not match network learnables.');

gradientLayers = string(gradients.Layer);

adaptiveGradientMask = ...
    gradientLayers == "adaptive_attention";

headGradientMask = ...
    ismember(gradientLayers, headNames);

backboneGradientMask = ...
    ~adaptiveGradientMask & ~headGradientMask;

adaptiveGradientNorms = zeros(nnz(adaptiveGradientMask), 1);
headGradientNorms = zeros(nnz(headGradientMask), 1);
backboneGradientNorms = zeros(nnz(backboneGradientMask), 1);

% Inspect adaptive attention gradients
indices = find(adaptiveGradientMask);

for i = 1:numel(indices)

    gradientValue = gradients.Value{indices(i)};
    gradientData = extractdata(gradientValue);

    assert(all(isfinite(gradientData), 'all'), ...
        'Non-finite adaptive attention gradient detected.');

    adaptiveGradientNorms(i) = ...
        double(norm(gradientData(:)));

end

% Inspect classification head gradients
indices = find(headGradientMask);

for i = 1:numel(indices)

    gradientValue = gradients.Value{indices(i)};
    gradientData = extractdata(gradientValue);

    assert(all(isfinite(gradientData), 'all'), ...
        'Non-finite classification head gradient detected.');

    headGradientNorms(i) = ...
        double(norm(gradientData(:)));

end

% Inspect backbone gradients
indices = find(backboneGradientMask);

for i = 1:numel(indices)

    gradientValue = gradients.Value{indices(i)};
    gradientData = extractdata(gradientValue);

    assert(all(isfinite(gradientData), 'all'), ...
        'Non-finite backbone gradient detected.');

    backboneGradientNorms(i) = ...
        double(norm(gradientData(:)));

end

fprintf('\nGradient summary:\n');

fprintf('Adaptive attention entries: %d\n', ...
    numel(adaptiveGradientNorms));

fprintf('Adaptive non-zero gradients: %d\n', ...
    nnz(adaptiveGradientNorms > 0));

fprintf('Classification head entries: %d\n', ...
    numel(headGradientNorms));

fprintf('Classification head non-zero gradients: %d\n', ...
    nnz(headGradientNorms > 0));

fprintf('Backbone entries: %d\n', ...
    numel(backboneGradientNorms));

fprintf('Backbone non-zero gradients: %d\n', ...
    nnz(backboneGradientNorms > 0));

assert(any(adaptiveGradientNorms > 0), ...
    'No non-zero adaptive attention gradients detected.');

assert(any(headGradientNorms > 0), ...
    'No non-zero classification head gradients detected.');

fprintf('\nPASS: Adaptive attention gradients are finite.\n');
fprintf('PASS: Classification head gradients are finite.\n');
fprintf('PASS: Backbone gradients are finite.\n');

fprintf('\nSTEP 17 completed successfully.\n');

%% LOCAL FUNCTION: Gradient Computation

function [loss, gradients] = computeAdaptiveGradients(net, X, T)

    predictions = forward(net, X);

    probabilities = softmax(predictions);

    loss = crossentropy(probabilities, T);

    gradients = dlgradient(loss, net.Learnables);

end

%% LOCAL FUNCTION: Isolated Adaptive-Layer Gradient Test

function [loss, gradients] = isolatedAdaptiveGradientTest(net, X)

    % Forward propagation through the isolated adaptive layer
    Y = forward(net, X);

    % Simple scalar objective for the isolated gradient test
    loss = mean(Y.^2, 'all');

    % Calculate gradients for isolated network learnables
    gradients = dlgradient(loss, net.Learnables);

end