%% Adaptive Attention Parameter-Update Diagnostic
% MATLAB R2026a
% Synthetic-data test only.
% Does not modify baseline scripts or save research models.

clear;
clc;
rng(42);

fprintf('\n=== Adaptive Attention Parameter-Update Test ===\n');

assert(exist('AdaptiveAttentionLayer', 'class') == 8, ...
    'AdaptiveAttentionLayer.m is not on the MATLAB path.');

%% 1. Create a small synthetic dataset

numObservations = 12;
inputSize = [7 7 32];
numClasses = 4;

X = randn([inputSize numObservations], 'single');

labelValues = repmat(1:numClasses, 1, ...
    ceil(numObservations / numClasses));

labelValues = labelValues(1:numObservations);

Y = categorical(labelValues(:), 1:numClasses, ...
    {'glioma_tumor', 'meningioma_tumor', ...
     'no_tumor', 'pituitary_tumor'});

assert(size(X, 4) == numel(Y), ...
    'Input and label counts do not match.');

fprintf('Input dimensions: %s\n', mat2str(size(X)));

%% 2. Construct a minimal network

layers = [
    imageInputLayer(inputSize, ...
        'Normalization', 'none', ...
        'Name', 'diagnostic_input')

    AdaptiveAttentionLayer(4, 3, 'adaptive_attention')

    globalAveragePooling2dLayer( ...
        'Name', 'diagnostic_gap')

    fullyConnectedLayer(numClasses, ...
        'Name', 'diagnostic_fc')
];

net = dlnetwork(layers);

fprintf('Network construction: PASS\n');

%% 3. Record the initial adaptive-attention parameters

learnablesBefore = net.Learnables;

maskBefore = ...
    string(learnablesBefore.Layer) == "adaptive_attention";

beforeTable = learnablesBefore(maskBefore, :);

parameterNames = [
    "Wc1"; "bc1"; "Wc2"; "bc2"; ...
    "Ws"; "bs"; "Wg1"; "bg1"; "Wg2"; "bg2"
];

actualNames = string(beforeTable.Parameter);

beforeValues = cell(numel(parameterNames), 1);

for k = 1:numel(parameterNames)

    idx = find(actualNames == parameterNames(k), 1);

    assert(~isempty(idx), ...
        'Missing parameter: %s', parameterNames(k));

    beforeValues{k} = gather( ...
        extractdata(beforeTable.Value{idx}));

end

fprintf('Initial parameter capture: PASS\n');

%% 4. Run a short optimization test

options = trainingOptions('adam', ...
    'MaxEpochs', 3, ...
    'MiniBatchSize', 4, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'none');

fprintf('\nStarting synthetic training...\n');

[trainedNet, info] = trainnet( ...
    X, Y, net, 'crossentropy', options); %#ok<NASGU>

fprintf('Synthetic training call completed.\n');

%% 5. Compare parameters before and after optimization

learnablesAfter = trainedNet.Learnables;

maskAfter = ...
    string(learnablesAfter.Layer) == "adaptive_attention";

afterTable = learnablesAfter(maskAfter, :);

afterNames = string(afterTable.Parameter);

allFinite = true;
anyChanged = false;

fprintf('\n=== Parameter Update Report ===\n');

for k = 1:numel(parameterNames)

    name = parameterNames(k);

    idx = find(afterNames == name, 1);

    assert(~isempty(idx), ...
        'Trained parameter missing: %s', name);

    afterValue = gather( ...
        extractdata(afterTable.Value{idx}));

    delta = double(afterValue(:)) ...
          - double(beforeValues{k}(:));

    changeNorm = norm(delta);

    finiteDelta = all(isfinite(delta));
    changed = changeNorm > 0;

    fprintf('%-4s | Change norm: %.12g | Finite: %d | Changed: %d\n', ...
        char(name), changeNorm, finiteDelta, changed);

    allFinite = allFinite && finiteDelta;
    anyChanged = anyChanged || changed;

end

%% 6. Report the diagnostic result

fprintf('\n=== Diagnostic Result ===\n');

if allFinite && anyChanged

    fprintf(['PASS: At least one adaptive-attention parameter ' ...
        'changed during optimization.\n']);

    fprintf(['Next step: verify optimization in the complete ' ...
        'EfficientNet-B0 network.\n']);

else

    fprintf(['FAIL: A valid adaptive-attention parameter update ' ...
        'was not confirmed.\n']);

    fprintf(['Do not begin full research training. Inspect the ' ...
        'layer and optimizer path first.\n']);

end

fprintf('No model was saved by this script.\n');