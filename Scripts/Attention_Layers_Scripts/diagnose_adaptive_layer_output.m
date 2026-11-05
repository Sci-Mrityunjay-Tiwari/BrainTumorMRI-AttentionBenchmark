%% Diagnose Adaptive Attention on Real EfficientNet Features
% MATLAB R2026a
% No training. No model files saved.

clear;
clc;

fprintf('\n============================================\n');
fprintf('ADAPTIVE LAYER OUTPUT DIAGNOSTIC\n');
fprintf('============================================\n');

%% 1. Load EfficientNet-B0

netBase = imagePretrainedNetwork("efficientnetb0");
lgraph = layerGraph(netBase);

%% 2. Remove original classification head

tailLayers = {
    'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool'
    'efficientnet-b0|model|head|dense|MatMul'
    'Softmax'
};

for i = 1:numel(tailLayers)

    names = string({lgraph.Layers.Name});

    if any(names == tailLayers{i})
        lgraph = removeLayers(lgraph, tailLayers{i});
    end
end

%% 3. Run feature extractor

netFeatures = dlnetwork(lgraph);

X = dlarray(rand(224,224,3,1,'single'),'SSCB');

F = forward(netFeatures,X);

Fdata = extractdata(F);

fprintf('\nOriginal feature statistics:\n');
printStats(Fdata);

%% 4. Explicitly restore the batch dimension

% The feature extractor returned [7 7 1280].
% Add the singleton batch dimension to make [7 7 1280 1].

if ndims(Fdata) == 3
    Fdata = reshape(Fdata,size(Fdata,1),size(Fdata,2), ...
        size(Fdata,3),1);
end

F = dlarray(Fdata,'SSCB');

fprintf('\nFormatted feature dimensions: %s\n', ...
    mat2str(size(Fdata)));

%% 5. Initialize adaptive attention

fprintf('\nInitializing adaptive attention...\n');

adaptiveLayer = AdaptiveAttentionLayer( ...
    16,7,'adaptive_attention');

layout = networkDataLayout([7 7 1280 1],'SSCB');

adaptiveLayer = initialize(adaptiveLayer,layout);

fprintf('Adaptive layer initialization: PASS\n');

%% 6. Test adaptive attention

fprintf('\nRunning adaptive attention...\n');

Z = forward(adaptiveLayer,F);

Zdata = extractdata(Z);

fprintf('\nAdaptive output statistics:\n');
printStats(Zdata);

%% 7. Validate output

assert(all(isfinite(Zdata),'all'), ...
    'Adaptive output contains NaN or Inf.');

assert(any(abs(Zdata(:)) > 1e-10), ...
    'Adaptive output is entirely zero.');

assert(size(Zdata,1) == 7 && ...
       size(Zdata,2) == 7 && ...
       size(Zdata,3) == 1280 && ...
       size(Zdata,4) == 1, ...
       'Unexpected adaptive output dimensions.');
fprintf('\nAdaptive layer preserves nonzero features: PASS\n');

fprintf('\n============================================\n');
fprintf('DIAGNOSTIC FINISHED\n');
fprintf('============================================\n');

%% Local helper

function printStats(A)

    fprintf('Dimensions: %s\n',mat2str(size(A)));
    fprintf('Minimum: %.8g\n',double(min(A,[],'all')));
    fprintf('Maximum: %.8g\n',double(max(A,[],'all')));
    fprintf('Mean: %.8g\n',double(mean(A,'all')));
    fprintf('Standard deviation: %.8g\n',double(std(A(:))));

end