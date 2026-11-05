%% EfficientNet-B0 Intermediate Feature Diagnostic
% MATLAB R2026a
% No training and no model files are saved.

clear;
clc;

fprintf('\n============================================\n');
fprintf('INTERMEDIATE FEATURE DIAGNOSTIC\n');
fprintf('============================================\n');

%% 1. Load EfficientNet-B0

netBase = imagePretrainedNetwork("efficientnetb0");
lgraph = layerGraph(netBase);

%% 2. Define the spatial feature endpoint

endpoint = 'efficientnet-b0|model|head|MulLayer';

%% 3. Remove the original classification tail

tailLayers = {
    'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool'
    'efficientnet-b0|model|head|dense|MatMul'
    'Softmax'
    };

for i = 1:numel(tailLayers)

    names = string({lgraph.Layers.Name});

    if any(names == tailLayers{i})
        lgraph = removeLayers(lgraph,tailLayers{i});
    end
end

%% 4. Build a feature-extraction network

fprintf('\nConstructing feature extractor...\n');

netFeatures = dlnetwork(lgraph);

fprintf('Feature extractor constructed.\n');

%% 5. Test the feature extractor

X = dlarray(rand(224,224,3,1,'single'),'SSCB');

fprintf('\nRunning feature extractor...\n');

try

    F = forward(netFeatures,X);

    Fdata = extractdata(F);

    fprintf('Feature dimensions: %s\n',mat2str(size(Fdata)));
    fprintf('Minimum: %.8g\n',double(min(Fdata,[],'all')));
    fprintf('Maximum: %.8g\n',double(max(Fdata,[],'all')));
    fprintf('Mean: %.8g\n',double(mean(Fdata,'all')));
    fprintf('Standard deviation: %.8g\n',double(std(Fdata(:))));

catch ME

    fprintf('\nFeature extraction failed:\n');
    fprintf('%s\n',ME.message);

end

fprintf('\n============================================\n');
fprintf('DIAGNOSTIC FINISHED\n');
fprintf('============================================\n');