%% Diagnose EfficientNet-B0 Output
% MATLAB R2026a
% Inspect the pretrained network's output and available layer names.

clear;
clc;

fprintf('\n============================================\n');
fprintf('FEATURE-FLOW DIAGNOSTIC\n');
fprintf('============================================\n');

%% 1. Load EfficientNet-B0

fprintf('\nLoading pretrained EfficientNet-B0...\n');

netBase = imagePretrainedNetwork("efficientnetb0");

fprintf('Network class: %s\n',class(netBase));

%% 2. Inspect the network output

X = rand(224,224,3,1,'single');

X = dlarray(X,'SSCB');

fprintf('\nTesting original network prediction...\n');

try
    Y = predict(netBase,X);

    Ydata = extractdata(Y);

    fprintf('Output dimensions: %s\n',mat2str(size(Ydata)));
    fprintf('Output minimum: %.8g\n',double(min(Ydata,[],'all')));
    fprintf('Output maximum: %.8g\n',double(max(Ydata,[],'all')));
    fprintf('Output mean: %.8g\n',double(mean(Ydata,'all')));
    fprintf('Output standard deviation: %.8g\n', ...
        double(std(Ydata(:))));

catch ME
    fprintf('Prediction failed:\n%s\n',ME.message);
end

%% 3. Inspect network layers

fprintf('\nInspecting final network layers...\n');

layers = netBase.Layers;

for i = max(1,numel(layers)-12):numel(layers)

    fprintf('%3d | %-60s | %s\n', ...
        i,layers(i).Name,class(layers(i)));

end

%% 4. Inspect the original feature endpoint in the graph

fprintf('\nInspecting feature endpoint connections...\n');

lgraph = layerGraph(netBase);

endpoint = 'efficientnet-b0|model|head|MulLayer';

disp(lgraph.Connections( ...
    strcmp(lgraph.Connections.Source,endpoint) | ...
    strcmp(lgraph.Connections.Destination,endpoint),:));

fprintf('\n============================================\n');
fprintf('DIAGNOSTIC FINISHED\n');
fprintf('============================================\n');