%% Inspect EfficientNet-B0 feature endpoint
clear;
clc;

netBase = imagePretrainedNetwork("efficientnetb0");
lgraph = layerGraph(netBase);

fprintf('\n=== NETWORK LAYERS NEAR THE OUTPUT ===\n');

layers = lgraph.Layers;

for i = max(1,numel(layers)-12):numel(layers)
    fprintf('%3d | %-55s | %s\n', ...
        i, layers(i).Name, class(layers(i)));
end

fprintf('\n=== CONNECTIONS TO AND FROM THE ENDPOINT ===\n');

connections = lgraph.Connections;

endpoint = 'efficientnet-b0|model|head|MulLayer';

disp(connections( ...
    strcmp(connections.Source,endpoint) | ...
    strcmp(connections.Destination,endpoint),:));

fprintf('\n=== ENDPOINT DETAILS ===\n');

idx = find(strcmp({layers.Name},endpoint),1);

if isempty(idx)
    error('Endpoint not found in the current network.');
end

disp(layers(idx));

fprintf('\n=== NETWORK INPUT LAYERS ===\n');

for i = 1:numel(layers)
    if isa(layers(i),'nnet.cnn.layer.ImageInputLayer')
        disp(layers(i));
    end
end