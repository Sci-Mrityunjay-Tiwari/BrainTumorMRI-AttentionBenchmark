%% Inspect EfficientNet-B0 Feature Connections
clear;
clc;

netBase = imagePretrainedNetwork("efficientnetb0");
lgraph = layerGraph(netBase);

connections = lgraph.Connections;
layers = lgraph.Layers;

fprintf('\n=== FINAL CONVOLUTIONAL BLOCK CONNECTIONS ===\n');

patterns = {
    'efficientnet-b0|model|blocks_15|conv2d_1|Conv2D'
    'efficientnet-b0|model|blocks_15|se|MulLayer_1'
    'efficientnet-b0|model|head|conv2d|Conv2D'
    'efficientnet-b0|model|head|tpu_batch_normalization|FusedBatchNorm'
    'efficientnet-b0|model|head|SigmoidLayer'
    'efficientnet-b0|model|head|MulLayer'
    'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool'
    };

for p = 1:numel(patterns)
    name = patterns{p};

    fprintf('\n--- Layer: %s ---\n', name);

    idx = find(strcmp({layers.Name}, name), 1);

    if isempty(idx)
        fprintf('Layer not found.\n');
        continue;
    end

    disp(layers(idx));

    relevant = strcmp(connections.Source, name) | ...
        strcmp(connections.Destination, name);

    disp(connections(relevant,:));
end

fprintf('\n=== CONNECTIONS AROUND THE CLASSIFICATION HEAD ===\n');

disp(connections(end-12:end,:));