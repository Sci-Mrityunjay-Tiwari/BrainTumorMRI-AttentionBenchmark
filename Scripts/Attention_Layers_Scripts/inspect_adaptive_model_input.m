%% INSPECT MODEL INPUT REQUIREMENTS

modelFile = fullfile(pwd, ...
    'TrainedModels', ...
    'efficientnet_adaptive_attention_softmax.mat');

S = load(modelFile);

netAdaptiveTrained = S.netAdaptiveTrained;
classNames = S.classNames;

% Find image input layer
inputLayer = netAdaptiveTrained.Layers(1);

fprintf('\nMODEL INPUT LAYER\n');
fprintf('-----------------------------\n');

disp(inputLayer);

fprintf('\nINPUT SIZE:\n');
disp(inputLayer.InputSize);

fprintf('\nNORMALIZATION:\n');
disp(inputLayer.Normalization);

fprintf('\nNORMALIZATION STATISTICS:\n');

if isprop(inputLayer,'Mean')
    disp(inputLayer.Mean);
end

if isprop(inputLayer,'StandardDeviation')
    disp(inputLayer.StandardDeviation);
end

fprintf('\nCLASS ORDER:\n');
disp(classNames);