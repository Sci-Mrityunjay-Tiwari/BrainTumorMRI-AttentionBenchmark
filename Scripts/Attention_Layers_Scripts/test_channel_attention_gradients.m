%% Brain Tumour Research: Channel Attention Diagnostic
% Preliminary test for MATLAB R2026a.
% Does not modify the original research files.

clear;
clc;

fprintf('=== Channel Attention Diagnostic ===\n');
fprintf('MATLAB version: %s\n\n', version);

%% 1. Confirm the custom layer is available

assert(exist('ChannelAttentionLayer', 'class') == 8, ...
    'ChannelAttentionLayer.m was not found on the MATLAB path.');

%% 2. Create the custom layer

layer = ChannelAttentionLayer(16, 'channel_attention_test');

fprintf('Layer created successfully.\n');

%% 3. Create the correct network data layout

% EfficientNetB0 final feature-map dimensions:
% Height = 7, Width = 7, Channels = 1280, Batch = 1

featureSize = [7 7 1280 1];

try
    layout = networkDataLayout(featureSize, 'SSCB');

    fprintf('Network data layout created successfully.\n');
    fprintf('Layout size: %s\n', mat2str(layout.Size));
    fprintf('Layout format: %s\n', layout.Format);

catch ME
    fprintf('\nLAYOUT CREATION FAILED:\n%s\n', ME.message);
    return;
end

%% 4. Initialize the layer

try
    layer = initialize(layer, layout);

    fprintf('\nLayer initialized successfully.\n');

catch ME
    fprintf('\nINITIALIZATION FAILED:\n%s\n', ME.message);
    disp(ME.stack);
    return;
end

%% 5. Inspect learnable parameters

fprintf('\nLearnable parameter dimensions:\n');

fprintf('W1: ');
disp(size(layer.W1));

fprintf('b1: ');
disp(size(layer.b1));

fprintf('W2: ');
disp(size(layer.W2));

fprintf('b2: ');
disp(size(layer.b2));

%% 6. Perform a forward pass

try
    X = dlarray(randn(featureSize, 'single'), 'SSCB');

    Y = predict(layer, X);

    fprintf('\nForward pass completed successfully.\n');
    fprintf('Input size:  %s\n', mat2str(size(X)));
    fprintf('Output size: %s\n', mat2str(size(Y)));

    yData = extractdata(Y);

    fprintf('Contains NaN: %d\n', any(isnan(yData(:))));
    fprintf('Contains Inf: %d\n', any(isinf(yData(:))));

catch ME
    fprintf('\nFORWARD PASS FAILED:\n%s\n', ME.message);
    disp(ME.stack);
    return;
end

fprintf('\nPreliminary diagnostic completed.\n');
fprintf('Gradient propagation has NOT yet been tested.\n');