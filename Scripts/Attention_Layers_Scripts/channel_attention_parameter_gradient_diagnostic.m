%% Channel Attention Parameter Gradient Diagnostic
% Diagnostic only. Does not modify the original layer.

clear;
clc;

fprintf('=== Parameter Gradient Diagnostic ===\n');

featureSize = [7 7 1280 1];

layer = ChannelAttentionLayer(16, 'channel_attention_test');
layout = networkDataLayout(featureSize, 'SSCB');
layer = initialize(layer, layout);

X = dlarray(randn(featureSize, 'single'), 'SSCB');

try
    [loss, gW1, gb1, gW2, gb2] = dlfeval( ...
        @parameterGradientTest, layer, X);

    fprintf('Loss: %g\n', double(extractdata(loss)));

    names = {'W1', 'b1', 'W2', 'b2'};
    grads = {gW1, gb1, gW2, gb2};

    for k = 1:numel(grads)
        fprintf('\n%s:\n', names{k});

        if isempty(grads{k})
            fprintf('  Gradient is EMPTY.\n');
            continue;
        end

        g = extractdata(grads{k});

        fprintf('  Size: ');
        disp(size(g));

        fprintf('  Norm: %g\n', norm(g(:)));
        fprintf('  Contains NaN: %d\n', any(isnan(g(:))));
        fprintf('  Contains Inf: %d\n', any(isinf(g(:))));
    end

catch ME
    fprintf('\nPARAMETER GRADIENT TEST FAILED:\n%s\n', ME.message);

    for k = 1:numel(ME.stack)
        fprintf('  %s, line %d\n', ...
            ME.stack(k).name, ME.stack(k).line);
    end
end

function [loss, gW1, gb1, gW2, gb2] = ...
    parameterGradientTest(layer, X)

Y = forward(layer, X);
loss = mean(Y.^2, 'all');

[gW1, gb1, gW2, gb2] = dlgradient( ...
    loss, layer.W1, layer.b1, layer.W2, layer.b2);
end