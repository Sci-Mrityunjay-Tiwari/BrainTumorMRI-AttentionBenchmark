function runChannelAttentionGradientTest()
% Tests input and learnable-parameter gradients for ChannelAttentionLayer.
% Does not modify trained models.

clc;
fprintf('=== Channel Attention Gradient Test ===\n');

featureSize = [7 7 1280 1];

layer = ChannelAttentionLayer(16, 'channel_attention_test');
layout = networkDataLayout(featureSize, 'SSCB');
layer = initialize(layer, layout);

X = dlarray(randn(featureSize, 'single'), 'SSCB');

try
    [loss, gX, gW1, gb1, gW2, gb2] = ...
        dlfeval(@gradientTest, layer, X);

    fprintf('Loss: %g\n', double(extractdata(loss)));

    names = {'Input', 'W1', 'b1', 'W2', 'b2'};
    grads = {gX, gW1, gb1, gW2, gb2};

    for k = 1:numel(grads)
        fprintf('\n%s gradient:\n', names{k});

        if isempty(grads{k})
            fprintf('  EMPTY gradient\n');
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
    fprintf('\nGRADIENT TEST FAILED:\n%s\n', ME.message);

    for k = 1:numel(ME.stack)
        fprintf('  %s, line %d\n', ...
            ME.stack(k).name, ME.stack(k).line);
    end
end

end


function [loss, gX, gW1, gb1, gW2, gb2] = ...
    gradientTest(layer, X)

Y = forward(layer, X);

% Scalar objective used only for this diagnostic.
loss = mean(Y.^2, 'all');

% Calculate input gradient.
gX = dlgradient(loss, X);

% Calculate learnable-parameter gradients.
[gW1, gb1, gW2, gb2] = dlgradient( ...
    loss, layer.W1, layer.b1, layer.W2, layer.b2);

end