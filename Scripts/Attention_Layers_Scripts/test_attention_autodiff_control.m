%% Automatic Differentiation Control Test
% Tests the attention mathematics independently of the custom layer.

clear;
clc;

fprintf('=== Attention Autodiff Control Test ===\n');

C = 32;
Cr = 2;
H = 7;
W = 7;

X  = dlarray(randn(H,W,C,1,'single'), 'SSCB');
W1 = dlarray(randn(Cr,C,'single') * 0.01);
b1 = dlarray(zeros(Cr,1,'single'));
W2 = dlarray(randn(C,Cr,'single') * 0.01);
b2 = dlarray(zeros(C,1,'single'));

try
    [loss, gW1, gb1, gW2, gb2] = dlfeval( ...
        @controlGradient, X, W1, b1, W2, b2);

    fprintf('Loss: %g\n', double(extractdata(loss)));

    names = {'W1','b1','W2','b2'};
    grads = {gW1,gb1,gW2,gb2};

    for k = 1:numel(grads)
        g = extractdata(grads{k});

        fprintf('%s gradient norm: %g\n', ...
            names{k}, norm(g(:)));

        fprintf('%s gradient contains NaN: %d\n', ...
            names{k}, any(isnan(g(:))));
    end

catch ME
    fprintf('CONTROL TEST FAILED:\n%s\n', ME.message);
end

function [loss,gW1,gb1,gW2,gb2] = ...
    controlGradient(X,W1,b1,W2,b2)

avgF = squeeze(mean(X,[1 2]));
maxF = squeeze(max(X,[],[1 2]));

avgF = reshape(avgF,[],1);
maxF = reshape(maxF,[],1);

avgHidden = max(W1*avgF+b1,0);
maxHidden = max(W1*maxF+b1,0);

avgOut = W2*avgHidden+b2;
maxOut = W2*maxHidden+b2;

attention = sigmoid(avgOut+maxOut);
attention = reshape(attention,1,1,[],1);

Y = X .* attention;
loss = mean(Y.^2,'all');

[gW1,gb1,gW2,gb2] = dlgradient( ...
    loss,W1,b1,W2,b2);
end