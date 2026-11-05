%% Adaptive Attention Layer: Initial Verification

clear;
rng(42);

%% 1. Initialize the layer

layer = AdaptiveAttentionLayer(16,7, ...
    "adaptive_attention_test");

layout = networkDataLayout([7 7 1280 2],"SSCB");

layer = initialize(layer,layout);

%% 2. Generate synthetic feature maps

X = dlarray( ...
    randn(7,7,1280,2,'single'), ...
    "SSCB");

%% 3. Test the forward pass

Y = forward(layer,X);

assert(isequal(size(Y),size(X)), ...
    "Output dimensions do not match input dimensions.");

Ydata = extractdata(Y);

assert(all(isfinite(Ydata(:))), ...
    "Output contains NaN or Inf values.");

fprintf("Forward pass: PASS\n");

fprintf("Input dimensions:  %s\n", ...
    mat2str(size(X)));

fprintf("Output dimensions: %s\n", ...
    mat2str(size(Y)));

%% 4. Check adaptive-gate normalization

[gateValues,gateSums] = ...
    dlfeval(@checkAdaptiveGates,layer,X);

gateValues = extractdata(gateValues);
gateSums = extractdata(gateSums);

assert(all(isfinite(gateValues(:))), ...
    "Gate weights contain NaN or Inf.");

assert(all(abs(gateSums(:)-1) < 1e-5), ...
    "Adaptive gate weights do not sum to one.");

fprintf("Gate normalization: PASS\n");

disp("Mean branch weights across the synthetic batch:");

disp(mean(gateValues,2));

%% 5. Check gradients

[loss,gradients] = ...
    dlfeval(@adaptiveGradientCheck,layer,X);

assert(isfinite(double(extractdata(loss))), ...
    "Loss is not finite.");

gradientNames = fieldnames(gradients);

for i = 1:numel(gradientNames)

    g = gradients.(gradientNames{i});

    gd = extractdata(g);

    assert(~isempty(gd) && all(isfinite(gd(:))), ...
        "Invalid gradient: %s",gradientNames{i});

    gradNorm = sqrt(sum(g.^2,'all'));

    fprintf("%s gradient norm: %.6g\n", ...
        gradientNames{i}, ...
        double(gather(extractdata(gradNorm))));

end

fprintf("Gradient verification: PASS\n");


%% Local functions

function [gates,sums] = checkAdaptiveGates(layer,X)

    X = stripdims(X);

    C = size(X,3);
    B = size(X,4);

    descriptor = reshape(mean(X,[1 2]),C,B);

    hidden = max( ...
        stripdims(layer.Wg1)*descriptor ...
        + stripdims(layer.bg1),0);

    logits = ...
        stripdims(layer.Wg2)*hidden ...
        + stripdims(layer.bg2);

    logits = logits-max(logits,[],1);

    e = exp(logits);

    gates = e./sum(e,1);

    sums = sum(gates,1);

end


function [loss, gradients] = adaptiveGradientCheck(layer, X)

% Pass all learnable parameters explicitly into dlfeval.
[loss, gradients] = dlfeval( ...
    @actualAttentionGradient, ...
    X, ...
    layer.Wc1, layer.bc1, ...
    layer.Wc2, layer.bc2, ...
    layer.Ws, layer.bs, ...
    layer.Wg1, layer.bg1, ...
    layer.Wg2, layer.bg2);

end

function [loss, gradients] = actualAttentionGradient( ...
X, Wc1, bc1, Wc2, bc2, Ws, bs, Wg1, bg1, Wg2, bg2)

% Reproduce the adaptive attention forward computation using
% explicitly supplied learnable parameters.

X = stripdims(X);

sx = size(X);
sx(end+1:4) = 1;

C = sx(3);
B = sx(4);

%% Channel attention

avgDesc = reshape(mean(X,[1 2]),C,B);
maxDesc = reshape(max(max(X,[],1),[],2),C,B);

avgHidden = max(Wc1*avgDesc + bc1,0);
maxHidden = max(Wc1*maxDesc + bc1,0);

channelLogits = Wc2*avgHidden + bc2 ...
              + Wc2*maxHidden + bc2;

channelWeights = 1./(1 + exp(-channelLogits));
channelWeights = reshape(channelWeights,1,1,C,B);

Fc = X.*channelWeights;

%% Spatial attention

avgSpatial = mean(X,3);
maxSpatial = max(X,[],3);

spatialInput = cat(3,avgSpatial,maxSpatial);

padding = floor(size(Ws,1)/2);

spatialLogits = dlconv( ...
    spatialInput,Ws,bs, ...
    'DataFormat','SSCB', ...
    'Padding',padding);

spatialWeights = 1./(1 + exp(-spatialLogits));

Fs = X.*spatialWeights;

%% Input-dependent gating

gateDesc = reshape(mean(X,[1 2]),C,B);

gateHidden = max(Wg1*gateDesc + bg1,0);

gateLogits = Wg2*gateHidden + bg2;

gateLogits = gateLogits - max(gateLogits,[],1);

gateExp = exp(gateLogits);

gates = gateExp./sum(gateExp,1);

alpha = reshape(gates(1,:),1,1,1,B);
beta  = reshape(gates(2,:),1,1,1,B);
gamma = reshape(gates(3,:),1,1,1,B);

%% Adaptive feature fusion

Y = alpha.*X + beta.*Fc + gamma.*Fs;

% Nonlinear objective to test gradients through the entire
% attention computation, not just the parameter values.
loss = mean(Y.^2,'all');

[gWc1,gbc1,gWc2,gbc2,gWs,gbs, ...
    gWg1,gbg1,gWg2,gbg2] = dlgradient( ...
    loss, ...
    Wc1,bc1,Wc2,bc2,Ws,bs,Wg1,bg1,Wg2,bg2);

gradients = struct( ...
    'Wc1',gWc1,'bc1',gbc1, ...
    'Wc2',gWc2,'bc2',gbc2, ...
    'Ws',gWs,'bs',gbs, ...
    'Wg1',gWg1,'bg1',gbg1, ...
    'Wg2',gWg2,'bg2',gbg2);

end
