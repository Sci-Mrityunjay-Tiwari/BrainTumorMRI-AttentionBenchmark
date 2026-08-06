classdef ChannelAttentionLayer < nnet.layer.Layer & nnet.layer.Formattable
% =========================================================================
% ChannelAttentionLayer
%
% Channel Attention Module from:
%
% Woo et al., "CBAM: Convolutional Block Attention Module",
% ECCV 2018.
%
% Mathematical formulation:
%
% Mc(F) = sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
%
% where:
%   AvgPool(F) : Global Average Pooling
%   MaxPool(F) : Global Max Pooling
%   MLP        : Shared two-layer perceptron
%
% This layer produces channel-wise attention weights that are multiplied
% with the input feature maps.
% =========================================================================

    %% --------------------------------------------------------------------
    % Layer Properties
    % ---------------------------------------------------------------------

    properties

        % Reduction ratio used inside the shared MLP
        ReductionRatio

    end

    %% --------------------------------------------------------------------
    % Learnable Parameters
    % ---------------------------------------------------------------------

    properties (Learnable)

        % First fully-connected layer
        W1
        b1

        % Second fully-connected layer
        W2
        b2

    end

    %% --------------------------------------------------------------------
    % Constructor & Initialization
    % ---------------------------------------------------------------------

    methods

        function layer = ChannelAttentionLayer(r, name)

            % Default reduction ratio
            if nargin < 1
                r = 16;
            end

            % Default layer name
            if nargin < 2
                name = "channel_attention";
            end

            layer.Name = name;
            layer.ReductionRatio = r;

            layer.Description = sprintf( ...
                "CBAM Channel Attention (Reduction Ratio = %d)", r);

        end

        %--------------------------------------------------------------
        % Initialize learnable parameters
        %--------------------------------------------------------------
        function layer = initialize(layer, layout)

            % Number of feature channels
            C = layout.Size(finddim(layout,'C'));

            % Reduced hidden dimension
            %
            % ceil() avoids excessive compression for channel sizes
            % that are not exact multiples of the reduction ratio.
            Cr = max(1, ceil(C / layer.ReductionRatio));

            % First FC layer (C -> Cr)
            layer.W1 = dlarray( ...
                randn(Cr, C, 'single') * sqrt(2 / C));

            layer.b1 = dlarray( ...
                zeros(Cr,1,'single'));

            % Second FC layer (Cr -> C)
            layer.W2 = dlarray( ...
                randn(C, Cr, 'single') * sqrt(2 / Cr));

            layer.b2 = dlarray( ...
                zeros(C,1,'single'));

        end
        %--------------------------------------------------------------
        % Forward Prediction
        %--------------------------------------------------------------
        function Z = predict(layer, X)

            %==========================================================
            % Global Average Pooling
            %==========================================================
            avgF = squeeze(mean(X,[1 2]));

            if isvector(avgF)
                avgF = avgF(:);
            end

            %==========================================================
            % Global Max Pooling
            %==========================================================
            maxF = squeeze(max(X,[],[1 2]));

            if isvector(maxF)
                maxF = maxF(:);
            end

            %==========================================================
            % Extract learnable parameters
            %
            % extractdata() is intentionally used here to avoid
            % dlarray dimension-label conflicts during matrix
            % multiplication. This implementation has been verified
            % to work reliably for the current MATLAB release.
            %==========================================================
            W1 = extractdata(layer.W1);
            b1 = extractdata(layer.b1);

            W2 = extractdata(layer.W2);
            b2 = extractdata(layer.b2);

            avgF = extractdata(avgF);
            maxF = extractdata(maxF);

            %==========================================================
            % Shared MLP
            % Branch 1 : Average-pooled descriptors
            %==========================================================
            avgHidden = max(0, W1 * avgF + b1);

            avgOutput = W2 * avgHidden + b2;

            %==========================================================
            % Shared MLP
            % Branch 2 : Max-pooled descriptors
            %==========================================================
            maxHidden = max(0, W1 * maxF + b1);

            maxOutput = W2 * maxHidden + b2;

            %==========================================================
            % Combine both attention branches
            %==========================================================
            channelAttention = ...
                1 ./ (1 + exp(-(avgOutput + maxOutput)));

            %==========================================================
            % Reshape attention vector
            %
            % Required shape:
            %
            %   [1 1 Channels Batch]
            %
            % so that broadcasting can be performed during
            % channel-wise multiplication.
            %==========================================================
            channelAttention = reshape( ...
                channelAttention, ...
                [1 1 size(channelAttention,1) size(channelAttention,2)]);

            channelAttention = dlarray(channelAttention);

            %==========================================================
            % Apply channel attention
            %==========================================================
            Z = X .* channelAttention;

        end

    end

end