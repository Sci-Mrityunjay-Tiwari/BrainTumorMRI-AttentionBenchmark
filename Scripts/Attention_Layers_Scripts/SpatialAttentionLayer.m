classdef SpatialAttentionLayer < nnet.layer.Layer & nnet.layer.Formattable
    % =========================================================================
    % SpatialAttentionLayer
    %
    % Spatial Attention Module from:
    %
    % Woo et al., "CBAM: Convolutional Block Attention Module",
    % ECCV 2018.
    %
    % Mathematical formulation:
    %
    % Ms(F) = sigmoid(f7×7([AvgPool(F); MaxPool(F)]))
    %
    % where:
    %   AvgPool(F) : Channel-wise average pooling
    %   MaxPool(F) : Channel-wise max pooling
    %   f7×7       : Convolution with a 7×7 kernel
    %
    % This layer produces a spatial attention map that is multiplied with
    % the input feature maps.
    % =========================================================================

    %% --------------------------------------------------------------------
    % Layer Properties
    % ---------------------------------------------------------------------

    properties

        % Spatial convolution kernel size
        KernelSize

    end

    %% --------------------------------------------------------------------
    % Learnable Parameters
    % ---------------------------------------------------------------------

    properties (Learnable)

        % Convolution kernel
        W

        % Bias
        b

    end

    %% --------------------------------------------------------------------
    % Constructor & Initialization
    % ---------------------------------------------------------------------

    methods

        function layer = SpatialAttentionLayer(k, name)

            % Default kernel size
            if nargin < 1
                k = 7;
            end

            % Default layer name
            if nargin < 2
                name = "spatial_attention";
            end

            layer.KernelSize = k;
            layer.Name = name;

            layer.Description = sprintf( ...
                "CBAM Spatial Attention (Kernel = %d×%d)", k, k);

        end

        %--------------------------------------------------------------
        % Initialize learnable parameters
        %--------------------------------------------------------------
        function layer = initialize(layer, layout)

            %#ok<INUSD>
            % layout is unused but retained for compatibility with the
            % custom layer initialization interface.

            k = layer.KernelSize;

            % Convolution kernel
            layer.W = dlarray( ...
                randn(k, k, 2, 1, 'single') * 0.01);

            % Bias
            layer.b = dlarray( ...
                zeros(1,1,1,1,'single'));

        end
        %--------------------------------------------------------------
        % Forward Prediction
        %--------------------------------------------------------------
        function Z = predict(layer, X)

            %==========================================================
            % Channel-wise Average Pooling
            %==========================================================
            avgC = mean(X,3);

            %==========================================================
            % Channel-wise Max Pooling
            %==========================================================
            maxC = max(X,[],3);

            %==========================================================
            % Concatenate pooled descriptors
            %
            % Resulting tensor:
            %
            %   Height × Width × 2 × Batch
            %
            % Channel 1 : Average-pooled descriptor
            % Channel 2 : Max-pooled descriptor
            %==========================================================
            descriptors = cat(3, avgC, maxC);

            %==========================================================
            % Spatial Convolution
            %
            % Apply a k×k convolution to generate the spatial
            % attention map.
            %==========================================================
            padding = floor(layer.KernelSize / 2);

            attentionMap = dlconv( ...
                descriptors, ...
                layer.W, ...
                layer.b, ...
                'Padding', padding);

            %==========================================================
            % Sigmoid Activation
            %
            % Produces spatial attention weights in the range [0,1].
            %==========================================================
            spatialAttention = ...
                1 ./ (1 + exp(-attentionMap));

            %==========================================================
            % Apply Spatial Attention
            %
            % The attention map is automatically broadcast across
            % all feature channels.
            %==========================================================
            Z = X .* spatialAttention;

        end

    end

end