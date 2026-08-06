fprintf('\n=== GoogLeNet Training ===\n\n');
runNumber = 1;   % Change only this line for each new experiment run
% Define number of classes
numClasses = numel(classNames);
fprintf('Classes: %d | Training samples: %d\n', ...
    numClasses, numel(imdsTrain.Files));

fprintf('Loading GoogLeNet...\n');

netBase = imagePretrainedNetwork("googlenet");
lgraph  = layerGraph(netBase);

%% ------------------------------------------------------------------------
% Remove original classification head
% ------------------------------------------------------------------------

allLayers = lgraph.Layers;
layersToRemove = {};

for i = length(allLayers):-1:1

    layer = allLayers(i);

    if isa(layer,'nnet.cnn.layer.ClassificationOutputLayer') || ...
       isa(layer,'nnet.cnn.layer.SoftmaxLayer') || ...
       isa(layer,'nnet.cnn.layer.FullyConnectedLayer') || ...
       isa(layer,'nnet.cnn.layer.GlobalAveragePooling2DLayer')

        layersToRemove = [layersToRemove layer.Name];

    else

        lastFeatLayer = layer.Name;
        break

    end

end

for i = 1:length(layersToRemove)

    try
        lgraph = removeLayers(lgraph,layersToRemove{i});
    catch
    end

end

fprintf('✓ Classification head removed\n');

%% ------------------------------------------------------------------------
% Add new classification head
% ------------------------------------------------------------------------

newHead = [

    globalAveragePooling2dLayer('Name','gap_new')

    fullyConnectedLayer(256,...
        'Name','fc_new',...
        'WeightLearnRateFactor',10,...
        'BiasLearnRateFactor',10)

    batchNormalizationLayer('Name','bn_new')

    reluLayer('Name','relu_new')

    dropoutLayer(0.5,'Name','drop_new')

    fullyConnectedLayer(numClasses,...
        'Name','fc_out',...
        'WeightLearnRateFactor',10,...
        'BiasLearnRateFactor',10)

    softmaxLayer('Name','softmax_new')

    classificationLayer('Name','output_new')

];

lgraph = addLayers(lgraph,newHead);
lgraph = connectLayers(lgraph,lastFeatLayer,'gap_new');

fprintf('✓ New classification head added\n\n');

%% ------------------------------------------------------------------------
% Training
% ------------------------------------------------------------------------

fprintf('=== Training GoogLeNet (10 epochs) ===\n');

opts = trainingOptions( ...
    'adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize',32, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',1e-4, ...
    'ValidationData',augVal, ...
    'ValidationFrequency',20, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'OutputNetwork','best-validation-loss', ...
    'Verbose',true);

fprintf('\nTraining started...\n');

tic;

net_ggl = trainNetwork(augTrain,lgraph,opts);

trainingTime = toc;

fprintf('\n✓ Training complete\n');
fprintf('Training Time : %.2f minutes\n\n',trainingTime/60);

%% ------------------------------------------------------------------------
% Evaluation
% ------------------------------------------------------------------------

fprintf('=== Evaluation ===\n');

trueLabels = imdsTest.Labels;

tic;

predLabels = classify(net_ggl,augTest);

inferenceTime = toc;

timePerImage = inferenceTime / numel(imdsTest.Files);

C = confusionmat(trueLabels,predLabels);

[acc,prec,rec,f1,spec] = computeMetrics(C);

fprintf('\n');
fprintf('Accuracy:      %.2f%%\n',acc*100);
fprintf('Precision:     %.2f%%\n',prec*100);
fprintf('Recall:        %.2f%%\n',rec*100);
fprintf('F1-Score:      %.2f%%\n',f1*100);
fprintf('Specificity:   %.2f%%\n',spec*100);

fprintf('Inference Time : %.4f sec\n',inferenceTime);
fprintf('Time / Image   : %.6f sec\n\n',timePerImage);
%% ------------------------------------------------------------------------
% Save results
% ------------------------------------------------------------------------

% Save evaluation metrics to CSV
saveResults( ...
    'GoogleNet', ...
    runNumber, ...
    acc, ...
    prec, ...
    rec, ...
    f1, ...
    spec, ...
    trainingTime, ...
    timePerImage);

%% ------------------------------------------------------------------------
% Save trained model
% ------------------------------------------------------------------------

if ~exist('TrainedModels','dir')
    mkdir('TrainedModels');
end

save( ...
    'TrainedModels/googlenet_braintumor.mat', ...
    'net_ggl', ...
    'classNames', ...
    'trueLabels', ...
    'predLabels', ...
    'C', ...
    'acc', ...
    'prec', ...
    'rec', ...
    'f1', ...
    'spec', ...
    'trainingTime', ...
    'inferenceTime', ...
    'timePerImage' ...
    );

fprintf('✓ TrainedModel saved: Models/googlenet_braintumor.mat\n');

%% ------------------------------------------------------------------------
% Training Summary
% ------------------------------------------------------------------------

fprintf('\nSaving evaluation results...\n');
fprintf('✓ Results saved to Results/all_results.csv\n');

fprintf('\n=============================================================\n');
fprintf('                GOOGLENET TRAINING COMPLETE\n');
fprintf('=============================================================\n');

fprintf('Accuracy            : %.2f%%\n', acc*100);
fprintf('Precision           : %.2f%%\n', prec*100);
fprintf('Recall              : %.2f%%\n', rec*100);
fprintf('F1-Score            : %.2f%%\n', f1*100);
fprintf('Specificity         : %.2f%%\n', spec*100);

fprintf('\nTraining Time       : %.2f minutes\n', trainingTime/60);
fprintf('Inference Time      : %.4f seconds\n', inferenceTime);
fprintf('Inference / Image   : %.6f seconds\n', timePerImage);

fprintf('\nResults CSV         : Results/all_results.csv\n');
fprintf('Model File          : TrainedModels/googlenet_braintumor.mat\n');

fprintf('\n✓✓✓ GOOGLENET COMPLETE ✓✓✓\n\n');