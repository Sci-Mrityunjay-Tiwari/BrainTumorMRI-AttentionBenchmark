%% Adaptive Attention - Publication Confusion Matrix
% Source: Verified saved evaluation MAT file
% Test set: 489 images
% Accuracy: 95.09%

clc;

%% 1. Load verified evaluation results

resultPath = fullfile('EvaluationResults', ...
    'adaptive_attention_test_results.mat');

S = load(resultPath, 'C', 'classNames', 'accuracy');

C = S.C;

%% 2. Validate source data

expectedMatrix = [
    131 7  1  0;
    9   128 0  3;
    2   0  72 1;
    0   1  0  134
    ];

assert(isequal(C, expectedMatrix), ...
    'Confusion matrix does not match verified results.');

assert(sum(C(:)) == 489, ...
    'Unexpected test-set size.');

correctPredictions = sum(diag(C));
calculatedAccuracy = correctPredictions / sum(C(:));

assert(abs(calculatedAccuracy - S.accuracy) < 1e-10, ...
    'Saved accuracy does not match confusion matrix.');

%% 3. Define publication display labels

displayNames = {
    'Glioma'
    'Meningioma'
    'No tumour'
    'Pituitary'
    };

%% 4. Generate confusion matrix

fig = figure( ...
    'Color', 'w', ...
    'Position', [100 100 950 800]);

cm = confusionchart(C, displayNames);

cm.Title = sprintf( ...
    'EfficientNet-B0 with Adaptive Attention: Internal Test Set');

cm.XLabel = 'Predicted Class';
cm.YLabel = 'True Class';

cm.FontName = 'Times New Roman';
cm.FontSize = 16;

cm.Normalization = 'absolute';

%% 5. Export figure

outputPath = fullfile('EvaluationResults', ...
    'AdaptiveAttention_ConfusionMatrix_Verified.png');

exportgraphics(fig, outputPath, 'Resolution', 600);

%% 6. Display verification

fprintf('\n===== FIGURE VERIFICATION =====\n');

fprintf('Correct predictions: %d\n', correctPredictions);
fprintf('Incorrect predictions: %d\n', ...
    sum(C(:)) - correctPredictions);

fprintf('Accuracy: %.4f%%\n', calculatedAccuracy * 100);

fprintf('Total images: %d\n', sum(C(:)));

fprintf('Figure saved to:\n%s\n', outputPath);

fprintf('\n===== COMPLETE =====\n');