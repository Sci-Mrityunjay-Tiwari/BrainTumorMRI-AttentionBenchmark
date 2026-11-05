%% EFFICIENTNET-B0 BASELINE AUDIT

clear;
clc;

%% Load checkpoint

S = load('TrainedModels/efficientnetb0_braintumor.mat');

fprintf('\n===== CHECKPOINT VARIABLES =====\n');
whos('-file','TrainedModels/efficientnetb0_braintumor.mat');

%% Display saved metrics

fprintf('\n===== SAVED BASELINE METRICS =====\n');

fprintf('Accuracy: %.6f\n',S.acc);
fprintf('Precision: %.6f\n',S.prec);
fprintf('Recall: %.6f\n',S.rec);
fprintf('F1: %.6f\n',S.f1);

%% Load consolidated predictions

T = readtable( ...
    'EvaluationResults/Five_Model_ImageLevel_Predictions.csv');

trueLabels = categorical(T.TrueLabel);
predLabels = categorical(T.EfficientNetB0);

fprintf('\n===== RECOMPUTED BASELINE PERFORMANCE =====\n');

fprintf('Correct: %d\n',sum(trueLabels == predLabels));
fprintf('Incorrect: %d\n',sum(trueLabels ~= predLabels));
fprintf('Accuracy: %.6f\n',mean(trueLabels == predLabels));

%% Compare saved confusion matrix

fprintf('\n===== SAVED CONFUSION MATRIX =====\n');
disp(S.C);

fprintf('\n===== RECOMPUTED CONFUSION MATRIX =====\n');

classNames = categories(trueLabels);

C = confusionmat(trueLabels,predLabels, ...
    'Order',categorical(classNames,classNames));

disp(C);

%% Compare saved predictions

fprintf('\n===== PREDICTION CONSISTENCY =====\n');

fprintf('Saved predictions match consolidated predictions: %d\n', ...
    isequal(S.predLabels,predLabels));

fprintf('Saved true labels match consolidated labels: %d\n', ...
    isequal(S.trueLabels,trueLabels));