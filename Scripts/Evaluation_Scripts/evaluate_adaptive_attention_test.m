%% Adaptive Attention - Test Set Evaluation

clc;

fprintf('\n============================================\n');
fprintf(' ADAPTIVE ATTENTION: TEST SET EVALUATION\n');
fprintf('============================================\n');

%% 1. Verify variables

assert(exist('netAdaptiveTrained','var') == 1);
assert(exist('augTest','var') == 1);
assert(exist('classNames','var') == 1);

numClasses = numel(classNames);

fprintf('Test images: %d\n',numel(imdsTest.Files));
fprintf('Classes: %d\n',numClasses);

%% 2. Prepare test datastore

reset(augTest);

trueLabels = categorical();
predictedLabels = categorical();

allScores = [];

%% 3. Run predictions

fprintf('\nEvaluating test images...\n');

while hasdata(augTest)

    batch = read(augTest);

    X = cat(4,batch.input{:});

    dlX = dlarray(single(X),'SSCB');

    if canUseGPU
        dlX = gpuArray(dlX);
    end

    Y = predict(netAdaptiveTrained,dlX);

    scores = gather(extractdata(Y));

    [~,predIdx] = max(scores,[],1);

    batchPredictions = categorical( ...
        predIdx(:),1:numClasses,classNames);

    trueLabels = [trueLabels; batch.response(:)];

    predictedLabels = [predictedLabels; batchPredictions];

    allScores = [allScores, scores];

end

%% 4. Calculate confusion matrix

classOrder = categorical(classNames,classNames);

C = confusionmat( ...
    trueLabels,predictedLabels, ...
    'Order',classOrder);

fprintf('\n--- CONFUSION MATRIX ---\n');

disp(array2table(C, ...
    'VariableNames',matlab.lang.makeValidName(classNames), ...
    'RowNames',matlab.lang.makeValidName(classNames)));

%% 5. Calculate classification metrics

TP = diag(C);

FP = sum(C,1)' - TP;

FN = sum(C,2) - TP;

TN = sum(C,'all') - TP - FP - FN;

precision = TP ./ max(TP + FP,1);

recall = TP ./ max(TP + FN,1);

f1Score = 2 .* precision .* recall ./ ...
    max(precision + recall,eps);

support = sum(C,2);

accuracy = sum(TP) / sum(C,'all');

macroPrecision = mean(precision);

macroRecall = mean(recall);

macroF1 = mean(f1Score);

weightedF1 = sum(f1Score .* support) / sum(support);

specificity = TN ./ max(TN + FP,1);

%% 6. Display results

resultsTable = table( ...
    classNames(:), ...
    precision, ...
    recall, ...
    specificity, ...
    f1Score, ...
    support, ...
    'VariableNames', ...
    {'Class','Precision','Recall','Specificity','F1Score','Support'});

fprintf('\n--- PER-CLASS METRICS ---\n');

disp(resultsTable);

fprintf('\n--- OVERALL PERFORMANCE ---\n');

fprintf('Test Accuracy:       %.4f (%.2f%%)\n',accuracy,accuracy*100);

fprintf('Macro Precision:     %.4f\n',macroPrecision);

fprintf('Macro Recall:        %.4f\n',macroRecall);

fprintf('Macro F1-score:      %.4f\n',macroF1);

fprintf('Weighted F1-score:   %.4f\n',weightedF1);

%% 7. Plot confusion matrix

figure('Name','Adaptive Attention - Test Confusion Matrix');

confusionchart(C,classNames);

title('EfficientNet-B0 + Adaptive Attention: Test Set');

%% 8. Save evaluation results

if ~exist('EvaluationResults','dir')
    mkdir('EvaluationResults');
end

save(fullfile('EvaluationResults', ...
    'adaptive_attention_test_results.mat'), ...
    'C','resultsTable','accuracy', ...
    'macroPrecision','macroRecall', ...
    'macroF1','weightedF1','specificity', ...
    'trueLabels','predictedLabels','allScores', ...
    'classNames');

writetable(resultsTable, ...
    fullfile('EvaluationResults', ...
    'adaptive_attention_classification_metrics.csv'));

fprintf('\nTest evaluation completed and results saved.\n');