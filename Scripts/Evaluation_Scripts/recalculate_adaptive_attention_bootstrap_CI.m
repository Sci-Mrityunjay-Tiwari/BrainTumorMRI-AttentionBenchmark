%% Stratified Bootstrap Confidence Intervals
% Adaptive Attention - Internal Test Set
% 10,000 bootstrap replicates

clear;
clc;

%% 1. Load verified predictions

S = load('adaptive_attention_test_results.mat');

trueLabels = S.trueLabels;
predictedLabels = S.predictedLabels;
classNames = string(S.classNames);

N = numel(trueLabels);
numClasses = numel(classNames);
numBoot = 10000;

rng(42,'twister');

%% 2. Identify samples belonging to each true class

classIndices = cell(numClasses,1);

for k = 1:numClasses
    classIndices{k} = find(string(trueLabels) == classNames(k));
end

%% 3. Allocate bootstrap results

bootstrapAccuracy = zeros(numBoot,1);
bootstrapMacroF1 = zeros(numBoot,1);

%% 4. Perform stratified image-level bootstrap

for b = 1:numBoot

    sampledIndices = [];

    % Resample within each true class
    for k = 1:numClasses

        indices = classIndices{k};
        n = numel(indices);

        sampled = indices(randi(n,n,1));

        sampledIndices = [sampledIndices; sampled];

    end

    yTrue = trueLabels(sampledIndices);
    yPred = predictedLabels(sampledIndices);

    % Confusion matrix
    C = confusionmat(yTrue,yPred, ...
        'Order',categorical(classNames,classNames));

    % Accuracy
    bootstrapAccuracy(b) = sum(diag(C))/sum(C(:));

    % Macro-F1
    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2) - TP;

    precision = TP ./ (TP + FP);
    recall = TP ./ (TP + FN);

    F1 = 2 .* precision .* recall ./ ...
         (precision + recall);

    bootstrapMacroF1(b) = mean(F1);

end

%% 5. Calculate percentile confidence intervals

accuracyCI = prctile(bootstrapAccuracy,[2.5 97.5])*100;

macroF1CI = prctile(bootstrapMacroF1,[2.5 97.5])*100;

%% 6. Display results

fprintf('\nADAPTIVE ATTENTION - STRATIFIED BOOTSTRAP RESULTS\n');
fprintf('Test images: %d\n',N);
fprintf('Bootstrap replicates: %d\n',numBoot);

fprintf('\nAccuracy 95%% CI: %.2f - %.2f%%\n', ...
    accuracyCI(1),accuracyCI(2));

fprintf('Macro-F1 95%% CI: %.2f - %.2f%%\n', ...
    macroF1CI(1),macroF1CI(2));

%% 7. Save results

BootstrapResults = table( ...
    accuracyCI(1),accuracyCI(2), ...
    macroF1CI(1),macroF1CI(2), ...
    'VariableNames', ...
    {'Accuracy_CI_Lower','Accuracy_CI_Upper', ...
     'MacroF1_CI_Lower','MacroF1_CI_Upper'});

writetable(BootstrapResults, ...
    'AdaptiveAttention_Bootstrap_CI.csv');

save('AdaptiveAttention_Bootstrap_CI.mat', ...
    'bootstrapAccuracy','bootstrapMacroF1', ...
    'accuracyCI','macroF1CI');

disp('Bootstrap confidence intervals saved successfully.');