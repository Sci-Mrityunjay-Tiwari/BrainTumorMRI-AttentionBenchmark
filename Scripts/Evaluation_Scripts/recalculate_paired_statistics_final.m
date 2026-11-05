%% FINAL PAIRED STATISTICAL ANALYSIS
% Recalculates comparisons among the four models with verified
% image-aligned prediction vectors.
%
% Adaptive Attention is deliberately excluded from paired comparisons
% because its updated prediction vector lacks verified image identifiers.

clear;
clc;

rng(20261105);

%% 1. Load original image-aligned predictions

load('EvaluationResults/Five_Model_ImageLevel_Predictions.mat');

T = combinedResults;

trueLabels = categorical(T.TrueLabel);

modelNames = {
    'EfficientNet-B0'
    'Channel Attention'
    'Spatial Attention'
    'CBAM'
};

predictionColumns = {
    'EfficientNetB0'
    'ChannelAttention'
    'SpatialAttention'
    'CBAM'
};

Y = cell(4,1);

for m = 1:4
    Y{m} = categorical(T.(predictionColumns{m}));
end

n = height(T);
nModels = 4;

%% 2. Verify data

assert(n == 489, 'Unexpected test-set size.');

for m = 1:nModels
    assert(numel(Y{m}) == n, ...
        'Prediction vector length mismatch.');
end

fprintf('Test images: %d\n',n);
fprintf('Models included: %d\n',nModels);

%% 3. Generate six pairwise comparisons

pairs = nchoosek(1:nModels,2);
nPairs = size(pairs,1);

nBootstrap = 10000;

classOrder = categories(trueLabels);
nClasses = numel(classOrder);

classIndices = cell(nClasses,1);

for c = 1:nClasses
    classIndices{c} = find(trueLabels == classOrder{c});
end

pairedResults = table();
mcnemarResults = table();

%% 4. Paired stratified bootstrap

for p = 1:nPairs

    i = pairs(p,1);
    j = pairs(p,2);

    fprintf('\nComparison: %s vs %s\n', ...
        modelNames{i},modelNames{j});

    accDiff = zeros(nBootstrap,1);
    f1Diff = zeros(nBootstrap,1);

    for b = 1:nBootstrap

        sampledIndices = [];

        for c = 1:nClasses

            idx = classIndices{c};

            sampledIndices = [sampledIndices; ...
                idx(randi(numel(idx),numel(idx),1))];

        end

        yt = trueLabels(sampledIndices);

        pi = Y{i}(sampledIndices);
        pj = Y{j}(sampledIndices);

        accI = mean(pi == yt)*100;
        accJ = mean(pj == yt)*100;

        f1I = calculateMacroF1(yt,pi,classOrder);
        f1J = calculateMacroF1(yt,pj,classOrder);

        accDiff(b) = accI-accJ;
        f1Diff(b) = f1I-f1J;

    end

    accCI = prctile(accDiff,[2.5 97.5]);
    f1CI = prctile(f1Diff,[2.5 97.5]);

    pairedResults = [pairedResults; table( ...
        string(modelNames{i}), ...
        string(modelNames{j}), ...
        mean(accDiff), ...
        accCI(1),accCI(2), ...
        mean(f1Diff), ...
        f1CI(1),f1CI(2), ...
        'VariableNames', ...
        {'Model1','Model2', ...
        'AccuracyDifference','AccuracyCILower', ...
        'AccuracyCIUpper','MacroF1Difference', ...
        'MacroF1CILower','MacroF1CIUpper'})];

    %% 5. Exact McNemar test

    correctI = (Y{i} == trueLabels);
    correctJ = (Y{j} == trueLabels);

    bDiscordant = sum(correctI & ~correctJ);
    cDiscordant = sum(~correctI & correctJ);

    d = bDiscordant+cDiscordant;

    if d == 0
        pValue = 1;
    else
        pValue = min(1, ...
            2*binocdf(min(bDiscordant,cDiscordant),d,0.5));
    end

    mcnemarResults = [mcnemarResults; table( ...
        string(modelNames{i}), ...
        string(modelNames{j}), ...
        bDiscordant,cDiscordant,pValue, ...
        'VariableNames', ...
        {'Model1','Model2','DiscordantB', ...
        'DiscordantC','ExactP'})];

end

%% 6. Holm correction

[pSorted,sortIdx] = sort(mcnemarResults.ExactP);

adjustedSorted = zeros(nPairs,1);

for k = 1:nPairs

    adjustedSorted(k) = min(1, ...
        (nPairs-k+1)*pSorted(k));

end

for k = 2:nPairs
    adjustedSorted(k) = max( ...
        adjustedSorted(k),adjustedSorted(k-1));
end

holmP = zeros(nPairs,1);
holmP(sortIdx) = adjustedSorted;

mcnemarResults.HolmAdjustedP = holmP;
mcnemarResults.Significant = holmP < 0.05;

%% 7. Display results

fprintf('\n============================================\n');
fprintf('TABLE 13: PAIRED BOOTSTRAP RESULTS\n');
fprintf('============================================\n');

disp(pairedResults);

fprintf('\n============================================\n');
fprintf('TABLE 14: McNEMAR TEST RESULTS\n');
fprintf('============================================\n');

disp(mcnemarResults);

%% 8. Save results

outputFolder = ...
    'EvaluationResults/Final_Paired_Statistical_Analysis';

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

writetable(pairedResults, ...
    fullfile(outputFolder,'Table13_PairedBootstrap.csv'));

writetable(mcnemarResults, ...
    fullfile(outputFolder,'Table14_McNemar_Holm.csv'));

save(fullfile(outputFolder, ...
    'Final_Paired_Statistical_Results.mat'), ...
    'pairedResults','mcnemarResults', ...
    'nBootstrap','modelNames');

fprintf('\nFinal paired statistical results saved.\n');

%% LOCAL FUNCTION

function macroF1 = calculateMacroF1(yt,yp,classOrder)

    f1 = zeros(numel(classOrder),1);

    for c = 1:numel(classOrder)

        actual = (yt == classOrder{c});
        predicted = (yp == classOrder{c});

        TP = sum(actual & predicted);
        FP = sum(~actual & predicted);
        FN = sum(actual & ~predicted);

        precision = TP/(TP+FP+eps);
        recall = TP/(TP+FN+eps);

        f1(c) = 2*precision*recall/ ...
            (precision+recall+eps);

    end

    macroF1 = mean(f1)*100;

end