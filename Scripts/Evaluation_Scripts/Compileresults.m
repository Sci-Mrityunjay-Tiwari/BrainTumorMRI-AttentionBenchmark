%% ========================================================================
%  FINAL RESULTS COMPILATION
%  Load all 7 trained models and generate comprehensive comparison
%  ========================================================================

clear; clc; close all;

fprintf('\n╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║       FINAL RESULTS COMPILATION - ALL 7 MODELS                ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

% Create directories for outputs
if ~exist('Results', 'dir'), mkdir('Results'); end
if ~exist('Figures', 'dir'), mkdir('Figures'); end

% ── LOAD DATA (for evaluation) ───────────────────────────────────────────
fprintf('Loading dataset for evaluation...\n');
rng(42);
dataPath = 'BrainTumorDataset/';
imds = imageDatastore(dataPath,'IncludeSubfolders',true,'LabelSource','foldernames');
classNames  = categories(imds.Labels);
numClasses  = numel(classNames);
[imdsTrain,imdsRem] = splitEachLabel(imds, 0.70,'randomized');
[imdsVal,imdsTest]  = splitEachLabel(imdsRem,0.50,'randomized');
inputSize = [224 224 3];
augOpts = imageDataAugmenter('RandRotation',[-15 15],'RandXReflection',true, ...
'RandXScale',[0.9 1.1],'RandYScale',[0.9 1.1],'RandXShear',[-5 5]);
augTest  = augmentedImageDatastore(inputSize,imdsTest,'ColorPreprocessing','gray2rgb');

fprintf('✓ Dataset loaded\n\n');

% ── DEFINE MODELS TO LOAD ────────────────────────────────────────────────
models = {
    'efficientnetb0_braintumor',         'EfficientNetB0',
    'resnet50_braintumor',               'ResNet50',
    'googlenet_braintumor',              'GoogLeNet',
    'vgg16_braintumor',                  'VGG16',
    'efficientnet_cbam_braintumor',      'EfficientNet-CBAM',
    'efficientnet_ch_only_braintumor',   'EfficientNet-CA',
    'efficientnet_sp_only_braintumor',   'EfficientNet-SA'
};

nModels = size(models, 1);
results = {};

% ── LOAD AND EVALUATE ALL MODELS ─────────────────────────────────────────
fprintf('Loading and evaluating all models...\n');
fprintf('%-35s %10s %10s %10s %10s\n', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1');
fprintf(repmat('─',1,75)); fprintf('\n');

for i = 1:nModels
    modelFile = sprintf('Models/%s.mat', models{i,1});
    modelName = models{i,2};
    
    if ~isfile(modelFile)
        fprintf('✗ %s NOT FOUND\n', modelName);
        continue;
    end
    
    % Load model
    load(modelFile, 'net_eff', 'net_res', 'net_vgg', 'net_ggl', ...
                   'net_cbam', 'net_ch_only', 'net_sp_only');
    
    % Get the trained network
    if i==1, net = net_eff;
    elseif i==2, net = net_res;
    elseif i==3, net = net_ggl;
    elseif i==4, net = net_vgg;
    elseif i==5, net = net_cbam;
    elseif i==6, net = net_ch_only;
    elseif i==7, net = net_sp_only;
    end
    
    % Evaluate
    predLabels = classify(net, augTest);
    trueLabels = imdsTest.Labels;
    
    % Compute metrics properly
    C = confusionmat(trueLabels, predLabels);
    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2) - TP;
    TN = sum(C(:)) - TP - FP - FN;
    
    acc  = sum(TP) / sum(C(:));
    prec = mean(TP ./ (TP + FP + eps));
    rec  = mean(TP ./ (TP + FN + eps));
    f1   = 2 * prec * rec / (prec + rec + eps);
    spec = mean(TN ./ (TN + FP + eps));
    
    % Store results
    results{i,1} = modelName;
    results{i,2} = acc;
    results{i,3} = prec;
    results{i,4} = rec;
    results{i,5} = f1;
    results{i,6} = spec;
    results{i,7} = C;
    
    % Print
    fprintf('%-35s %10.2f%% %10.2f%% %10.2f%% %10.2f%%\n', ...
        modelName, acc*100, prec*100, rec*100, f1*100);
end

fprintf(repmat('─',1,75)); fprintf('\n\n');

% ── SAVE RESULTS TO CSV ──────────────────────────────────────────────────
fprintf('Saving results to CSV...\n');
fname = 'Results/all_models_comparison.csv';
fid = fopen(fname, 'w');
fprintf(fid, 'Model,Accuracy,Precision,Recall,F1-Score,Specificity\n');
for i = 1:nModels
    if ~isempty(results{i,1})
        fprintf(fid, '%s,%.4f,%.4f,%.4f,%.4f,%.4f\n', ...
            results{i,1}, results{i,2}, results{i,3}, results{i,4}, results{i,5}, results{i,6});
    end
end
fclose(fid);
fprintf('✓ Saved: %s\n\n', fname);

% ── CREATE COMPARISON PLOTS ──────────────────────────────────────────────
fprintf('Creating comparison visualizations...\n');

% Prepare data
modelNames = {};
accuracies = [];
for i = 1:nModels
    if ~isempty(results{i,1})
        modelNames = [modelNames; results{i,1}];
        accuracies = [accuracies; results{i,2}*100];
    end
end

% PLOT 1: Accuracy Comparison (Separate Figure)
figure('Position', [100 100 1200 600], 'Color', 'white');
bar(1:length(accuracies), accuracies, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'black', 'LineWidth', 2);
set(gca, 'XTickLabel', modelNames, 'XTickLabelRotation', 45, 'FontSize', 11);
ylabel('Accuracy (%)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Model', 'FontSize', 13, 'FontWeight', 'bold');
title('Brain Tumor Classification - Model Accuracy Comparison', 'FontSize', 14, 'FontWeight', 'bold');
grid on; grid minor;
ylim([60 100]);
for i = 1:length(accuracies)
    text(i, accuracies(i)+1, sprintf('%.2f%%', accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end
saveas(gcf, 'Figures/01_accuracy_comparison.png');
fprintf('✓ Saved: Figures/01_accuracy_comparison.png\n');

% PLOT 2: Detailed Metrics Comparison (Separate Figure)
figure('Position', [100 100 1400 600], 'Color', 'white');
metrics = [];
for i = 1:nModels
    if ~isempty(results{i,1})
        metrics = [metrics; results{i,2}, results{i,3}, results{i,4}, results{i,5}];
    end
end
x = 1:length(accuracies);
width = 0.2;
hold on;
bar(x-1.5*width, metrics(:,1)*100, width, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'black', 'LineWidth', 1.5);
bar(x-0.5*width, metrics(:,2)*100, width, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'black', 'LineWidth', 1.5);
bar(x+0.5*width, metrics(:,3)*100, width, 'FaceColor', [0.4 0.8 0.2], 'EdgeColor', 'black', 'LineWidth', 1.5);
bar(x+1.5*width, metrics(:,4)*100, width, 'FaceColor', [0.8 0.2 0.4], 'EdgeColor', 'black', 'LineWidth', 1.5);
set(gca, 'XTickLabel', modelNames, 'XTickLabelRotation', 45, 'FontSize', 11);
ylabel('Score (%)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Model', 'FontSize', 13, 'FontWeight', 'bold');
title('Detailed Metrics Comparison - Accuracy, Precision, Recall, F1-Score', 'FontSize', 14, 'FontWeight', 'bold');
legend('Accuracy', 'Precision', 'Recall', 'F1-Score', 'FontSize', 11, 'Location', 'lower right');
grid on; grid minor;
ylim([60 105]);
hold off;
saveas(gcf, 'Figures/02_metrics_comparison.png');
fprintf('✓ Saved: Figures/02_metrics_comparison.png\n');

% PLOT 3: Top 3 Models (Separate Figure)
figure('Position', [100 100 1000 600], 'Color', 'white');
[~, idx] = sort(accuracies, 'descend');
top3_names = modelNames(idx(1:3));
top3_acc = accuracies(idx(1:3));
h = bar(1:3, top3_acc, 'EdgeColor', 'black', 'LineWidth', 2);
h.FaceColor = 'flat';
colors_rgb = {[0.2 0.8 0.2], [0.2 0.6 0.8], [0.8 0.4 0.2]};
for j = 1:3
    h.CData(j,:) = colors_rgb{j};
end
set(gca, 'XTickLabel', top3_names, 'XTickLabelRotation', 45, 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Model Rank', 'FontSize', 13, 'FontWeight', 'bold');
title('Top 3 Best Performing Models', 'FontSize', 14, 'FontWeight', 'bold');
grid on; grid minor;
ylim([94 100]);
for j = 1:3
    text(j, top3_acc(j)+0.3, sprintf('%.2f%%', top3_acc(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
end
saveas(gcf, 'Figures/03_top3_models.png');
fprintf('✓ Saved: Figures/03_top3_models.png\n');

% PLOT 4: Ablation Study (EfficientNet variants only)
figure('Position', [100 100 1000 600], 'Color', 'white');
ablation_names = {'EfficientNetB0\n(Baseline)', 'EfficientNet-CA\n(Channel Only)', ...
                  'EfficientNet-CBAM\n(Full)', 'EfficientNet-SA\n(Spatial Only)'};
ablation_acc = [results{1,2}*100, results{6,2}*100, results{5,2}*100, results{7,2}*100];
h = bar(1:4, ablation_acc, 'EdgeColor', 'black', 'LineWidth', 2);
h.FaceColor = 'flat';
ablation_colors = {[0.7 0.7 0.7], [0.2 0.8 0.2], [0.2 0.6 0.8], [0.8 0.6 0.2]};
for j = 1:4
    h.CData(j,:) = ablation_colors{j};
end
set(gca, 'XTickLabel', strrep(ablation_names, '\n', ' '), 'XTickLabelRotation', 45, 'FontSize', 11);
ylabel('Accuracy (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('Ablation Study - EfficientNet Variants', 'FontSize', 14, 'FontWeight', 'bold');
grid on; grid minor;
ylim([93 97]);
for j = 1:4
    text(j, ablation_acc(j)+0.15, sprintf('%.2f%%', ablation_acc(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end
saveas(gcf, 'Figures/04_ablation_study.png');
fprintf('✓ Saved: Figures/04_ablation_study.png\n');

% ── FINAL SUMMARY ────────────────────────────────────────────────────────
fprintf('\n╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    RESULTS COMPILATION COMPLETE                ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('BEST MODEL: %s (%.2f%% Accuracy)\n\n', top3_names{1}, top3_acc(1));

fprintf('FILES GENERATED:\n');
fprintf('  ✓ Results/all_models_comparison.csv\n');
fprintf('  ✓ Figures/all_models_comparison.png\n\n');

fprintf('Ready for paper submission! 📊\n\n');