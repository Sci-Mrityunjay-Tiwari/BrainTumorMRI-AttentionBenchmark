%% Adaptive Attention: Recalculate Results and Generate Confusion Matrix

clear;
clc;
close all;

%% 1. Load saved results

S = load('adaptive_attention_test_results.mat');

trueLabels = S.trueLabels;
predictedLabels = S.predictedLabels;

classNames = string(S.classNames);

%% 2. Recalculate confusion matrix

C_new = confusionmat(trueLabels, predictedLabels, ...
    'Order', categorical(classNames, classNames));

disp('CLASS ORDER:');
disp(classNames);

disp('RECALCULATED CONFUSION MATRIX:');
disp(C_new);

%% 3. Calculate class-wise metrics

TP = diag(C_new);
FP = sum(C_new,1)' - TP;
FN = sum(C_new,2) - TP;
TN = sum(C_new(:)) - TP - FP - FN;

precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);
specificity = TN ./ (TN + FP);
F1 = 2 .* precision .* recall ./ (precision + recall);

support = sum(C_new,2);

%% 4. Calculate overall metrics

accuracy_new = sum(TP) / sum(C_new(:));

macroPrecision_new = mean(precision);
macroRecall_new = mean(recall);
macroF1_new = mean(F1);

weightedF1_new = sum(F1 .* support) / sum(support);

%% 5. Display recalculated metrics

fprintf('\nRECALCULATED PERFORMANCE METRICS\n');
fprintf('Accuracy:          %.4f%%\n', accuracy_new*100);
fprintf('Macro-Precision:   %.4f%%\n', macroPrecision_new*100);
fprintf('Macro-Recall:      %.4f%%\n', macroRecall_new*100);
fprintf('Macro-F1:          %.4f%%\n', macroF1_new*100);
fprintf('Weighted-F1:       %.4f%%\n', weightedF1_new*100);

%% 6. Compare saved and recalculated metrics

disp(' ');
disp('COMPARISON WITH SAVED METRICS');

Comparison = table( ...
    [S.accuracy; S.macroPrecision; S.macroRecall; ...
     S.macroF1; S.weightedF1]*100, ...
    [accuracy_new; macroPrecision_new; macroRecall_new; ...
     macroF1_new; weightedF1_new]*100, ...
    'VariableNames', {'Saved','Recalculated'}, ...
    'RowNames', {'Accuracy','MacroPrecision','MacroRecall', ...
                 'MacroF1','WeightedF1'});

disp(Comparison);

%% 7. Export confusion matrix as CSV

writematrix(C_new, 'AdaptiveAttention_Updated_ConfusionMatrix.csv');

%% 8. Publication-quality confusion matrix (custom drawn)
%  - diagonal (45 deg) class labels on the x-axis
%  - row-normalized boxes placed right next to the matrix

% Display names: underscores replaced by spaces (avoids subscript issue)
dispNames = cellstr(strrep(classNames, '_', ' '));
n = numel(dispNames);

% Normalized summaries
rowSum = sum(C_new, 2);
colSum = sum(C_new, 1);
recallPct   = 100 * diag(C_new) ./ rowSum;      % row-normalized (correct)
rowErrPct   = 100 - recallPct;                  % row-normalized (error)
precisionPct = 100 * diag(C_new)' ./ colSum;    % column-normalized (correct)
colErrPct    = 100 - precisionPct;              % column-normalized (error)
recallPct(isnan(recallPct)) = 0;       rowErrPct(isnan(rowErrPct)) = 0;
precisionPct(isnan(precisionPct)) = 0; colErrPct(isnan(colErrPct)) = 0;

% Colors
greenC = [0.20 0.55 0.35];
pinkC  = [0.97 0.85 0.84];
fs     = 15;

fig = figure('Units','pixels', 'Position',[100 100 950 800], 'Color','w');
ax = axes(fig); hold(ax, 'on');

gap = 0.08;                 % small gap between matrix and summary boxes
maxDiag = max(diag(C_new));

% ---- Main confusion matrix ----
for i = 1:n
    for j = 1:n
        v = C_new(i,j);
        if i == j
            a = 0.45 + 0.55 * v / maxDiag;          % intensity by count
            col = 1 - a*(1 - greenC);
            txtCol = 'w'; if a < 0.7, txtCol = 'k'; end
        elseif v > 0
            col = pinkC; txtCol = 'k';
        else
            col = [1 1 1]; txtCol = 'k';
        end
        drawCell(ax, j, i, 1, 1, col);
        if v > 0
            text(ax, j, i, num2str(v), 'HorizontalAlignment','center', ...
                'Color', txtCol, 'FontName','Arial', 'FontSize', fs);
        end
    end
end

% ---- Row-normalized boxes (right, adjacent to matrix) ----
xr1 = n + 0.5 + gap + 0.5;          % centre of first summary column
xr2 = xr1 + 1;                      % centre of second summary column
for i = 1:n
    drawCell(ax, xr1, i, 1, 1, greenC);
    text(ax, xr1, i, sprintf('%.1f%%', recallPct(i)), ...
        'HorizontalAlignment','center','Color','w', ...
        'FontName','Arial','FontSize',fs);
    if rowErrPct(i) > 0
        drawCell(ax, xr2, i, 1, 1, pinkC);
        text(ax, xr2, i, sprintf('%.1f%%', rowErrPct(i)), ...
            'HorizontalAlignment','center','Color','k', ...
            'FontName','Arial','FontSize',fs);
    else
        drawCell(ax, xr2, i, 1, 1, [1 1 1]);
    end
end

% ---- Column-normalized boxes (bottom, adjacent to matrix) ----
yc1 = n + 0.5 + gap + 0.5;
yc2 = yc1 + 1;
for j = 1:n
    drawCell(ax, j, yc1, 1, 1, greenC);
    text(ax, j, yc1, sprintf('%.1f%%', precisionPct(j)), ...
        'HorizontalAlignment','center','Color','w', ...
        'FontName','Arial','FontSize',fs);
    if colErrPct(j) > 0
        drawCell(ax, j, yc2, 1, 1, pinkC);
        text(ax, j, yc2, sprintf('%.1f%%', colErrPct(j)), ...
            'HorizontalAlignment','center','Color','k', ...
            'FontName','Arial','FontSize',fs);
    else
        drawCell(ax, j, yc2, 1, 1, [1 1 1]);
    end
end

% ---- Axes formatting ----
set(ax, 'YDir','reverse', ...
        'XLim',[0.5 xr2+0.5], 'YLim',[0.5 yc2+0.5], ...
        'XTick',1:n, 'XTickLabel',dispNames, ...
        'YTick',1:n, 'YTickLabel',dispNames, ...
        'XTickLabelRotation',45, ...             % diagonal labels
        'TickLabelInterpreter','none', ...
        'TickLength',[0 0], 'Box','off', ...
        'XColor','k', 'YColor','k', ...
        'FontName','Arial', 'FontSize',fs, ...
        'DataAspectRatio',[1 1 1], 'Layer','top');
ax.XAxis.Color = 'none';  ax.YAxis.Color = 'none';   % hide axis lines
ax.XAxis.Visible = 'on';  ax.YAxis.Visible = 'on';
set(ax, 'XColor','k', 'YColor','k');                 % keep tick labels black

title(ax, 'Confusion Matrix - Adaptive Attention', ...
    'FontName','Arial', 'FontSize',fs+1, 'FontWeight','bold');
xlabel(ax, 'Predicted Class', 'FontName','Arial', 'FontSize',fs+1);
ylabel(ax, 'True Class',      'FontName','Arial', 'FontSize',fs+1);

%% 9. Export confusion matrix figure

exportgraphics(gcf, ...
    'AdaptiveAttention_ConfusionMatrix.png', ...
    'Resolution',600);

exportgraphics(gcf, ...
    'AdaptiveAttention_ConfusionMatrix.pdf', ...
    'ContentType','vector');

%% 10. Completion message

disp(' ');
disp('==============================================');
disp('ADAPTIVE ATTENTION ANALYSIS COMPLETED');
disp('==============================================');
disp('Generated files:');
disp('1. AdaptiveAttention_Updated_ConfusionMatrix.csv');
disp('2. AdaptiveAttention_ConfusionMatrix.png');
disp('3. AdaptiveAttention_ConfusionMatrix.pdf');

%% Local helper function
function drawCell(ax, cx, cy, w, h, col)
    patch(ax, cx + [-w w w -w]/2, cy + [-h -h h h]/2, col, ...
        'EdgeColor',[0.2 0.2 0.2], 'LineWidth',0.5);
end