%% ========================================================================
%  ABLATION STUDY BAR CHART
%  Visualize contribution of Channel and Spatial Attention components
%  ========================================================================

clear; clc; close all;

fprintf('\n╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║              ABLATION STUDY VISUALIZATION                      ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

% Create figure directory
if ~exist('Figures', 'dir'), mkdir('Figures'); end

% ── ABLATION STUDY DATA ──────────────────────────────────────────────────
% Model results from training
ablation_models = {
    'EfficientNetB0 (Baseline)',
    'EfficientNet-ChannelAttention Only',
    'EfficientNet-CBAM (Full)',
    'EfficientNet-SpatialAttention Only'
};

ablation_accuracy = [94.27, 96.52, 96.32, 94.68];  % Best run for CBAM
ablation_improvement = [0, 2.25, 2.05, 0.41];      % vs baseline

% ── PLOT 1: Accuracy with Baseline Reference ────────────────────────────
figure('Position', [100 100 1200 700], 'Color', 'white');

% Create bar chart
x = 1:4;
h = bar(x, ablation_accuracy, 0.6, 'EdgeColor', 'black', 'LineWidth', 2.5);
h.FaceColor = 'flat';

% Color coding
colors = [
    0.7 0.7 0.7;      % Baseline (gray)
    0.2 0.8 0.2;      % Channel Only (green)
    0.2 0.6 0.8;      % Full CBAM (blue)
    0.8 0.6 0.2       % Spatial Only (orange)
];

for i = 1:4
    h.CData(i,:) = colors(i,:);
end

% Add baseline line
yline(94.27, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 2.5, 'Label', 'Baseline (EfficientNetB0)');

% Formatting
set(gca, 'XTick', x, 'XTickLabel', ablation_models, 'FontSize', 12);
xtickangle(45);
ylabel('Accuracy (%)', 'FontSize', 14, 'FontWeight', 'bold');
title('Ablation Study: CBAM Component Contribution Analysis', 'FontSize', 15, 'FontWeight', 'bold');
grid on; grid minor; 
ylim([92 98]);

% Add value labels on bars
for i = 1:4
    text(i, ablation_accuracy(i) + 0.25, sprintf('%.2f%%', ablation_accuracy(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
    % Add improvement label
    if i > 1
        text(i, 92.5, sprintf('+%.2f%%', ablation_improvement(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0 0.5 0], 'FontWeight', 'bold');
    end
end

% Add legend
legend('Accuracy', 'Baseline', 'Location', 'northwest', 'FontSize', 11);

saveas(gcf, 'Figures/ablation_study_accuracy.png');
fprintf('✓ Saved: Figures/ablation_study_accuracy.png\n');

% ── PLOT 2: Component Contribution Comparison ────────────────────────────
fprintf('Creating Figure 2...\n');
figure('Position', [100 100 1000 700], 'Color', 'white');

% Data for comparison
components = categorical({'Baseline', 'Channel', 'Spatial', 'Channel +\nSpatial'});
components = reordercats(components, {'Baseline', 'Channel', 'Spatial', 'Channel +\nSpatial'});
performance = [94.27, 96.52, 94.68, 96.32];

h = bar(components, performance, 0.6, 'EdgeColor', 'black', 'LineWidth', 2.5);
h.FaceColor = 'flat';

% Color coding
component_colors = [
    0.7 0.7 0.7;      % Baseline
    0.2 0.8 0.2;      % Channel
    0.8 0.6 0.2;      % Spatial
    0.2 0.6 0.8       % Full
];

for i = 1:4
    h.CData(i,:) = component_colors(i,:);
end

ylabel('Accuracy (%)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Attention Component(s)', 'FontSize', 14, 'FontWeight', 'bold');
title('Component-wise Performance Contribution', 'FontSize', 15, 'FontWeight', 'bold');
grid on; grid minor;
ylim([92 98]);

% Add value labels
for i = 1:4
    text(i, performance(i) + 0.25, sprintf('%.2f%%', performance(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
end

saveas(gcf, 'Figures/ablation_study_components.png');
fprintf('✓ Saved: Figures/ablation_study_components.png\n');

% ── PLOT 3: Improvement Over Baseline ────────────────────────────────────
fprintf('Creating Figure 3...\n');
figure('Position', [100 100 1000 700], 'Color', 'white');

improvement_vals = [0, 2.25, 0.41, 2.05];
model_names = {'Baseline\n(No Attention)', 'Channel\nAttention Only', ...
               'Spatial\nAttention Only', 'CBAM\n(Full)'};

h = bar(1:4, improvement_vals, 0.6, 'EdgeColor', 'black', 'LineWidth', 2.5);
h.FaceColor = 'flat';

improvement_colors = [
    0.7 0.7 0.7;      % Baseline
    0.2 0.8 0.2;      % Channel (high)
    0.8 0.6 0.2;      % Spatial (low)
    0.2 0.6 0.8       % Full (medium)
];

for i = 1:4
    h.CData(i,:) = improvement_colors(i,:);
end

set(gca, 'XTick', 1:4, 'XTickLabel', model_names, 'FontSize', 11);
ylabel('Improvement Over Baseline (%)', 'FontSize', 14, 'FontWeight', 'bold');
title('Ablation Study: Performance Gain Analysis', 'FontSize', 15, 'FontWeight', 'bold');
grid on; grid minor;
ylim([0 2.5]);

% Add value labels
for i = 1:4
    if improvement_vals(i) > 0
        text(i, improvement_vals(i) + 0.1, sprintf('+%.2f%%', improvement_vals(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11, 'Color', [0 0.5 0]);
    else
        text(i, 0.1, 'Baseline', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
    end
end

saveas(gcf, 'Figures/ablation_study_improvement.png');
fprintf('✓ Saved: Figures/ablation_study_improvement.png\n');

% ── PRINT SUMMARY ────────────────────────────────────────────────────────
fprintf('\n╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                   ABLATION STUDY SUMMARY                       ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('BASELINE (EfficientNetB0):                94.27%%\n\n');

fprintf('COMPONENT ANALYSIS:\n');
fprintf('  Channel Attention Only:                96.52%%  (+2.25%%) ✅ BEST\n');
fprintf('  Spatial Attention Only:                94.68%%  (+0.41%%) ⚠️  Minimal\n');
fprintf('  Full CBAM (Channel + Spatial):         96.32%%  (+2.05%%)\n\n');

fprintf('KEY FINDINGS:\n');
fprintf('  • Channel Attention contributes ~2.25%% improvement\n');
fprintf('  • Spatial Attention alone provides minimal benefit (~0.41%%)\n');
fprintf('  • Full CBAM (2.05%%) is slightly less effective than Channel alone\n');
fprintf('  • Channel Attention is the PRIMARY driver of performance gain\n');
fprintf('  • Spatial Attention may introduce unnecessary complexity\n\n');

fprintf('CONCLUSION:\n');
fprintf('Channel Attention is the critical component for this task.\n');
fprintf('Spatial Attention adds complexity without consistent benefit.\n\n');

fprintf('✓ Ablation study complete! Charts saved to Figures/\n\n');