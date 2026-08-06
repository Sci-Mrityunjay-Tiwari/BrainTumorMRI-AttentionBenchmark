function [acc,prec,rec,f1,spec] = computeMetrics(C)
% ========================================================================
% COMPUTEMETRICS
% Computes classification performance metrics from a confusion matrix.
%
% Outputs:
%   acc  - Overall Accuracy
%   prec - Macro Precision
%   rec  - Macro Recall
%   f1   - Macro F1-Score
%   spec - Macro Specificity
% ========================================================================

% Number of classes
K = size(C,1);

% Confusion matrix components
TP = diag(C);
FP = sum(C,1)' - TP;
FN = sum(C,2)  - TP;
TN = sum(C(:)) - TP - FP - FN;

% Macro evaluation metrics
acc  = sum(TP) / sum(C(:));
prec = mean(TP ./ (TP + FP + eps));
rec  = mean(TP ./ (TP + FN + eps));
f1   = 2 * prec * rec / (prec + rec + eps);
spec = mean(TN ./ (TN + FP + eps));

% --------------------------------------------------------------------
% Per-class metrics (printed to Command Window)
% --------------------------------------------------------------------

defaultNames = {'Glioma','Meningioma','No Tumor','Pituitary'};

if K == numel(defaultNames)
    classNames = defaultNames;
else
    classNames = cell(K,1);
    for k = 1:K
        classNames{k} = sprintf('Class %d',k);
    end
end

fprintf('\nPer-class results:\n');
fprintf('%-14s %10s %10s %10s %10s\n',...
    'Class','Prec%','Rec%','F1%','Spec%');

for k = 1:K

    p = TP(k)/(TP(k)+FP(k)+eps);
    r = TP(k)/(TP(k)+FN(k)+eps);
    f = 2*p*r/(p+r+eps);
    s = TN(k)/(TN(k)+FP(k)+eps);

    fprintf('%-14s %9.2f %9.2f %9.2f %9.2f\n',...
        classNames{k},...
        p*100,...
        r*100,...
        f*100,...
        s*100);
end
end