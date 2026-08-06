function saveResults(modelName, runNumber, acc, prec, rec, f1, spec, ...
    trainingTime, timePerImage)
% ========================================================================
% SAVERESULTS
% Appends one experiment to Results/all_results.csv
%
% Inputs:
%   modelName       - Model name
%   runNumber       - Experimental run number
%   acc             - Accuracy
%   prec            - Precision
%   rec             - Recall
%   f1              - F1-score
%   spec            - Specificity
%   trainingTime    - Training time (seconds)
%   timePerImage    - Inference time per image (seconds)
% ========================================================================

% Create Results folder if needed
if ~exist('Results','dir')
    mkdir('Results');
end

fname = fullfile('Results','all_results.csv');

% Create CSV with header
if ~exist(fname,'file')

    fid = fopen(fname,'w');

    if fid==-1
        error('Unable to create %s',fname);
    end

    fprintf(fid,...
        'Model,Run,Accuracy (%%),Precision (%%),Recall (%%),F1-Score (%%),Specificity (%%),Training Time (min),Inference Time/Image (ms)\n');

    fclose(fid);

end

fid = fopen(fname,'a');

if fid==-1
    error('Unable to open %s.',fname);
end

fprintf(fid,...
    '%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f\n',...
    modelName,...
    runNumber,...
    acc*100,...
    prec*100,...
    rec*100,...
    f1*100,...
    spec*100,...
    trainingTime/60,...
    timePerImage*1000);

fclose(fid);

fprintf('✓ Results saved to %s\n',fname);

end