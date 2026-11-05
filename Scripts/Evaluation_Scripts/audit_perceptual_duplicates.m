%% PERCEPTUAL DUPLICATE IMAGE AUDIT
clc;
clear;

fprintf('\n============================================\n');
fprintf(' PERCEPTUAL DUPLICATE IMAGE AUDIT\n');
fprintf('============================================\n');

%% Dataset paths

datasetNames = {
    'Bhuvaji'
    'BRISC_Test'
    'BDNeuro_Test'
};

datasetPaths = {
    'BrainTumorDataset'
    fullfile('brisc2025','classification_task','test')
    fullfile('BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI','test')
};

%% Generate perceptual hashes

hashes = cell(3,1);
filePaths = cell(3,1);
classLabels = cell(3,1);

for d = 1:3

    fprintf('\nProcessing: %s\n',datasetNames{d});

    rootPath = datasetPaths{d};

    if ~isfolder(rootPath)
        error('Dataset folder not found: %s',rootPath);
    end

    imds = imageDatastore(rootPath, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

    n = numel(imds.Files);

    hashes{d} = zeros(n,1,'uint64');
    filePaths{d} = imds.Files;
    classLabels{d} = cellstr(imds.Labels);

    for k = 1:n

        img = readimage(imds,k);

        % Convert to grayscale
        if size(img,3) == 3
            img = im2gray(img);
        end

        % Convert to double and resize
        img = im2double(img);
        img = imresize(img,[32 32]);

        % Two-dimensional DCT
        D = dct2(img);

        % Extract low-frequency 8x8 coefficients
        lowFreq = D(1:8,1:8);

        % Exclude DC coefficient
        coefficients = lowFreq(:);
        coefficients(1) = [];

        threshold = median(coefficients);

        % Construct 64-bit perceptual hash
        bits = lowFreq(:);
        hashValue = uint64(0);

        for b = 1:64

            if bits(b) > threshold
                hashValue = bitset(hashValue,b);
            end

        end

        hashes{d}(k) = hashValue;

        if mod(k,250) == 0
            fprintf('Processed %d / %d images\n',k,n);
        end

    end

end

%% Compare Bhuvaji against external test datasets

comparisonPairs = [
    1 2
    1 3
];

results = table();

for p = 1:size(comparisonPairs,1)

    referenceIndex = comparisonPairs(p,1);
    externalIndex = comparisonPairs(p,2);

    referenceHashes = hashes{referenceIndex};
    externalHashes = hashes{externalIndex};

    nExternal = numel(externalHashes);

    nearestDistance = zeros(nExternal,1);
    nearestReference = zeros(nExternal,1);

    fprintf('\nComparing %s against %s\n', ...
        datasetNames{referenceIndex}, ...
        datasetNames{externalIndex});

    for k = 1:nExternal

        xorValues = bitxor(referenceHashes,externalHashes(k));

% Calculate Hamming distance across all 64 bits
distances = zeros(size(xorValues));

for bitPosition = 1:64

    distances = distances + ...
        double(bitget(xorValues,bitPosition));

end

[nearestDistance(k),nearestReference(k)] = min(distances);
        if mod(k,100) == 0
            fprintf('Compared %d / %d images\n',k,nExternal);
        end

    end

    %% Similarity summary

    fprintf('\nResults: %s\n',datasetNames{externalIndex});

    thresholds = [0 5 8 10];

    for t = thresholds

        count = sum(nearestDistance <= t);

        fprintf('Nearest Hamming distance <= %2d: %d images\n', ...
            t,count);

    end

    %% Store results

    comparisonTable = table( ...
        repmat(string(datasetNames{referenceIndex}),nExternal,1), ...
        repmat(string(datasetNames{externalIndex}),nExternal,1), ...
        string(classLabels{externalIndex}), ...
        string(filePaths{externalIndex}), ...
        string(filePaths{referenceIndex}(nearestReference)), ...
        nearestDistance, ...
        'VariableNames', ...
        {'ReferenceDataset','ExternalDataset','ExternalClass', ...
         'ExternalFile','NearestBhuvajiFile','HammingDistance'});

    results = [results; comparisonTable];

end

%% Save results

if ~isfolder('EvaluationResults')
    mkdir('EvaluationResults');
end

writetable(results, ...
    'EvaluationResults/perceptual_duplicate_audit.csv');

save('EvaluationResults/perceptual_duplicate_audit.mat', ...
    'results','hashes','filePaths','classLabels');

fprintf('\n============================================\n');
fprintf('PERCEPTUAL AUDIT COMPLETED\n');
fprintf('============================================\n');

fprintf('Results saved in EvaluationResults.\n');

%% Local function is not required.