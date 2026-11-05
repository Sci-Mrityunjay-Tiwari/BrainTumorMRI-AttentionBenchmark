%% EXACT DUPLICATE IMAGE AUDIT
clc;
clear;

fprintf('\n============================================\n');
fprintf(' EXACT DUPLICATE IMAGE AUDIT\n');
fprintf('============================================\n');

%% Dataset definitions

datasetNames = {
    'Bhuvaji'
    'BRISC_Test'
    'BDNeuro_Test'
};

datasetPaths = {
    'BrainTumorDataset'
    fullfile('brisc2025','classification_task','test')
    fullfile('BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI', ...
             'test')
};

%% Collect image information

allHashes = {};
allDatasets = {};
allLabels = {};
allFiles = {};

extensions = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'};

for d = 1:numel(datasetNames)

    fprintf('\nProcessing: %s\n',datasetNames{d});

    rootPath = datasetPaths{d};

    if ~isfolder(rootPath)
        error('Dataset folder not found: %s',rootPath);
    end

    imds = imageDatastore(rootPath, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

    fileCount = numel(imds.Files);

    fprintf('Images found: %d\n',fileCount);

    for k = 1:fileCount

        filePath = imds.Files{k};

        [~,~,ext] = fileparts(filePath);

        if ~any(strcmpi(ext,extensions))
            continue;
        end

        % Read original file bytes
        fid = fopen(filePath,'rb');

        if fid < 0
            warning('Could not open: %s',filePath);
            continue;
        end

        fileBytes = fread(fid,Inf,'*uint8');
        fclose(fid);

        % Calculate SHA-256 hash
        md = java.security.MessageDigest.getInstance('SHA-256');

        md.update(typecast(fileBytes,'int8'));

        digest = typecast(md.digest(),'uint8');

        hashValue = lower(reshape(dec2hex(digest,2).',1,[]));

        % Store information
        allHashes{end+1,1} = hashValue;
        allDatasets{end+1,1} = datasetNames{d};
        allLabels{end+1,1} = char(imds.Labels(k));
        allFiles{end+1,1} = filePath;

        if mod(k,500) == 0
            fprintf('Processed %d / %d images\n',k,fileCount);
        end

    end

end

%% Create audit table

auditTable = table( ...
    allDatasets, ...
    allLabels, ...
    allHashes, ...
    allFiles, ...
    'VariableNames', ...
    {'Dataset','ClassLabel','SHA256','FilePath'});

%% Identify exact duplicate hashes

[uniqueHashes,~,groupIndex] = unique(auditTable.SHA256);

groupCounts = accumarray(groupIndex,1);

duplicateGroups = find(groupCounts > 1);

fprintf('\n============================================\n');
fprintf(' AUDIT RESULTS\n');
fprintf('============================================\n');

fprintf('Total images hashed: %d\n',height(auditTable));

fprintf('Unique file hashes: %d\n',numel(uniqueHashes));

fprintf('Duplicate hash groups: %d\n',numel(duplicateGroups));

%% Compare Bhuvaji against external datasets

externalNames = {'BRISC_Test','BDNeuro_Test'};

for d = 1:numel(externalNames)

    externalName = externalNames{d};

    isBhuvaji = strcmp(auditTable.Dataset,'Bhuvaji');
    isExternal = strcmp(auditTable.Dataset,externalName);

    bhuvajiHashes = auditTable.SHA256(isBhuvaji);
    externalHashes = auditTable.SHA256(isExternal);

    commonHashes = intersect(bhuvajiHashes,externalHashes);

    fprintf('\n--------------------------------------------\n');
    fprintf('Bhuvaji vs %s\n',externalName);
    fprintf('Exact matching images: %d\n',numel(commonHashes));

end

%% Save all audit information

writetable(auditTable, ...
    'EvaluationResults/external_duplicate_audit_all_images.csv');

% Save duplicate records
duplicateTable = auditTable( ...
    ismember(auditTable.SHA256, ...
    uniqueHashes(duplicateGroups)),:);

writetable(duplicateTable, ...
    'EvaluationResults/external_duplicate_audit_matches.csv');

save('EvaluationResults/external_duplicate_audit.mat', ...
    'auditTable','duplicateTable');

fprintf('\n============================================\n');
fprintf('AUDIT COMPLETED\n');
fprintf('============================================\n');

fprintf('Audit files saved in EvaluationResults.\n');