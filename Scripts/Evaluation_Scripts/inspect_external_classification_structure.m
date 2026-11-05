%% EXTERNAL DATASET CLASS DISTRIBUTION
clc;
clear;

datasetPaths = {
    fullfile('brisc2025','classification_task')
    'BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI'
};

datasetNames = {'BRISC 2025','BDNeuro-MRI'};

for d = 1:numel(datasetPaths)

    rootPath = datasetPaths{d};

    fprintf('\n====================================\n');
    fprintf('%s\n',datasetNames{d});
    fprintf('====================================\n');

    if ~isfolder(rootPath)
        fprintf('PATH NOT FOUND: %s\n',rootPath);
        continue;
    end

    imds = imageDatastore(rootPath, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

    counts = countEachLabel(imds);

    disp(counts);

    fprintf('Total images: %d\n',numel(imds.Files));

    fprintf('\nSample image information:\n');

    if ~isempty(imds.Files)

        sampleImage = readimage(imds,1);

        fprintf('Filename: %s\n',imds.Files{1});
        fprintf('Image dimensions: %s\n', ...
            mat2str(size(sampleImage)));

        fprintf('Image data type: %s\n', ...
            class(sampleImage));

    end

end

fprintf('\nInspection completed.\n');