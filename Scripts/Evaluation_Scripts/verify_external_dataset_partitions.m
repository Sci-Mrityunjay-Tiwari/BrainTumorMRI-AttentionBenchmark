%% VERIFY EXTERNAL DATASET PARTITIONS
clc;
clear;

datasets = {
    'BRISC 2025', ...
    fullfile('brisc2025','classification_task'), ...
    {'train','test'};

    'BDNeuro-MRI', ...
    'BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI', ...
    {'train','val','test'}
    };

for d = 1:size(datasets,1)

    datasetName = datasets{d,1};
    rootPath = datasets{d,2};
    partitions = datasets{d,3};

    fprintf('\n========================================\n');
    fprintf('%s\n',datasetName);
    fprintf('========================================\n');

    grandTotal = 0;

    for p = 1:numel(partitions)

        partitionName = partitions{p};
        partitionPath = fullfile(rootPath,partitionName);

        fprintf('\nPARTITION: %s\n',partitionName);

        if ~isfolder(partitionPath)
            fprintf('Partition not found: %s\n',partitionPath);
            continue;
        end

        imds = imageDatastore(partitionPath, ...
            'IncludeSubfolders',true, ...
            'LabelSource','foldernames');

        counts = countEachLabel(imds);

        disp(counts);

        partitionTotal = numel(imds.Files);
        grandTotal = grandTotal + partitionTotal;

        fprintf('Partition total: %d\n',partitionTotal);

        % Inspect an actual MRI image
        if partitionTotal > 0

            sampleImage = readimage(imds,1);

            fprintf('Sample MRI filename: %s\n', ...
                imds.Files{1});

            fprintf('Image dimensions: %s\n', ...
                mat2str(size(sampleImage)));

            fprintf('Image data type: %s\n', ...
                class(sampleImage));

        end

    end

    fprintf('\n----------------------------------------\n');
    fprintf('TOTAL PARTITION IMAGES: %d\n',grandTotal);
    fprintf('----------------------------------------\n');

end

fprintf('\nVerification completed.\n');