%% EXTERNAL DATASET INSPECTION
clc;
clear;

fprintf('\n========================================\n');
fprintf(' EXTERNAL DATASET STRUCTURE INSPECTION\n');
fprintf('========================================\n');

%% Dataset paths

datasetPaths = {
    'brisc2025'
    'BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI'
};

imageExtensions = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'};

%% Inspect each dataset

for d = 1:numel(datasetPaths)

    rootPath = datasetPaths{d};

    fprintf('\n\n========================================\n');
    fprintf('DATASET: %s\n',rootPath);
    fprintf('========================================\n');

    if ~isfolder(rootPath)
        fprintf('ERROR: Folder not found: %s\n',rootPath);
        continue;
    end

    subfolders = dir(rootPath);
    subfolders = subfolders([subfolders.isdir]);
    subfolders = subfolders(~ismember({subfolders.name},{'.','..'}));

    fprintf('\nImmediate subfolders:\n');

    if isempty(subfolders)
        fprintf('No subfolders found.\n');
    end

    totalImages = 0;

    for k = 1:numel(subfolders)

        folderName = subfolders(k).name;
        folderPath = fullfile(rootPath,folderName);

        files = dir(fullfile(folderPath,'**','*'));

        imageCount = 0;

        for j = 1:numel(files)

            if ~files(j).isdir

                [~,~,ext] = fileparts(files(j).name);

                if any(strcmpi(ext,imageExtensions))
                    imageCount = imageCount + 1;
                end

            end

        end

        totalImages = totalImages + imageCount;

        fprintf('%-35s : %d images\n', ...
            folderName,imageCount);

    end

    %% Count all images, including images directly in root

    allFiles = dir(fullfile(rootPath,'**','*'));

    rootImageCount = 0;

    for j = 1:numel(allFiles)

        if ~allFiles(j).isdir

            [~,~,ext] = fileparts(allFiles(j).name);

            if any(strcmpi(ext,imageExtensions))
                rootImageCount = rootImageCount + 1;
            end

        end

    end

    fprintf('\nTOTAL IMAGE COUNT: %d\n',rootImageCount);

    fprintf('\nFiles directly inside dataset root:\n');

    rootFiles = dir(rootPath);
    rootFiles = rootFiles(~[rootFiles.isdir]);

    for k = 1:numel(rootFiles)
        fprintf('%s\n',rootFiles(k).name);
    end

end

fprintf('\n\nInspection completed.\n');