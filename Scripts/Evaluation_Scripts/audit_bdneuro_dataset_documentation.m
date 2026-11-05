%% PHASE 1 - STEP 1.25
% Inspect BDNeuro dataset documentation

clc;

datasetRoot = fullfile(pwd, ...
    'BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI');

%% 1. Read README

readmePath = fullfile(datasetRoot,'README.md');

fprintf('\n========== README CONTENT ==========\n');

if isfile(readmePath)

    readmeText = fileread(readmePath);
    disp(readmeText);

else

    fprintf('README.md not found.\n');

end

%% 2. Read class distribution CSV

distributionPath = fullfile(datasetRoot,'class_distribution.csv');

fprintf('\n========== CLASS DISTRIBUTION CSV ==========\n');

if isfile(distributionPath)

    distributionTable = readtable(distributionPath);
    disp(distributionTable);

else

    fprintf('class_distribution.csv not found.\n');

end

%% 3. Inspect split counts directly

splitNames = {'train','val','test'};

fprintf('\n========== PHYSICAL SPLIT COUNTS ==========\n');

extensions = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'};

for s = 1:numel(splitNames)

    splitRoot = fullfile(datasetRoot,splitNames{s});

    fprintf('\nSplit: %s\n',splitNames{s});

    splitTotal = 0;

    classFolders = dir(splitRoot);
    classFolders = classFolders([classFolders.isdir]);

    for c = 1:numel(classFolders)

        folderName = classFolders(c).name;

        if ismember(folderName,{'.','..'})
            continue;
        end

        classPath = fullfile(splitRoot,folderName);

        imageCount = 0;

        for e = 1:numel(extensions)

            imageFiles = dir(fullfile(classPath, ...
                ['*' extensions{e}]));

            imageCount = imageCount + numel(imageFiles);

        end

        fprintf('%-20s %d\n',folderName,imageCount);

        splitTotal = splitTotal + imageCount;

    end

    fprintf('TOTAL: %d\n',splitTotal);

end

fprintf('\n========== STEP 1.25 COMPLETE ==========\n');