%% PHASE 1 - STEP 1.23
% Compare BDNeuro physical test images against prediction records

clc;

%% 1. Define paths

datasetRoot = fullfile(pwd, ...
    'BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI');

testRoot = fullfile(datasetRoot, 'test');

predictionFile = fullfile( ...
    'EvaluationResults', ...
    'ExternalValidation', ...
    'BDNeuro_MRI_Predictions.csv');

%% 2. Load prediction records

T = readtable(predictionFile);

trueLabels = string(T.TrueLabel);
imagePaths = string(T.ImagePath);

classNames = [
    "glioma_tumor"
    "meningioma_tumor"
    "no_tumor"
    "pituitary_tumor"
];

fprintf('\n========== BDNEURO DIRECTORY AUDIT ==========\n');

fprintf('Test directory:\n%s\n', testRoot);

fprintf('\n========== PHYSICAL IMAGE COUNTS ==========\n');

extensions = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'};

totalPhysicalImages = 0;

for k = 1:numel(classNames)

    classFolder = fullfile(testRoot, char(classNames(k)));

    if ~isfolder(classFolder)

        fprintf('\nMISSING CLASS FOLDER: %s\n', classFolder);
        continue;

    end

    imageFiles = [];

    for e = 1:numel(extensions)

        found = dir(fullfile(classFolder, ...
            ['*' extensions{e}]));

        imageFiles = [imageFiles; found]; %#ok<AGROW>

    end

    physicalCount = numel(imageFiles);
    predictionCount = sum(trueLabels == classNames(k));

    totalPhysicalImages = totalPhysicalImages + physicalCount;

    fprintf('\nClass: %s\n', classNames(k));
    fprintf('Physical images: %d\n', physicalCount);
    fprintf('Prediction records: %d\n', predictionCount);
    fprintf('Difference: %d\n', physicalCount-predictionCount);

end

fprintf('\nTotal physical images: %d\n', totalPhysicalImages);
fprintf('Total prediction records: %d\n', height(T));

%% 3. Check whether recorded image paths exist

fprintf('\n========== PREDICTION PATH EXISTENCE ==========\n');

pathExists = false(height(T),1);

for k = 1:height(T)

    pathExists(k) = isfile(char(imagePaths(k)));

end

fprintf('Existing image paths: %d\n', sum(pathExists));
fprintf('Missing image paths: %d\n', sum(~pathExists));

%% 4. Check whether any physical images are absent from predictions

fprintf('\n========== UNPREDICTED PHYSICAL IMAGES ==========\n');

for k = 1:numel(classNames)

    classFolder = fullfile(testRoot, char(classNames(k)));

    if ~isfolder(classFolder)
        continue;
    end

    imageFiles = [];

    for e = 1:numel(extensions)

        found = dir(fullfile(classFolder, ...
            ['*' extensions{e}]));

        imageFiles = [imageFiles; found]; %#ok<AGROW>

    end

    predictedPaths = imagePaths(trueLabels == classNames(k));

    predictedNames = strings(numel(predictedPaths),1);

    for j = 1:numel(predictedPaths)
        [~,name,ext] = fileparts(predictedPaths(j));
        predictedNames(j) = name + ext;
    end

    physicalNames = string({imageFiles.name})';

    missingNames = setdiff(physicalNames, predictedNames);

    fprintf('\n%s: %d unpredicted physical images\n', ...
        classNames(k), numel(missingNames));

    if ~isempty(missingNames)
        disp(missingNames(1:min(10,numel(missingNames))));
    end

end

fprintf('\n========== STEP 1.23 COMPLETE ==========\n');