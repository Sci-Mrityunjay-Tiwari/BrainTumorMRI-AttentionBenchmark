% ─── Reload dataset & splits (run this at the START of every session) ───
rng(42);   % CRITICAL: always same seed = same splits
dataPath = 'BrainTumorDataset/';
imds = imageDatastore(dataPath,'IncludeSubfolders',true,'LabelSource','foldernames');
classNames  = categories(imds.Labels);
numClasses  = numel(classNames);
[imdsTrain,imdsRem] = splitEachLabel(imds, 0.70,'randomized');
[imdsVal,imdsTest]  = splitEachLabel(imdsRem,0.50,'randomized');
inputSize = [224 224 3];
augOpts = imageDataAugmenter('RandRotation',[-15 15],'RandXReflection',true, ...
    'RandXScale',[0.9 1.1],'RandYScale',[0.9 1.1],'RandXShear',[-5 5]);
augTrain = augmentedImageDatastore(inputSize,imdsTrain,'DataAugmentation',augOpts,'ColorPreprocessing','gray2rgb');
augVal   = augmentedImageDatastore(inputSize,imdsVal,  'ColorPreprocessing','gray2rgb');
augTest  = augmentedImageDatastore(inputSize,imdsTest, 'ColorPreprocessing','gray2rgb');
fprintf('Dataset loaded: %d train | %d val | %d test\n', ...
    numel(imdsTrain.Files),numel(imdsVal.Files),numel(imdsTest.Files));
