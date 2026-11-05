%% PHASE 1 - STEP 1.24
% Inspect actual BDNeuro dataset directory structure

clc;

datasetRoot = fullfile(pwd, ...
    'BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI');

fprintf('\n========== DATASET ROOT ==========\n');
fprintf('%s\n', datasetRoot);

fprintf('\n========== ROOT CONTENTS ==========\n');

rootItems = dir(datasetRoot);

for k = 1:numel(rootItems)

    if ~ismember(rootItems(k).name,{'.','..'})

        if rootItems(k).isdir
            fprintf('[FOLDER] %s\n', rootItems(k).name);
        else
            fprintf('[FILE]   %s\n', rootItems(k).name);
        end

    end

end

fprintf('\n========== TEST DIRECTORY CONTENTS ==========\n');

testRoot = fullfile(datasetRoot,'test');

if isfolder(testRoot)

    testItems = dir(testRoot);

    for k = 1:numel(testItems)

        if ~ismember(testItems(k).name,{'.','..'})

            itemPath = fullfile(testRoot,testItems(k).name);

            if testItems(k).isdir

                fprintf('\n[FOLDER] %s\n',testItems(k).name);

                contents = dir(itemPath);

                imageCount = 0;

                for j = 1:numel(contents)

                    if ~contents(j).isdir

                        [~,~,ext] = fileparts(contents(j).name);

                        if ismember(lower(ext), ...
                                {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})

                            imageCount = imageCount + 1;

                        end

                    end

                end

                fprintf('Image count: %d\n',imageCount);

            else

                fprintf('[FILE] %s\n',testItems(k).name);

            end

        end

    end

else

    fprintf('TEST DIRECTORY NOT FOUND.\n');

end

fprintf('\n========== STEP 1.24 COMPLETE ==========\n');