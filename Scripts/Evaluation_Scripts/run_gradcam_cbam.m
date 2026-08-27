clear; clc; close all;

load('TrainedModels/efficientnet_cbam_braintumor.mat', 'net_cbam');

classNames = {'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'};
inputSz = [224 224 3];

% Grad-CAM feature layer: channel attention output
featureLayer = 'channel_attention';

testPath = 'BrainTumorDataset/';
imdsTest = imageDatastore(testPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('Generating Grad-CAM visualizations (Channel Attention)...\n');

figure('Name','Grad-CAM Comparison','Color','white', ...
    'Units','inches','Position',[0 0 20 14]);

for c = 1:4

    idx = find(imdsTest.Labels == classNames{c}, 1);

    img = readimage(imdsTest, idx);

    if size(img,3) == 1
        img = repmat(img, [1 1 3]);
    end

    imgR = imresize(img, inputSz(1:2));
    imgN = single(imgR) / 255;

    % Grad-CAM from channel attention layer
    sm = gradCAM( ...
        net_cbam, ...
        imgN, ...
        classNames{c}, ...
        'FeatureLayer', featureLayer);

    hm = mat2gray(imresize(sm, inputSz(1:2)));

    ov = imfuse( ...
        imgR, ...
        ind2rgb(im2uint8(hm), jet(256)), ...
        'blend');

    subplot(4,4,(c-1)*4+1);
    imshow(imgR);
    title(['Input: ' classNames{c}], ...
        'FontSize',11,'FontWeight','bold');

    subplot(4,4,(c-1)*4+2);
    imagesc(hm);
    colormap(jet);
    axis image off;
    title('Grad-CAM Heatmap','FontSize',11);

    subplot(4,4,(c-1)*4+3);
    imagesc(hm);
    colormap(jet);
    axis image off;
    title('Activation Map','FontSize',11);

    subplot(4,4,(c-1)*4+4);
    imshow(ov);
    title('CBAM Overlay', ...
        'FontSize',11,'FontWeight','bold');

end

sgtitle( ...
    'Grad-CAM Analysis — EfficientNet-CBAM (Channel Attention Output)', ...
    'FontSize',14,'FontWeight','bold');

if ~exist('Figures', 'dir')
    mkdir('Figures');
end

saveas(gcf,'Figures/GradCAM_CBAM_Comparison.png');

fprintf('✓ Saved: Figures/GradCAM_CBAM_Comparison.png\n');