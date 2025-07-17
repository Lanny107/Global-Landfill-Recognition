%% Segmentation Evaluation Script (Polished for GitHub Sharing)
% Performs segmentation analysis using IoU, mIoU, OA, and mAcc metrics
% across different refinement stages: hole-filling, dilation, aspect ratio
% filtering, and area variance adjustment.

clear; clc;

% Configuration
scale_height = 2000;
scale_width = 2000;
gt_dir = "path";
pred_dir = "path";

% Load folder contents
folder_image_GT = dir(fullfile(gt_dir, '*.png'));
folder_image_SG = dir(fullfile(pred_dir, '*.png'));

IoU_image = [];

for imageNo = 3:height(folder_image_SG)
    warning off
    fprintf('Processing image #%d\n', imageNo);

    % Load and preprocess images
    path_SG = fullfile(folder_image_SG(imageNo).folder, folder_image_SG(imageNo).name);
    image_SG = imbinarize(imread(path_SG));
    image_SG_scale = imresize(image_SG, [scale_height, scale_width]);

    gt_name = strrep(folder_image_SG(imageNo).name, '_pred', '_gt');
    path_GT = fullfile(folder_image_GT(imageNo).folder, gt_name);
    image_GT = imbinarize(imread(path_GT));

    % Inverted masks
    image_GT_inv = ~image_GT;
    image_SG_inv = ~image_SG_scale;

    % --- IoU (original) ---
    IoU_original = computeIoU(image_GT, image_SG_scale);
    IoU_original_rs = computeIoU(image_GT_inv, image_SG_inv);
    mIoU_original = mean([IoU_original, IoU_original_rs]);

    % --- Feature 1: Fill holes ---
    CC = bwconncomp(image_SG_scale);
    if CC.NumObjects > 0
        image_SG_filled = imfill(image_SG_scale, 'holes');
        image_SG_filled_inv = ~image_SG_filled;
        IoU_holeFilled = computeIoU(image_GT, image_SG_filled);
        IoU_holeFilled_rs = computeIoU(image_GT_inv, image_SG_filled_inv);
        mIoU_holeFilled = mean([IoU_holeFilled, IoU_holeFilled_rs]);
    else
        image_SG_filled = image_SG_scale;
        mIoU_holeFilled = mIoU_original;
    end

    % --- Feature 2: Scale up if necessary ---
    imForScale = image_SG_filled;
    CC = bwconncomp(imForScale);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    stats = regionprops(imForScale, 'Perimeter');
    APratio = [stats.Perimeter]' ./ numPixels';

    if mean(APratio) > 0.01
        se = strel('square', 12);
        imForScale = imdilate(imForScale, se);
    end

    imForScale_dilate = imForScale;
    image_SG_dilate_inv = ~imForScale_dilate;
    IoU_scaleUp = computeIoU(image_GT, imForScale_dilate);
    IoU_scaleUp_rs = computeIoU(image_GT_inv, image_SG_dilate_inv);
    mIoU_scaleUp = mean([IoU_scaleUp, IoU_scaleUp_rs]);

    % --- Feature 3: Remove by aspect ratio ---
    image_SG_LS = imForScale_dilate;
    CC = bwconncomp(image_SG_LS);
    numPixels = cellfun(@numel, CC.PixelIdxList);

    for objectNum = 1:length(numPixels)
        region = bwareafilt(image_SG_LS, [numPixels(objectNum), numPixels(objectNum)]);
        props = regionprops(region, 'MajorAxisLength', 'MinorAxisLength');
        if ~isempty(props) && props.MajorAxisLength / props.MinorAxisLength > 5.5
            image_SG_LS(CC.PixelIdxList{objectNum}) = 0;
        end
    end

    image_SG_LS_inv = ~image_SG_LS;
    IoU_LSratio = computeIoU(image_GT, image_SG_LS);
    IoU_LSratio_rs = computeIoU(image_GT_inv, image_SG_LS_inv);
    mIoU_LSratio = mean([IoU_LSratio, IoU_LSratio_rs]);

    % --- Feature 4: Remove smallest area by CV ---
    image_SG_area = image_SG_LS;
    CC = bwconncomp(image_SG_area);
    while CC.NumObjects > 0
        numPixels = cellfun(@numel, CC.PixelIdxList);
        normCV = 1 - 1 / (1 + std(numPixels) / mean(numPixels));
        if normCV > 0.4
            [~, idx] = min(numPixels);
            image_SG_area(CC.PixelIdxList{idx}) = 0;
            CC = bwconncomp(image_SG_area);
        else
            break;
        end
    end

    image_calibrated = image_SG_area;
    image_calibrated_inv = ~image_calibrated;
    IoU_VarianceRmv = computeIoU(image_GT, image_calibrated);
    IoU_VarianceRmv_rs = computeIoU(image_GT_inv, image_calibrated_inv);
    mIoU_VarianceRmv = mean([IoU_VarianceRmv, IoU_VarianceRmv_rs]);

    % --- Store Results ---
    sub_IoU_image = [imageNo, IoU_original, IoU_original_rs, mIoU_original, ...
        IoU_holeFilled, IoU_holeFilled_rs, mIoU_holeFilled, ...
        IoU_LSratio, IoU_LSratio_rs, mIoU_LSratio, ...
        IoU_VarianceRmv, IoU_VarianceRmv_rs, mIoU_VarianceRmv, ...
        IoU_scaleUp, IoU_scaleUp_rs, mIoU_scaleUp];

    IoU_image = [IoU_image; sub_IoU_image];
end

% Summary statistics
fprintf('\n--- Mean IoU Results ---\n');
labels = {'Original', 'Original_RS', 'mIoU_Original', 'HoleFilled', 'HoleFilled_RS', 'mIoU_HoleFilled', ...
          'LSratio', 'LSratio_RS', 'mIoU_LSratio', 'VarianceRmv', 'VarianceRmv_RS', 'mIoU_VarianceRmv', ...
          'ScaleUp', 'ScaleUp_RS', 'mIoU_ScaleUp'};

for i = 1:length(labels)
    fprintf('%s: %.4f\n', labels{i}, mean(IoU_image(:, i+1)));
end

% Save results
writematrix(IoU_image, 'IoU_image.xlsx');

%% --- Utility Function ---
function iou = computeIoU(gt, pred)
    intersection = gt & pred;
    union = gt | pred;
    iou = round(sum(intersection(:)) / sum(union(:)), 4);
end
