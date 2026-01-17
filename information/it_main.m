clc;close all;clear all;warning off;%清除变量


namelist = dir('/home/zhen/bioMechanism/sal/VisualAttention-DeepRare2019/output_raw/*.jpg');
len = length(namelist);
it_entroy = zeros(len);
for ii = 1:len

    rand('seed', 100);
    randn('seed', 100);
    format long g;
     
    % 读取 Lena 图像
    
    filename = [namelist(ii).folder,'/',namelist(ii).name];
    lena_img = imread(filename);
     
    % 如果是彩色图像，则转换为灰度图像
    if size(lena_img, 3) == 3
        lena_img = rgb2gray(lena_img);
    end
     
    % 计算灰度直方图
    [counts, gray_levels] = imhist(lena_img);
     
    % 计算每个灰度级的概率
    total_pixels = sum(counts);
    probabilities = counts / total_pixels;
     
    % 计算信息熵
    entropy = 0;
    for i = 1:length(probabilities)
        if probabilities(i) > 0
            entropy = entropy - probabilities(i) * log2(probabilities(i));
        end
    end
     
    % 显示信息熵
    it_entroy(ii) = entropy;
    disp(['信息熵为: ', num2str(entropy)]);

    

%     
%     % 显示原图和加密后的图像
%     figure;
%     subplot(1, 2, 1);
%     imshow(uint8(lena_img));
%     title('原图');
%     subplot(1, 2, 2);
%     imhist(lena_img);
%     title('灰度直方图');

end

figure;
plot(it_entroy);
title('The values of entropy');
 
 
 