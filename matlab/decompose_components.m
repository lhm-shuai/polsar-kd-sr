% decompose_components.m  --  Stage I(a) of the pipeline
%
% Runs the Yamaguchi four-component decomposition on every coherency-matrix
% .mat file and writes one .npy per scene holding the three components used by
% the rest of the pipeline:
%
%     npy[..., 0] = Pm = Pd + Ph   (ship-characteristic scattering)  -> "R"
%     npy[..., 1] = Pv             (volume scattering)               -> "G"
%     npy[..., 2] = Ps = Podd      (surface scattering)              -> "B"
%
% The channel order above is the one the paper uses for the 8-bit rendering in
% Eqs. (7)-(8) and it is what the BT.601 luminance weights multiply. Do not
% permute it: the luma threshold depends on it.
%
% Requires: yamaguchi_4components_T3.m, utils/writeNPY.m, utils/constructNPYheader.m
%
% Paper: Section II-A.
% ---------------------------------------------------------------------------

clear;
clc;
close all;

%% ----------------------------- configuration -----------------------------
matFolderPath   = "D:\雷达\数据库\PSDDv1.0\mat";        % input: coherency .mat
outputNpyFolder = "D:\雷达\数据库\PSDDv1.0\四成分";      % output: component .npy
%% -------------------------------------------------------------------------

stretchPercent = 1;       % 线性拉伸百分比（增强对比度）

%  线性拉伸（增强伪彩色图对比度）
function imgStretched = linearStretch(img, percent)
    epsilon = 1e-6;
    % 筛选有效像素（剔除NaN和零值）
    validPixels = img(~isnan(img) & img > 0);
    if isempty(validPixels)
        imgStretched = zeros(size(img));
        return;
    end
    % 按百分比截取极值
    lower = prctile(validPixels, percent);
    upper = prctile(validPixels, 100 - percent);
    % 避免除数为零
    if (upper - lower) < epsilon
        imgStretched = zeros(size(img));
    else
        % 归一化到[0,1]区间
        imgStretched = (img - lower) / (upper - lower);
        imgStretched(imgStretched < 0) = 0;
        imgStretched(imgStretched > 1) = 1;
    end
end

%% 1. 定义文件路径
outputNpyFolder = "D:\雷达\数据库\PSDDv1.0\四成分";       % 最终npy输出路径

% 创建文件夹（不存在则自动创建）
if ~exist(outputNpyFolder, 'dir'); mkdir(outputNpyFolder); end

% 获取所有.mat文件的信息
fileList = dir(fullfile(matFolderPath, '*.mat'));
[~, idx] = sort({fileList.name});
fileList = fileList(idx);
numFiles = length(fileList);

% 循环读取每个文件
for k = 1:numFiles
    % ========== 核心修复：清空循环内变量，避免维度复用 ==========
    clearvars -except k numFiles fileList matFolderPath tempMatFolder outputNpyFolder stretchPercent linearStretch
    
    try
        % 构建完整文件路径
        matFilePath = fullfile(matFolderPath, fileList(k).name);
        load(matFilePath);

        % 合成复矩阵
        T12 = T12_real + 1j * T12_imag;
        T13 = T13_real + 1j * T13_imag;
        T23 = T23_real + 1j * T23_imag;

        % 山口四分量分解
        [Podd, Pdbl, Pvol, Phel] = yamaguchi_4components_T3(T11, T12, T13, T22, T23, T33, 0);
        Pman = Pdbl + Phel;

        % 去除NaN值
        Pman(isnan(Pman)) = 0;
        Pvol(isnan(Pvol)) = 0;
        Podd(isnan(Podd)) = 0;

        % 可选：启用线性拉伸（如需增强对比度）
        % Pman = linearStretch(Pman, stretchPercent);
        % Pvol = linearStretch(Pvol, stretchPercent);
        % Podd = linearStretch(Podd, stretchPercent);

        % 构建RGB矩阵
        [M, N]  = size(T11);
        RGB = zeros(M,N,3);
        RGB(:,:,1)= Pman; RGB(:,:,2)= Pvol; RGB(:,:,3)= Podd;

        % 提取文件名（不含扩展名）
        [~, baseName, ~] = fileparts(fileList(k).name);

        % 构建npy输出路径
        npyPath = fullfile(outputNpyFolder, [baseName, '.npy']);
        % 直接保存RGB矩阵为.npy
        writeNPY(RGB, npyPath); % 使用numpyio的writeNPY函数
        fprintf('已保存为npy：%s\n', npyPath);
       
    catch ME
        % 异常捕获：单个文件出错不中断循环
        fprintf('处理文件出错');
        continue;
    end
end

fprintf('\n所有文件保存完成！\n');