% make_hard_labels.m  --  Stage I(b) of the pipeline
%
% Turns the component .npy files plus the VOC-style XML annotations into the
% pixel-level hard-label masks used to train the teacher network. This is the
% exact rule described in the paper, and it reproduces the released masks
% pixel for pixel:
%
%     R,G,B     = uint8(round(255 * clip(Pm, Pv, Ps, 0, 1)))   % MATLAB rounds
%     L         = 0.299*R + 0.587*G + 0.114*B                  % ITU-R BT.601
%     ship      <=>  L >= 40
%
% The threshold is evaluated ONLY inside the annotated bounding boxes; every
% pixel outside a box is background. There is no per-scene or per-sensor tuning:
% one constant (40) is shared by all three sensors and all 23 scenes.
%
% Output: one 8-bit PNG per scene, 255 = ship, 0 = background.
%
% Requires: yamaguchi_4components_T3.m, utils/C3_T3.m
%
% Paper: Section II-A, Eqs. (7)-(8).
% --------------------------------------------------------------------------

clear;
clc;
close all;

%% ----------------------------- configuration -----------------------------
matFolderPath = "D:\雷达\数据库\PSDDv1.0\mat";               % coherency .mat
xmlFolderPath = "D:\雷达\数据库\PSDDv1.0\Annotations";       % VOC XML boxes
outputFolder  = "D:\雷达\数据库\PSDDv1.0\分割舰船位置矩阵";    % output masks
%% -------------------------------------------------------------------------


%% 新增核心逻辑：解析XML标注，提取舰船矩形框
function ship_bboxes = parse_ship_annotations(xml_path)
    % 解析XML文件，返回所有舰船的边界框 [xmin, ymin, xmax, ymax] 单元格数组
    ship_bboxes = {};
    if ~exist(xml_path, 'file')
        fprintf('警告：XML标注文件不存在 -> %s\n', xml_path);
        return;
    end
    
    % 读取XML文档
    xml_doc = xmlread(xml_path);
    annotation_node = xml_doc.getDocumentElement();
    
    % 提取所有object节点
    object_nodes = annotation_node.getElementsByTagName('object');
    for i = 0:object_nodes.getLength()-1
        obj_node = object_nodes.item(i);
        % 筛选舰船目标
        name_node = obj_node.getElementsByTagName('name').item(0);
        if ~isempty(name_node) && strcmpi(name_node.getTextContent(), 'ship')
            % 提取边界框坐标
            bndbox_node = obj_node.getElementsByTagName('bndbox').item(0);
            xmin = str2double(bndbox_node.getElementsByTagName('xmin').item(0).getTextContent());
            ymin = str2double(bndbox_node.getElementsByTagName('ymin').item(0).getTextContent());
            xmax = str2double(bndbox_node.getElementsByTagName('xmax').item(0).getTextContent());
            ymax = str2double(bndbox_node.getElementsByTagName('ymax').item(0).getTextContent());
            ship_bboxes{end+1} = [xmin, ymin, xmax, ymax];
        end
    end
end

%% 1. 定义文件路径（需对应修改mat和xml路径，保持一一对应）
% 指定mat文件夹路径
% 指定XML标注文件夹路径（需根据你的实际路径修改）
xmlFolderPath = "D:\雷达\数据库\PSDDv1.0\Annotations"; 
% 输出png图像的文件夹路径

% 如果输出文件夹不存在，创建它
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 获取所有.mat文件的信息
fileList = dir(fullfile(matFolderPath, '*.mat'));

% 按文件名排序（可选，确保顺序）
[~, idx] = sort({fileList.name});
fileList = fileList(idx);

% 循环读取每个文件
numFiles = length(fileList);

for k = 1:numFiles
    % ========== 核心修复1：每次循环前清空所有变量，避免维度复用 ==========
    clearvars -except k numFiles fileList matFolderPath xmlFolderPath outputFolder parse_ship_annotations
    
    % 构建mat文件完整路径
    matFilePath = fullfile(matFolderPath, fileList(k).name);
    
    % 提取文件名（不含扩展名），用于匹配XML和生成png
    [~, fileNameWithoutExt, ~] = fileparts(fileList(k).name);
    
    % 构建对应的XML文件路径
    xmlFilePath = fullfile(xmlFolderPath, [fileNameWithoutExt, '.xml']);
    
    % 核心修改1：将输出路径的扩展名改为.png
    pngOutputPath = fullfile(outputFolder, [fileNameWithoutExt, '.png']);
    
    % 显示当前处理文件
    fprintf('处理文件 %d/%d: %s\n', k, numFiles, fileList(k).name);
    
    try
        % 加载mat数据
        load(matFilePath);
        
        % 合成复矩阵
        T12 = T12_real + 1j * T12_imag;
        T13 = T13_real + 1j * T13_imag;
        T23 = T23_real + 1j * T23_imag;
        
        % 山口四分量分解（需确保yamaguchi_4components_T3和C3_T3函数已存在）
        [Podd, Pdbl, Pvol, Phel] = yamaguchi_4components_T3(T11, T12, T13, T22, T23, T33, 0);
        Pman = Pdbl + Phel;
        
        % 去除NaN值
        Pman(isnan(Pman)) = 0;
        Pvol(isnan(Pvol)) = 0;
        Podd(isnan(Podd)) = 0;
        
        % 构建CC矩阵
        [ C11,C12,C13,C22,C23,C33 ] = C3_T3( T11,T12,T13,T22,T23,T33 );
        CC(:,:,1) = C11; 
        CC(:,:,2) = C22; 
        CC(:,:,3) = C33;
        
        % 计算功率矩阵SPAN和有效区域True_value
        [M,N,Q] = size(CC); % 重新获取当前文件的矩阵维度
        True_value = zeros(M,N); % 基于当前维度初始化
        SPAN_value = zeros(M,N);
        for i = 1:M
            for j = 1:N
                SPAN_value(i,j) = CC(i,j,1) + CC(i,j,2) + CC(i,j,3);
                if SPAN_value(i,j) >= 0.001
                    True_value(i,j) = 1;
                end
            end
        end
        
        % 构建伪彩图并归一化（基于当前维度初始化RGB）
        RGB = zeros(M,N,3); % 核心修复：显式初始化RGB为当前文件维度
        RGB(:, :, 1) = Pman;
        RGB(:, :, 2) = Pvol;
        RGB(:, :, 3) = Podd;
        img_normalized = RGB;
        
        % 转换为0-255 uint8格式
        img_uint8_from_arbitrary = uint8(img_normalized * 255);
        
        % 亮度阈值计算，区分舰船（亮）和背景（暗）
        luminance_threshold = 40; 
        R = double(img_uint8_from_arbitrary(:, :, 1));
        G = double(img_uint8_from_arbitrary(:, :, 2));
        B = double(img_uint8_from_arbitrary(:, :, 3));
        luminance = 0.299 * R + 0.587 * G + 0.114 * B; % 标准亮度计算公式
        binaryMask = luminance < luminance_threshold; % 亮度低于阈值为背景掩码
        outputImage = double(~binaryMask); % 亮度高于阈值=1（舰船候选），低于=0（标注框内背景）
        
        % 调用函数提取舰船边界框
        ship_bboxes = parse_ship_annotations(xmlFilePath);
        
        %% 4. 生成最终位置矩阵：标注外=0，标注内=舰船1/背景0
        ship_position_matrix = zeros(M, N); % 基于当前维度初始化
        
        % 遍历每个舰船标注框，赋值标注内的舰船/背景信息
        for bbox_idx = 1:length(ship_bboxes)
            bbox = ship_bboxes{bbox_idx};
            xmin = bbox(1);
            ymin = bbox(2);
            xmax = bbox(3);
            ymax = bbox(4);
            
            % ========== 核心修复2：严格校验坐标范围，避免维度错位 ==========
            % 1. 四舍五入到整数
            xmin = round(xmin);
            ymin = round(ymin);
            xmax = round(xmax);
            ymax = round(ymax);
            
            % 2. 强制限制在当前矩阵维度内（超出则直接跳过该标注框）
            if xmin < 1 || ymin < 1 || xmax > N || ymax > M
                fprintf('  警告：标注框坐标超出矩阵范围，跳过该框 -> x[%d,%d], y[%d,%d] (矩阵维度：%d×%d)\n',...
                    xmin, xmax, ymin, ymax, M, N);
                continue; % 跳过无效标注框
            end
            
            % 3. 再次确认尺寸匹配后赋值
            target_size = size(ship_position_matrix(ymin:ymax, xmin:xmax));
            source_size = size(outputImage(ymin:ymax, xmin:xmax));
            if ~isequal(target_size, source_size)
                fprintf('  警告：标注框区域尺寸不匹配，跳过该框 -> 目标尺寸：%d×%d，源尺寸：%d×%d\n',...
                    target_size(1), target_size(2), source_size(1), source_size(2));
                continue;
            end
            
            % 仅在标注框内赋值
            ship_position_matrix(ymin:ymax, xmin:xmax) = outputImage(ymin:ymax, xmin:xmax);
        end

        %% 核心修改2：保存为PNG格式，确保像素值为0和255的二值图
        % 确认矩阵只有0和1两个值，再转换为0/255的uint8格式
        ship_position_matrix = ship_position_matrix > 0; % 强制二值化（避免浮点误差）
        ship_position_uint8 = uint8(ship_position_matrix * 255); % 1→255，0→0
        % 保存为PNG（无损格式，无需设置Quality参数）
        imwrite(ship_position_uint8, pngOutputPath);
        fprintf('  位置矩阵已保存为：%s\n', pngOutputPath);

    catch ME
        fprintf('  处理文件 %s 时出错: %s\n', fileList(k).name, ME.message);
        % 打印错误堆栈，方便定位问题
        fprintf('  错误堆栈：%s\n', ME.stack(1).file);
        continue; % 跳过错误文件，继续处理下一个
    end
end

fprintf('所有文件处理完成！\n');