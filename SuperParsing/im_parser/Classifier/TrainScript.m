if(~exist('HOME','var'))
    HOME = 'D:\im_parcer\Core';
    HOMEIMAGES = fullfile(HOME,'Images');
    HOMEANNOTATIONS = fullfile(HOME,'Annotations');
    HOMELABELSETS = {fullfile(HOME,'LabelsForgroundBK'),fullfile(HOME,'LabelsAnimalVehicle'),fullfile(HOME,'LabelsSemantic'),fullfile(HOME,'LabelsMaterial'),fullfile(HOME,'LabelsParts')};%%
    if(exist('LabelSetMask','var'))
        HOMELABELSETS = HOMELABELSETS(LabelSetMask);
    end
    HOMEDATA = fullfile(HOME,'Data');
end

if(~exist('segSuffix','var'))
    segSuffix = [];
end
fileList = dir_recurse(fullfile(HOMEIMAGES,'*.*'),0);
segIndex = cell(0); Labels = cell(0); Counts = cell(0);
for j=1:length(K)
    for i=1:length(HOMELABELSETS)
        [segIndex{i,j} Labels{i} Counts{i,j}] = LoadSegmentLabelIndex(fileList,[],HOMELABELSETS{i},...
                                                    fullfile(HOMEDATA,'Descriptors'),sprintf('SP_Desc_k%d%s',K(j),segSuffix));
    end
end

for i = 1:length(HOMELABELSETS)
    if(strfind(HOMELABELSETS{i},'LabelsMaterial')>0)
        if(length(Labels{i})==10 && Counts{i}(10) == 0)
            Labels{i}(10)=[];
            for j = 1:size(Counts,2)
                Counts{i,j}(10)=[];
            end
        end
    end
end

if(~exist('testSetNum','var'))
    testSetNum = 1;
end
if(exist('testSetName','var'))
    testName = testSetName;
    valName = [testName 'val'];
else
    testName = sprintf('TestSet%d',testSetNum);
    valName = sprintf('ValSet%d',testSetNum);
end
testSetFile = fullfile(HOME,[testName '.txt']);
valSetFile = fullfile(HOME,[valName '.txt']);
testFiles = importdata(testSetFile);
testMask = zeros(size(fileList))==1;
for i = 1:length(testFiles)
    testMask(strcmp(testFiles{i},fileList))=1;
end
valMask = zeros(size(fileList))==1;
if(exist(valSetFile,'file'))
    valFiles = importdata(valSetFile);
    for i = 1:length(valFiles)
        valMask(strcmp(valFiles{i},fileList))=1;
    end
end
testInd = find(testMask);
valInd = find(valMask);
trainInd = find(~testMask&~valMask);
trainFileList = fileList(~testMask&~valMask);


trainIndex=cell(0);
trainCounts=cell(0);
for i = 1:length(HOMELABELSETS)
    for j = 1:length(K)
        testMaskTemp = zeros(size(segIndex{i,j}.image))==1;
        for k = 1:length(testInd)
            testMaskTemp = testMaskTemp | (segIndex{i,j}.image==testInd(k));
        end
        valMaskTemp = zeros(size(segIndex{i,j}.image))==1;
        for k = 1:length(valInd)
            valMaskTemp = valMaskTemp | (segIndex{i,j}.image==valInd(k));
        end

        trainIndex{i,j}.label = segIndex{i,j}.label(~testMaskTemp&~valMaskTemp);
        trainIndex{i,j}.sp = segIndex{i,j}.sp(~testMaskTemp&~valMaskTemp);
        trainIndex{i,j}.spSize = segIndex{i,j}.spSize(~testMaskTemp&~valMaskTemp);
        indConverter = zeros(max(trainInd),1);indConverter(trainInd) = 1:length(trainInd);
        trainIndex{i,j}.image = indConverter(segIndex{i,j}.image(~testMaskTemp&~valMaskTemp));
        [l counts] = UniqueAndCounts(trainIndex{i,j}.label);
        trainCounts{i,j}=zeros(size(Counts{i,j}));
        trainCounts{i,j}(l)=counts;
    end
end
clear segIndex fileList testMaskTemp testFiles

if(~isfield(claParams,'num_nodes'))
    claParams.num_nodes = 8;
end
if(~isfield(claParams,'num_iterations'))
    claParams.num_iterations = 100;
end
if(~isfield(claParams,'stopval'))
    claParams.stopval = .001;
end
if(~isfield(claParams,'subSample'))
    claParams.subSample = 0;
end
if(~isfield(claParams,'testsetnum'))
    claParams.testsetnum = testSetNum;
end
if(~isfield(claParams,'segmentDescriptors'))
    claParams.segmentDescriptors = {'centered_mask_sp','bb_extent','pixel_area',...'centered_mask', %Shape
        'absolute_mask','top_height',...'bottom_height', %Location
        'int_text_hist_mr','dial_text_hist_mr',...'top_text_hist_mr','bottom_text_hist_mr','right_text_hist_mr','left_text_hist_mr' %Texture
        'sift_hist_int_','sift_hist_dial','sift_hist_bottom','sift_hist_top','sift_hist_right','sift_hist_left'... %Sift
        'mean_color','color_std','color_hist','dial_color_hist',... %Color
        'color_thumb','color_thumb_mask','gist_int'};
end
if(~isfield(claParams,'K'))
    claParams.K = K;
end
if(~isfield(claParams,'subSample'))
    claParams.subSample = 0;
end
if(~isfield(claParams,'balancedsubSample'))
    claParams.balancedsubSample = 1;
end
if(~isfield(claParams,'segSuffix'))
    claParams.segSuffix = segSuffix;
end

for i = 1:length(Labels)
    labelMask{i} = ones(size(Labels{i}))==1;
end

TrainClassifier(HOMEDATA, HOMELABELSETS, trainFileList, trainIndex, Labels, labelMask, claParams);


