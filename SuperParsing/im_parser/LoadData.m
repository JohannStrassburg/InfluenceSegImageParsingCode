HOMEDATA = fullfile(HOME,'Data');

%{
if(exist(HOMEANNOTATIONS,'file'))
    saveFile = fullfile(HOME,'D.mat');
    if(exist(saveFile,'file'))
        load(saveFile);
    else
        D = LMdatabase(HOMEANNOTATIONS);
        save(saveFile,'D');
    end
end
%}

%{-
listFile = fullfile(HOME,'fileList.txt');
if(exist(listFile,'file'))
    fileList = importdata(listFile);
else
    fileList = dir_recurse(fullfile(HOMEIMAGES,'*.*'),0);
end
if(~exist('K','var')); K=200; end
if(~exist('segSuffix','var')); segSuffix = []; end
if(~exist('useGlobal','var')); useGlobal = 1; end
segIndex = cell(0); Labels = cell(0); Counts = cell(0);
%{-
for j=1:length(K)
    %note you can run this command on multiple matlab instances to do simple multi-threading. 
    % but you have to go into the file and change the flag useBusyFile=true;
    %It won't try to compute the same image descriptors at the same time.
    %descFuns = sort({'centered_mask_sp','bb_extent','pixel_area','centered_mask','absolute_mask','top_height','bottom_height', 'int_text_hist_mr','dial_text_hist_mr','top_text_hist_mr','bottom_text_hist_mr','right_text_hist_mr','left_text_hist_mr','sift_hist_int_','sift_hist_dial','sift_hist_bottom','sift_hist_top','sift_hist_right','sift_hist_left','mean_color','color_std','color_hist','dial_color_hist','color_thumb','color_thumb_mask','gist_int'})';
    descFuns = sort({'centered_mask_sp','bb_extent','pixel_area','absolute_mask','top_height','int_text_hist_mr','dial_text_hist_mr','sift_hist_int_','sift_hist_dial','sift_hist_bottom','sift_hist_top','sift_hist_right','sift_hist_left','mean_color','color_std','color_hist','dial_color_hist','color_thumb','color_thumb_mask','gist_int'});
    descFuns = ComputeSegmentDescriptors( fileList, HOMEIMAGES, HOMEDATA,HOMEDATA, HOMECODE, 1, 1:length(fileList), K(j), segSuffix,descFuns);
    for i=1:length(HOMELABELSETS)
        [segIndex{i,j} Labels{i} Counts{i,j}] = LoadSegmentLabelIndex(fileList,[],HOMELABELSETS{i},fullfile(HOMEDATA,'Descriptors'),sprintf('SP_Desc_k%d%s',K(j),segSuffix));
    end
end

%{-
for i = 1:length(HOMELABELSETS)
    if(strfind(HOMELABELSETS{i},'LabelsMaterial')>0)
        if(length(Labels{i})==10 && strcmp(Labels{i}{10},'_'))
            Labels{i}(10)=[];
            for j = 1:size(Counts,2)
                Counts{i,j}(10)=[];
            end
        end
    end
end
%}
%if we are using a test set from a separate folder than the train set
if(exist('HOMETEST','var'))
    [foo trainFolder] = fileparts(HOME);
    testName = [trainFolder 'train'];
    testSetName = testName;
    
    testFileList = dir_recurse(fullfile(HOMETESTIMAGES,'*.*'),0);
    trainFileList = fileList;
    for j=1:length(K)
        descFuns = ComputeSegmentDescriptors( testFileList, HOMETESTIMAGES, HOMETESTDATA, HOMEDATA, HOMECODE, 1, 1:length(testFileList), K(j));
    end
    trainGlobalDesc = ComputeGlobalDescriptors(trainFileList, HOMEIMAGES, HOMELABELSETS, HOMEDATA);
    
    dicFile = fullfile(HOMETESTDATA,'Descriptors','Global','SpatialPyr','dictionary_200.mat'); make_dir(dicFile);
    copyfile(fullfile(HOMEDATA,'Descriptors','Global','SpatialPyr','dictionary_200.mat'),dicFile);
    dicFile = fullfile(HOMETESTDATA,'Descriptors','Global','SpatialPyrDense','dictionary_200.mat'); make_dir(dicFile);
    copyfile(fullfile(HOMEDATA,'Descriptors','Global','SpatialPyrDense','dictionary_200.mat'),dicFile);
    testGlobalDesc = ComputeGlobalDescriptors(testFileList, HOMETESTIMAGES, HOMETESTDATA);
    
    trainIndex = segIndex;
    trainCounts = Counts;
    testIndex=cell(0);
    testCounts=cell(0);
else
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
    testFileList = fileList(testMask);
    valFileList = fileList(valMask);
    trainFileList = fileList(~testMask&~valMask);
    if(exist('D','var'))
        Dtest = D(testMask);
        Dval = D(valMask);
        Dtrain = D(~testMask&~valMask);
        clear D;
    end
    
    if(useGlobal)
        trainGlobalDesc = ComputeGlobalDescriptors(trainFileList, HOMEIMAGES, HOMELABELSETS, HOMEDATA);
        testGlobalDesc = ComputeGlobalDescriptors(testFileList, HOMEIMAGES, HOMELABELSETS, HOMEDATA);
        valGlobalDesc = ComputeGlobalDescriptors(valFileList, HOMEIMAGES, HOMELABELSETS, HOMEDATA);
    else
        trainGlobalDesc = ComputeGlobalDescriptors([], HOMEIMAGES, HOMELABELSETS, HOMEDATA);
        testGlobalDesc = ComputeGlobalDescriptors([], HOMEIMAGES, HOMELABELSETS, HOMEDATA);
        valGlobalDesc = ComputeGlobalDescriptors([], HOMEIMAGES, HOMELABELSETS, HOMEDATA);
    end
    
    testIndex=cell(0);valIndex=cell(0);trainIndex=cell(0);
    testCounts=cell(0);valCounts=cell(0);trainCounts=cell(0);
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
            testIndex{i,j}.label = segIndex{i,j}.label(testMaskTemp);
            testIndex{i,j}.sp = segIndex{i,j}.sp(testMaskTemp);
            testIndex{i,j}.spSize = segIndex{i,j}.spSize(testMaskTemp);
            indConverter = zeros(max(testInd),1);indConverter(testInd) = 1:length(testInd);
            testIndex{i,j}.image = indConverter(segIndex{i,j}.image(testMaskTemp));
            [l counts] = UniqueAndCounts(testIndex{i,j}.label);
            testCounts{i,j}=zeros(size(Counts{i,j}));
            testCounts{i,j}(l)=counts;
            
            valIndex{i,j}.label = segIndex{i,j}.label(valMaskTemp);
            valIndex{i,j}.sp = segIndex{i,j}.sp(valMaskTemp);
            valIndex{i,j}.spSize = segIndex{i,j}.spSize(valMaskTemp);
            indConverter = zeros(max(valInd),1);indConverter(valInd) = 1:length(valInd);
            valIndex{i,j}.image = indConverter(segIndex{i,j}.image(valMaskTemp));
            [l counts] = UniqueAndCounts(valIndex{i,j}.label);
            valCounts{i,j}=zeros(size(Counts{i,j}));
            valCounts{i,j}(l)=counts;

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
end
%{=
for j = 1:1:length(K)
    lptemp{j} = ComputeLabelPenality(trainFileList,HOMEDATA,sprintf('SP_Desc_k%d%s',K(j),segSuffix),trainIndex(:,j),testName,Labels);
end
labelPenality = cell(size(lptemp{1}));
for j = 1:numel(labelPenality)
    temp = zeros([size(lptemp{1}{j}) length(lptemp)]);
    for k = 1:length(lptemp)
        temp(:,:,k) = lptemp{k}{j};
    end
    labelPenality{j} = mean(temp,3);
end
%}

for j = 1:1:length(K)
    Rs{j} = CalculateSearchRs(trainFileList,HOMEDATA,trainIndex{1,j},descFuns,K(j),segSuffix);
end

%labelSubSets = ComputeLabelSubSets(trainIndex(:,1),Labels);

if(~exist('UseClassifier','var'))
    UseClassifier = zeros(size(HOMELABELSETS));
end

claParams.num_nodes = 8;
claParams.stopval = .1;
claParams.num_iterations = 5;
%claParams.subSample = 400000;
%claParams.balancedsubSample = 1;
%claParams.testsetnum = 1;
claParams.segmentDescriptors = {'centered_mask_sp','bb_extent','pixel_area',...'centered_mask', %Shape
    'absolute_mask','top_height',...'bottom_height', %Location
    'int_text_hist_mr','dial_text_hist_mr',...'top_text_hist_mr','bottom_text_hist_mr','right_text_hist_mr','left_text_hist_mr' %Texture
    'sift_hist_int_','sift_hist_dial','sift_hist_bottom','sift_hist_top','sift_hist_right','sift_hist_left'... %Sift
    'mean_color','color_std','color_hist','dial_color_hist',... %Color
    'color_thumb','color_thumb_mask','gist_int'};
claParams.K = K;
claParams.segSuffix = segSuffix;

classifiers = cell(length(K),length(Labels));
for i = find(UseClassifier)
    labelMask = cell(1);
    labelMask{1} = ones(size(Labels{i}))==1;
    classifiers(:,i) = TrainClassifier(HOMEDATA, HOMELABELSETS(i), trainFileList, trainIndex(i), Labels(i), labelMask, claParams);
end
if(~exist('UseGlobalSVM','var'))
    UseGlobalSVM = 1:length(HOMELABELSETS);
end

globalSVM = cell(length(Labels),1);
globalSVMT = cell(size(Labels));
globalSVMRaw = cell(length(Labels),1);
if(exist('testParams','var') && isfield(testParams,'SVMDescs'))
    [globalSVMt globalSVMTt globalSVMRawt]= TrainGlobalSVM(HOMEDATA, HOMELABELSETS(UseGlobalSVM), Labels(UseGlobalSVM), trainIndex(UseGlobalSVM), trainGlobalDesc, valIndex(UseGlobalSVM), valGlobalDesc,testParams.SVMDescs, testSetNum, .9);
    globalSVM(UseGlobalSVM) = globalSVMt;
    globalSVMT(UseGlobalSVM) = globalSVMTt; 
    clear globalSVMRawt  globalSVMRawt globalSVMt;
end


if(exist('testParams','var') && isfield(testParams,'SVMSoftMaxDescs') && ~isempty(UseGlobalSVM))
    globalSVM = cell(length(Labels),length(testParams.SVMSoftMaxDescs));
    for i = 1:length(testParams.SVMSoftMaxDescs)
        [globalSVMt]= TrainGlobalSVM(HOMEDATA, HOMELABELSETS(UseGlobalSVM), Labels(UseGlobalSVM), trainIndex(UseGlobalSVM), trainGlobalDesc, valIndex(UseGlobalSVM), valGlobalDesc, testParams.SVMSoftMaxDescs(i), testSetNum);
        globalSVM(UseGlobalSVM,i) = globalSVMt;
        clear globalSVMRawt  globalSVMRawt globalSVMt;
    end
    SVMSoftMaxCutoff = rocNormSoftMax(HOMELABELSETS,globalSVM,valIndex);
end

clear indConverter l counts testMaskTemp segIndex fileList testMask testSetFile testFiles testMask testName;

loadDone = true;
