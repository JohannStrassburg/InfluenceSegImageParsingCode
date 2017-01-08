%clear;
HOME = 'D:\im_parser\LMSun';

HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'LabelsGeo'),fullfile(HOME,'LabelsSemantic')};
HOMEDATA = fullfile(HOME,'Data');
if(~exist('HOMECODE','var'))
    HOMECODE = pwd;
end

UseClassifier = [1 0];%];%
UseGlobalSVM = [];
UseLabelSet = [1 2];%6;%
K=200;
inoutEval = 1;

claParams.stopval = .001;
claParams.num_iterations = 100;
claParams.subSample = 600000;
claParams.balancedsubSample = 1;
claParams.testsetnum = 1;
claParams.init_weight = 'cFreq';

testSetNum = 1;
preset=1;
if(~exist('loadDone','var'))
    clear testParams;
    %testParams.SVMDescs = {'colorGist','coHist','spatialPry'};
    LoadData;
end
loadDone = true;

if(exist('rangeN','var'))
    testParams.range=rangeN(1):rangeN(2):length(testFileList);
end

testParams.K = K;
testParams.segmentDescriptors = {'centered_mask_sp','bb_extent','pixel_area',...'centered_mask', %Shape
    'absolute_mask','top_height',...'bottom_height', %Location
    'int_text_hist_mr','dial_text_hist_mr',...'top_text_hist_mr','bottom_text_hist_mr','right_text_hist_mr','left_text_hist_mr' %Texture
    'sift_hist_int_','sift_hist_dial','sift_hist_bottom','sift_hist_top','sift_hist_right','sift_hist_left'... %Sift
    'mean_color','color_std','color_hist','dial_color_hist',... %Color
    'color_thumb','color_thumb_mask','gist_int'}; %Appearance
testParams.TestString = 'Base';
testParams.MRFFold = 'MRF';
testParams.globalDescriptors = {'spatialPryScaled','colorGist','coHist'};
testParams.globalDescSuffix = '-SPscGistCoHist';
testParams.targetNN = 80;
testParams.smoothingConst = 1;
testParams.probType = 'ratio';
testParams.Rs = Rs;
testParams.retSetSize = [200];
testParams.ExpansionSize = [20];
testParams.minSPinRetSet = 1500;
testParams.LabelSmoothing = [0];
testParams.LabelPenality = {'conditional'};%,'pots''conditional'
testParams.InterLabelSmoothing = [0];
testParams.InterLabelPenality = {'pots'};%,'conditional','metric'
testParams.BConstCla = [0; .5];
testParams.BConst = [0; .1];
testParams.NormType = 'B.5.1';
testParams.maxPenality = 1000;
testParams.BKPenality = testParams.maxPenality;%testParams.maxPenality;
testParams.ExcludeFB = 0;
testParams.PixelMRF = 0;
testParams.fgFixed = 0;
testParams.SVMType = 'SVMVal';%'SVMTrain';%'SVMTest';%'SVMRaw1.5';%
testParams.weightBySize = 1;
testParams.colorModel =  testParams.maxPenality*[0];
testParams.colorSpace = 'lab';
testParams.clType = {'kmeans'};%,'gmm'
testParams.numClusters = [40];
testParams.edgeType = {'norm'}; %,'canny','bse'
testParams.connected = 8;
testParams.fgDataWeight = [100];
testParams.edgeParam = [11];%5 5
testParams.smoothData = [0];
testParams.segSuffix = segSuffix;

classifierTemp = classifiers;classifierTemp(0==UseClassifier)={[]};
globalSVMTemp = cell(size(globalSVM));globalSVMTemp(UseGlobalSVM,:)=globalSVM(UseGlobalSVM,:);

ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
EvaluateInOutDoor;

%{
pgd = testParams.globalDescriptors;
pgds = testParams.globalDescSuffix;
testParams.TestString = 'RetrievalSetDescTests';
testParams.globalDescriptors = {'spatialPryScaled'};
testParams.globalDescSuffix = 'SPScld';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
testParams.globalDescriptors = {'colorGist'};
testParams.globalDescSuffix = 'cGIST';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
testParams.globalDescriptors = {'coHist'};
testParams.globalDescSuffix = 'cHist';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
testParams.globalDescriptors = {'tinyIm'};
testParams.globalDescSuffix = 'tinyIm';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
testParams.globalDescriptors = {'spatialPryScaled','colorGist'};
testParams.globalDescSuffix = 'Combo-SPscGist';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
testParams.globalDescriptors = {'spatialPryScaled','colorGist','coHist'};
testParams.globalDescSuffix = 'Combo-SPscGistCoHist';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
testParams.globalDescriptors = {'spatialPryScaled','colorGist','coHist','tinyIm'};
testParams.globalDescSuffix = 'Combo-SPscGistCoHistTiny';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
EvaluateInOutDoor;
testParams.globalDescriptors = pgd;
testParams.globalDescSuffix = pgds;
%}

%{
pr = testParams.retSetSize;
testParams.TestString = 'RetrievalSetSizeTest';
for i = [1600]
    testParams.retSetSize = i;
    ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
    EvaluateInOutDoor;
end
testParams.retSetSize = pr;
%}

%{
ptn = testParams.targetNN;
psc = testParams.smoothingConst;
testParams.TestString = 'RadiusTest';
for i = 160
    testParams.targetNN = i;
    for j = [.2 .4 .8 1 2 4 8]
        testParams.smoothingConst = j;
        ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
    end
end
EvaluateInOutDoor;
testParams.targetNN = ptn;
testParams.smoothingConst = psc;
%}

%{
pls = testParams.LabelSmoothing;
testParams.TestString = 'SmoothingTest';
testParams.LabelSmoothing = [0 2 4 8 16 32 64 100 128 256 512];
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
EvaluateInOutDoor;
testParams.LabelSmoothing = pls;
%}

%{
pls = testParams.LabelSmoothing;
pils = testParams.InterLabelSmoothing;
testParams.TestString = 'FinalTest-Timing';
testParams.LabelSmoothing = [256];
testParams.InterLabelSmoothing = 256;%[64 100 128 256 512 768 1024 1280 1536 1792];
timing = ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
EvaluateInOutDoor;
testParams.LabelSmoothing = pls;
testParams.InterLabelSmoothing = pils;
%}

%{
testParams.TestString = 'InOutSVMTests';
testParams.RetrievalMetaMatch = 'GroundTruth';
if(~isfield(testParams,'TrainMetadata'))
    testParams.TrainMetadata = [];
    for i = 1:length(trainFileList)
        [fold base] = fileparts(trainFileList{i});
        load(fullfile(HOME,'Metadata',fold,[base '.mat'])); %metaData
        metaFields = fieldnames(metaData);
        for f = 1:length(metaFields)
            testParams.TrainMetadata.(metaFields{f}){i} = metaData.(metaFields{f});
        end
    end
end
if(~isfield(testParams,'TestMetadata'))
    testParams.TestMetadata = [];
    for i = 1:length(testFileList)
        [fold base] = fileparts(testFileList{i});
        load(fullfile(HOME,'Metadata',fold,[base '.mat'])); %metaData
        metaFields = fieldnames(metaData);
        for f = 1:length(metaFields)
            testParams.TestMetadata.(metaFields{f}){i} = metaData.(metaFields{f});
        end
    end
end
%metaFields = fieldnames(testParams.TestMetadata);
%for f = 1:length(metaFields)
%end
testParams.TestMetadataSVM = TrainLinearSVMForMetaData(testParams.TrainMetadata,testParams.TestMetadata,trainGlobalDesc, testGlobalDesc,{'spatialPryScaled','colorGist','coHist'});
testParams.RetrievalMetaMatch = 'SVM';
testParams.LabelSmoothing = [0];
%testParams.InterLabelSmoothing = [64 100 128 256 512 768 1024 1280 1536 1792];
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
EvaluateInOutDoor;
testParams.RetrievalMetaMatch = '';
%}

%{
testParams.TestString = 'DescTest';
ParseTestImagesDescEval(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
EvaluateInOutDoor;
%}

%{
testParams.TestString = 'DescTestFixedOrder';
pls = testParams.LabelSmoothing;
pils = testParams.InterLabelSmoothing;
testParams.LabelSmoothing = 256;
testParams.InterLabelSmoothing = [0 256];
testParams.segmentDescOrder = {'sift_hist_dial','top_height','dial_color_hist','sift_hist_bottom','mean_color','gist_int','sift_hist_right','sift_hist_top','int_text_hist_mr','color_thumb_mask',...
    'color_thumb','color_std','sift_hist_left','absolute_mask','dial_text_hist_mr','pixel_area','sift_hist_int_','bb_extent','color_hist','centered_mask_sp'};
ParseTestImagesOrderDesc(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
EvaluateInOutDoor;
testParams.LabelSmoothing = pls;
testParams.InterLabelSmoothing = pils;
%}

%{
st = testParams.SVMType;
testParams.TestString = 'ShortList';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
testParams.SVMType = 'SVMTop10';
fakeSVM = cell(size(UseLabelSet)); for i = 1:length(fakeSVM); fakeSVM{i} = i; end
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),fakeSVM,testParams);
EvaluateInOutDoor;
testParams.SVMType = 'SVMPerf';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),fakeSVM,testParams);
EvaluateInOutDoor;
testParams.SVMType = st;
%}
