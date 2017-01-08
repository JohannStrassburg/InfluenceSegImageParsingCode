
HOMEBASE = '/data/jstrassb/SuperParsing/Barcelona';
HOME = fullfile(HOMEBASE,'Experiments',EXPERIMENT);

HOMEIMAGES = fullfile(HOMEBASE,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'GeoLabels'),fullfile(HOME,'SemanticLabels')};%
HOMEDATA = fullfile(HOME,'Data');
HOMECODE = '/home/jstrassb/im_parser';

UseClassifier = [0 0];
UseGlobalSVM = [];
UseLabelSet = [2];
K=0;
inoutEval = 0;

claParams.stopval = .001;
claParams.num_iterations = 100;
claParams.subSample = 0;
claParams.balancedsubSample = 1;
claParams.testsetnum = 1;
claParams.init_weight = 'cFreq';

if(~exist('loadDone','var'))
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
testParams.CLSuffix = '';

classifierTemp = classifiers;classifierTemp(0==UseClassifier)={[]};
globalSVMTemp = cell(size(globalSVM));globalSVMTemp(UseGlobalSVM,:)=globalSVM(UseGlobalSVM,:);

if(~exist('fullSPDesc','var')); fullSPDesc = cell(length(HOMELABELSETS),length(K)); end
for i = find(~UseClassifier)
    if(isempty(fullSPDesc{i})); fullSPDesc{i} = LoadSegmentDesc(trainFileList,trainIndex{i},HOMEDATA,testParams.segmentDescriptors,testParams.K,testParams.segSuffix); end
end

ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams,fullSPDesc(UseLabelSet));
EvaluateInOutDoor;

%{
pgd = testParams.globalDescriptors;
pgds = testParams.globalDescSuffix;
testParams.TestString = 'RetrievalSetDescTests';
testParams.globalDescriptors = {'spatialPryScaled'};
testParams.globalDescSuffix = 'SPScld';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams,fullSPDesc(UseLabelSet));
testParams.globalDescriptors = {'colorGist'};
testParams.globalDescSuffix = 'cGIST';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams,fullSPDesc(UseLabelSet));
testParams.globalDescriptors = {'coHist'};
testParams.globalDescSuffix = 'cHist';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams,fullSPDesc(UseLabelSet));
testParams.globalDescriptors = {'tinyIm'};
testParams.globalDescSuffix = 'tinyIm';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams,fullSPDesc(UseLabelSet));
testParams.globalDescriptors = {'spatialPryScaled','colorGist'};
testParams.globalDescSuffix = 'Combo-SPscGist';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams,fullSPDesc(UseLabelSet));
testParams.globalDescriptors = {'spatialPryScaled','colorGist','coHist'};
testParams.globalDescSuffix = 'Combo-SPscGistCoHist';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams,fullSPDesc(UseLabelSet));
testParams.globalDescriptors = {'spatialPryScaled','colorGist','coHist','tinyIm'};
testParams.globalDescSuffix = 'Combo-SPscGistCoHistTiny';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams,fullSPDesc(UseLabelSet));
EvaluateInOutDoor;
testParams.globalDescriptors = pgd;
testParams.globalDescSuffix = pgds;
%}

%{
pr = testParams.retSetSize;
testParams.TestString = 'RetrievalSetSizeTest';
for i = [50 100 200 400 800 1600 length(trainFileList)]
    testParams.retSetSize = i;
    ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams,fullSPDesc(UseLabelSet));
end
EvaluateInOutDoor;
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
        ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams,fullSPDesc(UseLabelSet));
    end
end
EvaluateInOutDoor;
testParams.targetNN = ptn;
testParams.smoothingConst = psc;
%}

%{
pls = testParams.LabelSmoothing;
testParams.TestString = 'SmoothingTest';
testParams.LabelSmoothing = [0];% 2 4 8 16 32 64 100 128 256 512];
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams,fullSPDesc(UseLabelSet));
EvaluateInOutDoor;
testParams.LabelSmoothing = pls;
%}

%{
pls = testParams.LabelSmoothing;
pils = testParams.InterLabelSmoothing;
testParams.TestString = 'FinalTest';
testParams.LabelSmoothing = [100];
testParams.InterLabelSmoothing = [64 100 128];
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams,fullSPDesc(UseLabelSet));
EvaluateInOutDoor;
testParams.LabelSmoothing = pls;
testParams.InterLabelSmoothing = pils;
%}

%{
pls = testParams.LabelSmoothing;
pils = testParams.InterLabelSmoothing;
testParams.TestString = 'FinalTest-Timing';
testParams.LabelSmoothing = [100];
testParams.InterLabelSmoothing = [0 64 100 128];%[64 100 128 256 512 768 1024 1280 1536 1792];
timing = ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
testParams.LabelSmoothing = [0];
testParams.InterLabelSmoothing = [0];%[64 100 128 256 512 768 1024 1280 1536 1792];
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
EvaluateInOutDoor;
testParams.LabelSmoothing = pls;
testParams.InterLabelSmoothing = pils;
%}

%{
sd = testParams.segmentDescriptors;
pls = testParams.LabelSmoothing;
pils = testParams.InterLabelSmoothing;

testParams.TestString = 'SiftOnlyTest';
testParams.segmentDescriptors = {'sift_hist_dial'};
claParams.segmentDescriptors = testParams.segmentDescriptors;
claParams.testsetnum = 66;
classifierSift = classifierTemp;
for i = find(UseClassifier)
    labelMask = cell(1);
    labelMask{1} = ones(size(Labels{i}))==1;
    classifierSift(:,i) = TrainClassifier(HOMEDATA, HOMELABELSETS(i), trainFileList, trainIndex(i), Labels(i), labelMask, claParams);
end

testParams.LabelSmoothing = [0 100];
testParams.InterLabelSmoothing = [0 16 36 64 100 128 256 512];
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierSift(UseLabelSet),globalSVM(UseLabelSet),testParams,fullSPDesc(UseLabelSet));
EvaluateInOutDoor;

testParams.segmentDescriptors = sd;
testParams.LabelSmoothing = pls;
testParams.InterLabelSmoothing = pils;
%}

%{
testParams.TestString = 'DescTestTNN20';
ParseTestImagesDescEval(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams,fullSPDesc(UseLabelSet));
EvaluateInOutDoor;
%}

%{
testParams.TestString = 'DescTestFixedOrder';
pls = testParams.LabelSmoothing;
pils = testParams.InterLabelSmoothing;
testParams.LabelSmoothing = [0 100];
testParams.InterLabelSmoothing = [0 64];
testParams.segmentDescOrder = {'sift_hist_dial','dial_text_hist_mr','top_height','mean_color','sift_hist_top','dial_color_hist','sift_hist_left','pixel_area','color_thumb','centered_mask_sp','absolute_mask',...
    'sift_hist_right','color_thumb_mask','color_hist','sift_hist_int_','bb_extent','sift_hist_bottom','gist_int','int_text_hist_mr','color_std'};
ParseTestImagesOrderDesc(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams,fullSPDesc(UseLabelSet));
EvaluateInOutDoor;
testParams.LabelSmoothing = pls;
testParams.InterLabelSmoothing = pils;
%}

%{
st = testParams.SVMType;
testParams.TestString = 'ShortList';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams,fullSPDesc(UseLabelSet));
testParams.SVMType = 'SVMTop10';
fakeSVM = cell(size(UseLabelSet)); for i = 1:length(fakeSVM); fakeSVM{i} = i; end
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),fakeSVM,testParams,fullSPDesc(UseLabelSet));
EvaluateInOutDoor;
testParams.SVMType = 'SVMPerf';
ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),fakeSVM,testParams,fullSPDesc(UseLabelSet));
EvaluateInOutDoor;
testParams.SVMType = st;
%}
