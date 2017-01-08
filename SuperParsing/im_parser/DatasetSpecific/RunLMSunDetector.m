%clear;
HOME = 'D:\im_parser\LMSun';

HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'LabelsGeo'),fullfile(HOME,'LabelsSemantic')};
HOMEDATA = fullfile(HOME,'Data');
HOMECODE = 'D:\P4\jtighe_localhost\im_parser\Release';

UseClassifier = [0 0];%];%
UseGlobalSVM = [];
UseLabelSet = [2];%6;%
K=200;
inoutEval = 1;

%testSetName = 'SmallDetectorTest';
%testSetName = 'DetectorTestSet';
claParams.stopval = .001;
claParams.num_iterations = 100;
claParams.subSample = 600000;
claParams.balancedsubSample = 1;
claParams.testsetnum = 1;
claParams.init_weight = 'cFreq';
useGlobal = 0;

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
testParams.TestString = 'DetectorPreTrained';
testParams.MRFFold = 'ML';
testParams.globalDescriptors = {'spatialPryScaled','colorGist','coHist'};
testParams.globalDescSuffix = '-SPscGistCoHist';
testParams.targetNN = 80;
testParams.smoothingConst = 1;
testParams.probType = 'ratio';
testParams.Rs = Rs;
testParams.retSetSize = [800];
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
testParams.PixelMRF = 1;
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

testParams.ModelFold = 'Model20';
testParams.MaxModelPerCls = 100;
testParams.NMS = 0;
stuffLabels = importdata(fullfile(HOME,'StuffLabels.txt'));
testParams.StuffLabels = logical(zeros(size(Labels{2})));
for i = 1:length(stuffLabels)
    testParams.StuffLabels(strcmp(stuffLabels{i},Labels{2})) = true;
end

classifierTemp = classifiers;classifierTemp(0==UseClassifier)={[]};
globalSVMTemp = cell(size(globalSVM));globalSVMTemp(UseGlobalSVM,:)=globalSVM(UseGlobalSVM,:);

%max_mined = 20; ParseImWDetectorScript;
%max_mined = 10; TrainAndRunTest;
%max_mined = 20; TrainAndRunTest;
%max_mined = 100; TrainAndRunTest;

%ParseTestImagesWDetectors(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
%testParams.NMS = 1;
%ParseTestImagesWDetectors(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
testParams.NMS = 0;
testParams.ModelFold = 'Model1280';
ParseTestImagesWDetectors(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
%testParams.NMS = 1;
%ParseTestImagesWDetectors(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);
EvaluateInOutDoor;
