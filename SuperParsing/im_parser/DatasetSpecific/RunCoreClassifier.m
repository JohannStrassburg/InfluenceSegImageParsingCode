
HOME = 'D:\im_parcer\Core';
HOMECODE = 'D:\Perforce\im_parser\Release';
HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'LabelsForgroundBK'),fullfile(HOME,'LabelsAnimalVehicle'),fullfile(HOME,'LabelsSemantic'),fullfile(HOME,'LabelsMaterial'),fullfile(HOME,'LabelsParts')};%};%
HOMEDATA = fullfile(HOME,'Data');
UseClassifier = [1 1 1 1 1];%];%
UseGlobalSVM = [];%[2 3];%
UseLabelSet = [2 3 4 5];% 

claParams.num_iterations = 100;
claParams.subSample = 0;
claParams.balancedsubSample = 0;
claParams.testsetnum = 6;
claParams.stopval = .01;


testSetNum = 7;
preset=1;
if(~exist('loadDone','var'))
    clear testParams;
    testParams.SVMDescs = {'colorGist','coHist','spatialPry'};
    %testParams.SVMSoftMaxDescs = {'colorGist','coHist','spatialPry'};
    LoadData;
end
loadDone = true;


if(exist('rangeN','var'))
    testParams.range = SetupRange(rangeN(1),rangeN(2));
    %temp = [6 7 46 70 76 105 123 137 170 183 206 218];
    %testParams.range = temp(SetupRange(rangeN(1),rangeN(2),length(temp)));
end

testParams.K = K;
testParams.segmentDescriptors = {'centered_mask_sp','bb_extent','pixel_area',...'centered_mask', %Shape
    'absolute_mask','top_height',...'bottom_height', %Location
    'int_text_hist_mr','dial_text_hist_mr',...'top_text_hist_mr','bottom_text_hist_mr','right_text_hist_mr','left_text_hist_mr' %Texture
    'sift_hist_int_','sift_hist_dial','sift_hist_bottom','sift_hist_top','sift_hist_right','sift_hist_left'... %Sift
    'mean_color','color_std','color_hist','dial_color_hist',... %Color
    'color_thumb','color_thumb_mask','gist_int'}; %Appearance
testParams.TestString = 'CLStop01';
testParams.MRFFold = 'MRF-NoFG';
testParams.globalDescriptors = {'spatialPry','colorGist','coHist'};
testParams.targetNN = 20;
testParams.Rs = Rs;
testParams.retSetSize = [400];
testParams.ExpansionSize = [20];
testParams.minSPinRetSet = 1500;
%testParams.LabelSmoothing = {[8; 8; 8; 8],[8; 16; 8; 8],[8; 64; 8; 8]};
testParams.LabelSmoothing = [0 8 ];
testParams.LabelPenality = {'conditional'};%,'pots''conditional'
testParams.InterLabelSmoothing = [0 16];
testParams.InterLabelPenality = {'conditional'};%,'pots','metric'
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
%testParams.colorMaskFile = 'BK1000 R400 C1 B.5.1 CM8000 lab kmeans NC40 S04 IS0.000 Pcon IPcon Enorm EP11 SD0 Pix FgFix0 WbS1 FW100 Cn8';
testParams.edgeType = {'norm'}; %,'canny','bse'
testParams.connected = 8;
testParams.fgDataWeight = [100];
testParams.edgeParam = [11];%5 5
testParams.smoothData = [0];
%testParams.SVMSoftMaxCutoff = SVMSoftMaxCutoff(UseLabelSet);

classifierTemp = classifiers;classifierTemp(0==UseClassifier)={[]};
globalSVMTemp = cell(size(globalSVM));globalSVMTemp(UseGlobalSVM,:)=globalSVM(UseGlobalSVM,:);

ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVMTemp(UseLabelSet,:),testParams);

EvaluateTests(HOMEDATA,HOMELABELSETS(UseLabelSet(SetupRange(rangeN(1),rangeN(2),length(UseLabelSet)))),{testParams.TestString},testParams.MRFFold,[],[],testParams.MRFFold);
%EvaluateTests(HOMEDATA,HOMELABELSETS(UseLabelSet),{testParams.TestString},testParams.MRFFold,[],[],testParams.MRFFold);
