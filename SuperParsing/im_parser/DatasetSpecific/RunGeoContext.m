%clear;
HOME = 'D:\jtighe\im_parser\GeoContext';

HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'LabelsGeo'),fullfile(HOME,'LabelsSemantic')};
HOMEDATA = fullfile(HOME,'Data');
HOMECODE = 'D:\jtighe\Perforce\im_parser\Release';

UseClassifier = [1 1];
UseGlobalSVM = [];
UseLabelSet = [1 2];

claParams.num_iterations = 100;
claParams.subSample = 0;
claParams.balancedsubSample = 1;
claParams.testsetnum = 1;
K=0;

testSetNum = 1;
preset=1;
if(~exist('loadDone','var'))
    clear testParams;
    %testParams.SVMDescs = {'colorGist','coHist','spatialPry'};
    LoadData;
end
loadDone = true;

if(exist('rangeN','var'))
    testParams.range=SetupRange(rangeN(1),rangeN(2));
end

testParams.K = K;
testParams.segmentDescriptors = {'centered_mask_sp','bb_extent','pixel_area',...'centered_mask', %Shape
    'absolute_mask','top_height',...'bottom_height', %Location
    'int_text_hist_mr','dial_text_hist_mr',...'top_text_hist_mr','bottom_text_hist_mr','right_text_hist_mr','left_text_hist_mr' %Texture
    'sift_hist_int_','sift_hist_dial','sift_hist_bottom','sift_hist_top','sift_hist_right','sift_hist_left'... %Sift
    'mean_color','color_std','color_hist','dial_color_hist',... %Color
    'color_thumb','color_thumb_mask','gist_int'}; %Appearance
testParams.TestString = 'FullBar';% ['BaseTest' num2str(testSetNum)];
testParams.MRFFold = 'MRF';
testParams.globalDescriptors = {'spatialPry8','colorGist','coHist','tinyIm'};
testParams.targetNN = 20;
testParams.Rs = Rs;
testParams.retSetSize = [400];
testParams.ExpansionSize = [20];
testParams.minSPinRetSet = 1500;
testParams.LabelSmoothing = [0 8 16];%];
testParams.LabelPenality = {'conditional'};%'metric','pots'
testParams.InterLabelSmoothing = [0 4 8 16 32]; %
testParams.InterLabelPenality = {'conditional'};%'conditional','pots',
testParams.BConstCla = [0; .5];
testParams.BConst = [0; .05];
testParams.NormType = 'BConst.5.05';%'GeoClFromTighe';%
testParams.maxPenality=100;
testParams.BKPenality = 1;%testParams.maxPenality;
testParams.FGShift = 0.0;
testParams.SLThresh = 0.0;
testParams.GrabCutl1 = [.5];
testParams.GrabCutl2 = [5];
testParams.GrabCutFine = 1;
testParams.FGShift = 0;
testParams.ExcludeFB = 0;
testParams.PixelMRF = 0;
testParams.fgFixed = 0;
testParams.SVMType = 'SVMVal';%'SVMTrain';%'SVMTest';%'SVMRaw1.5';%
testParams.weightBySize = 1;
testParams.colorModel =  0;
testParams.colorMaskFile = '';

classifierTemp = classifiers;classifierTemp(0==UseClassifier)={[]};

ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifierTemp(UseLabelSet),globalSVM(UseLabelSet),testParams);
%EvaluateTests(HOMEDATA,HOMELABELSETS(UseLabelSet(SetupRange(rangeN(1),rangeN(2),length(UseLabelSet)))),{testParams.TestString},testParams.MRFFold,[],[],testParams.MRFFold);
EvaluateTests(HOMEDATA,HOMELABELSETS(UseLabelSet),{testParams.TestString},testParams.MRFFold,[],[],testParams.MRFFold);