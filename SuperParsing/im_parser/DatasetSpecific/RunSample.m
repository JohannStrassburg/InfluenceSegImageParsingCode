
if(~exist('HOMECODE','var'))
    HOMECODE = pwd;
end
HOME = fullfile(HOMECODE,'SampleDataSet');
HOMEIMAGES = fullfile(HOME,'Images');
HOMELABELSETS = {fullfile(HOME,'GeoLabels'),fullfile(HOME,'SemanticLabels')};
HOMEDATA = fullfile(HOME,'Data');

UseClassifier = [0 0];
UseGlobalSVM = [];
UseLabelSet = [1 2];
K=200;

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
