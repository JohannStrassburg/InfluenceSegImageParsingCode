%clear;
HOME = 'D:\im_parser\MSRC';

HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'LabelsGeo'),fullfile(HOME,'LabelsSemanticReduced')};
HOMEDATA = fullfile(HOME,'Data');
HOMECODE = 'D:\P4\jtighe_localhost\im_parser\Release';

UseClassifier = [1 1];
UseGlobalSVM = [];
UseLabelSet = [2];

testSetNum = 1;
claParams.num_iterations = 100;
claParams.subSample = 0;
claParams.balancedsubSample = 1;
claParams.testsetnum = testSetNum;
claParams.stopval = .00001;
claParams.init_weight = '';

preset=1;
if(~exist('loadDone','var'))
    clear testParams;
    %testParams.SVMDescs ={'parserHist'};
    %testParams.SVMDescs = {'colorGist','coHist','spatialPry8'};
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
testParams.TestString = 'CL Test';
testParams.MRFFold = ['MRF2'];
testParams.globalDescriptors = {'spatialPryScaled','colorGist','coHist'};
testParams.globalDescSuffix = '-SPscGistCoHist';
testParams.targetNN = 20;
testParams.smoothingConst = 1;
testParams.probType = 'ratio';
testParams.Rs = Rs;
testParams.retSetSize = [200];
testParams.ExpansionSize = [20];
testParams.minSPinRetSet = 1500;
%testParams.LabelSmoothing = {[8; 8; 8; 8],[8; 16; 8; 8],[8; 64; 8; 8]};
testParams.LabelSmoothing = [0];
testParams.LabelPenality = {'conditional'};%,'pots''conditional'
testParams.InterLabelSmoothing = [0];
testParams.InterLabelPenality = {'pots'};%,'conditional','metric'
testParams.BConstCla = [0; .5];
testParams.BConst = [0; .1];
testParams.NormType = 'B.5.1';
testParams.CLSuffix = sprintf('N%03dS%.03f%s',claParams.num_iterations, claParams.stopval,claParams.init_weight);
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
testParams.segSuffix = segSuffix;

classifierTemp = classifiers;classifierTemp(0==UseClassifier)={[]};
globalSVMTemp = cell(size(globalSVM)); globalSVMTemp(UseGlobalSVM) = globalSVM(UseGlobalSVM);

%testParams.SVMType = 'SVMLinear';
%testParams.SVMOutput =  TrainLinearSVM(HOMEDATA, HOMELABELSETS, Labels, trainIndex, trainGlobalDesc, testIndex, testGlobalDesc,{'spatialPryScaled','colorGist','coHist'});
for i = find(UseClassifier)
    labelMask = cell(1);
    labelMask{1} = ones(size(Labels{i}))==1;
    classifiers(:,i) = TrainRandForest(HOMEDATA, HOMELABELSETS(i), trainFileList, trainIndex(i), Labels(i), labelMask, claParams);
end
        
for ni = 100%20:20:100
    for s = .001%10.^(-1:-1:-3)
        claParams.num_iterations = ni;
        claParams.stopval = s;
        classifiers = cell(length(K),length(Labels));
        for i = find(UseClassifier)
            labelMask = cell(1);
            labelMask{1} = ones(size(Labels{i}))==1;
            classifiers(:,i) = TrainClassifier(HOMEDATA, HOMELABELSETS(i), trainFileList, trainIndex(i), Labels(i), labelMask, claParams);
            for l = 1:length(classifiers{1,i}.h0)
                pc = sum(trainIndex{i}.label==l)./length(trainIndex{i}.label);
                classifiers{1,i}.h0(l) = log((1-pc)/pc);
            end
            fprintf('%d trees\n',size(classifiers{1,i}.wcs,1));
        end
        testParams.CLSuffix = sprintf('BalN%03dS%.03f%s',claParams.num_iterations, claParams.stopval,claParams.init_weight);
        ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS(UseLabelSet),testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex(UseLabelSet),trainCounts(UseLabelSet),labelPenality(UseLabelSet,UseLabelSet),Labels(UseLabelSet),classifiers(UseLabelSet),globalSVMTemp(UseLabelSet),testParams);
    end
end

%EvaluateTests(HOMEDATA,HOMELABELSETS(UseLabelSet(SetupRange(rangeN(1),rangeN(2),length(UseLabelSet)))),{testParams.TestString},testParams.MRFFold,[],[],testParams.MRFFold);
EvaluateTests(HOMEDATA,HOMELABELSETS(UseLabelSet),{testParams.TestString},testParams.MRFFold,[],[],testParams.MRFFold);