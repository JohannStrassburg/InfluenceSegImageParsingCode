%clear;
HOME = 'D:\jtighe\im_parser\LMAllSmall';

HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'GeoLabels'),fullfile(HOME,'SemanticLabels')};
HOMEDATA = fullfile(HOME,'Data');
HOMECODE = 'D:\jtighe\Perforce\im_parser\Release';

LoadData;

clear testParams;
testParams.globalDescriptors = {'spatialPry','colorGist','coHist'};%,'tinyIm',
testParams.retSetSize = [400];
testParams.ExpansionSize = [20];
testParams.minSPinRetSet = 1500;
testParams.K = K;
%we don't end up using all the descriptors we generate
testParams.segmentDescriptors = {'centered_mask_sp','bb_extent','pixel_area',...'centered_mask', %Shape
    'absolute_mask','top_height',...'bottom_height', %Location
    'int_text_hist_mr','dial_text_hist_mr',...'top_text_hist_mr','bottom_text_hist_mr','right_text_hist_mr','left_text_hist_mr' %Texture
    'sift_hist_int_','sift_hist_dial','sift_hist_bottom','sift_hist_top','sift_hist_right','sift_hist_left'... %Sift
    'mean_color','color_std','color_hist','dial_color_hist',... %Color
    'color_thumb','color_thumb_mask','gist_int'}; %Appearance
testParams.targetNN = 20;
testParams.Rs = Rs;
%{-
    testParams.LabelSmoothing = [0 1 2 4];
    testParams.LabelPenality = {'metric'};%,'conditional''pots'
    testParams.InterLabelSmoothing = [0 1 2];
    testParams.InterLabelPenality = {'conditional'};%'conditional','pots',
    testParams.LabelSetWeights = ones(length(Labels),1);
    
testParams.TestString = 'Nero';
testParams.SVMDescs = {'colorGist','coHist','spatialPry'};%
testParams.MinClusterSize = 5;

testParams.labelSubSets = labelSubSets;
testParams.labelSubSetsWeights = [0];
%testParams.SVMDescs = {'colorGist','coHist','spatialPry'};
%testParams.SLThresh = -1.0;

%{
if(~exist('fullSPDesc','var'))
    usedRetInds = FindAllRetInds(HOMEDATA,testFileList,trainGlobalDesc,testGlobalDesc,testParams);
    requiredTrainIndex = cell(size(trainIndex));
    for i = 2
        requiredTrainIndex{i} = PruneIndex(trainIndex{i},usedRetInds,length(usedRetInds),0);
        fullSPDesc{i,1} = LoadSegmentDesc(trainFileList,requiredTrainIndex{i},HOMEDATA,testParams.segmentDescriptors,testParams.K(1));
    end
    end
%}

%[results bestR svmR svmPick bestPick resultsExp bestRExp expRExp bestPickExp expPickExp] = ParseTestImagesRetCluster(HOMEDATA,HOMEDATA,HOMEIMAGES,HOMELABELSETS,testFileList,trainFileList,trainGlobalDesc,testGlobalDesc,trainIndex,testIndex,trainCounts,labelPenality,Labels,testParams);
%ParseTestImagesRetClusterPerfSL(HOMEDATA,HOMEDATA,HOMEIMAGES,HOMELABELSETS,testFileList,trainFileList,trainGlobalDesc,testGlobalDesc,trainIndex,testIndex,trainCounts,labelPenality,Labels,testParams);
timing = ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS,testFileList,trainFileList,trainGlobalDesc,testGlobalDesc,trainIndex,trainCounts,labelPenality,Labels,classifiers,globalSVM,testParams);

EvaluateTests(HOMEDATA,HOMELABELSETS,{testParams.TestString});
clear metadata; metadata.inOutDoor=1;
EvaluateTests(HOMEDATA,HOMELABELSETS,{testParams.TestString},' Indoor',metadata);
clear metadata; metadata.inOutDoor=2;
EvaluateTests(HOMEDATA,HOMELABELSETS,{testParams.TestString},' Outdoor',metadata);

