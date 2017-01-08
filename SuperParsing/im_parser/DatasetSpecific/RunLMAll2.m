%clear;
HOME = 'D:\jtighe\im_parser\LMAllSmall2';

HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'GeoLabels'),fullfile(HOME,'Segments')};%,fullfile(HOME,'TopLevelLabels'),fullfile(HOME,'BackgroundLabels'),fullfile(HOME,'ForegroundLabels')
HOMEDATA = fullfile(HOME,'Data');
HOMECODE = 'D:\jtighe\Perforce\im_parser\Release';

%LoadData;

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
    testParams.LabelSmoothing = [0 2 4 8];
    testParams.LabelPenality = {'metric'};%,'pots''conditional'
    testParams.InterLabelSmoothing = [0 2 4 8 16 32];
    testParams.InterLabelPenality = {'conditional'};%'conditional','pots',
    testParams.LabelSetWeights = ones(length(Labels),1);
    
testParams.TestString = 'SemGeo';
testParams.SVMDescs = {'colorGist','coHist','spatialPry'};%
testParams.MinClusterSize = 5;

%testParams.labelSubSets = labelSubSets;
testParams.labelSubSetsWeights = [0];

testParams.SLThresh = 0;
testParams.maxPenality=100;

testParams.BConstCla = [0; .5];
for b = [.1]
    testParams.BConst = [0; b];
    testParams.NormType = sprintf('BConst%.2f',testParams.BConst(2));

    timing = ParseTestImages(HOMEDATA,HOMEDATA,HOMELABELSETS,testFileList,trainFileList,trainGlobalDesc,testGlobalDesc,trainIndex,trainCounts,labelPenality,Labels,classifiers,globalSVM,testParams);
end
EvaluateTests(HOMEDATA,HOMELABELSETS,{testParams.TestString});

%{
EvaluateTests(HOMEDATA,HOMELABELSETS,{testParams.TestString});
clear metadata; metadata.inOutDoor=1;
EvaluateTests(HOMEDATA,HOMELABELSETS,{testParams.TestString},' Indoor',metadata);
clear metadata; metadata.inOutDoor=2;
EvaluateTests(HOMEDATA,HOMELABELSETS,{testParams.TestString},' Outdoor',metadata);

%}