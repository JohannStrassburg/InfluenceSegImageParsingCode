HOME = 'D:\im_parser\CamVid';
HOMECODE = 'D:\P4\jtighe_localhost\im_parser\Release';
HOMEIMAGES = fullfile(HOME,'Images');
HOMEIMAGESALL = fullfile(HOME,'ImagesAll');
HOMELABELSETS = {fullfile(HOME,'LabelsSemanticSimple')};
HOMEDATA = fullfile(HOME,'Data');

UseClassifier = [0];
UseGlobalSVM = [];
UseLabelSet = 1;
K=200;

%ComputeOpticalFlow(HOMEIMAGESALL,fullfile(HOMEDATA,'Descriptors'));

%fileList = dir_recurse(fullfile(HOMEIMAGES,'*'),0);
%descFuns = ComputeSegmentDescriptors( fileList, HOMEIMAGES, HOMEDATA,HOMEDATA, HOMECODE, 1, 1:length(fileList), K,'_flow');
%descFuns = ComputeSegmentDescriptors( fileList, HOMEIMAGES, HOMEDATA,HOMEDATA, HOMECODE, 1, 1:length(fileList), K,'_flow_raw');