HOME = 'D:\im_parcer\siftFlow';

HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'GeoLabels'),fullfile(HOME,'SemanticLabels')};
HOMEDATA = fullfile(HOME,'Data');

HOMETEST = 'D:\im_parcer\urbanScape';
HOMETESTIMAGES = fullfile(HOMETEST,'Images');
HOMETESTDATA = fullfile(HOMETEST,'Data');

WebTestList = {'R200 K200 S0 IS0 Pcon IPpot','R200 K200 S1 IS0 Pcon IPpot','R200 K200 S2 IS0 Pcon IPpot','R200 K200 S4 IS0 Pcon IPpot','R200 K200 S8 IS0 Pcon IPpot'};
WebTestName = {'Base','MRF Smoothing 1','MRF Smoothing 2','MRF Smoothing 4','MRF Smoothing 8'};

RunFullSystem;

