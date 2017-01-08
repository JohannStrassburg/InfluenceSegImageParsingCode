function [desc,t] = ComputeGlobalDescriptors(images, HOMEIMAGES, HOMELABELSETS, HOMEDATA )
HOMEDATA=fullfile(HOMEDATA,'Descriptors','Global');
t = zeros(6,1);
numIm = length(images);
tic
%desc.labelHist = BuildLabelHist(images(1:numIm), HOMELABELSETS, fullfile(HOMEDATA,'labelHist'),1);
%befor you uncomment this you now need to normalize the histogram as this function no longer returns a normalized histogram
t(6) = toc./numIm;

%desc.parserHist = BuildLabelHist( images(1:numIm), {'D:\jtighe\im_parser\MSRC\ClassifierOutput'}, fullfile(HOMEDATA,'ParsingLabelHist'), 1 );
%desc.parserHist = desc.parserHist{1};

params.maxImageSize = 1000;
params.gridSpacing = 1;
params.patchSize = 16;
params.dictionarySize = 200;
params.numTextonImages = 1000;
params.pyramidLevels = 3;
tic
desc.spatialPryScaled = BuildPyramid(images(1:numIm), HOMEIMAGES, fullfile(HOMEDATA,'SpatialPyrDenseScaled'), params, 1, 0 );
t(1) = toc./numIm;
tic
params.gridSpacing = 8;
%desc.spatialPry8Scaled = BuildPyramid(images(1:numIm), HOMEIMAGES, fullfile(HOMEDATA,'SpatialPyrScaled'), params, 1, 0 );
t(2) = toc./numIm;
tic
desc.colorGist = BuildGist(images(1:numIm), HOMEIMAGES, fullfile(HOMEDATA,'colorGist'),1,1);
t(3) = toc./numIm;
tic
%desc.tinyIm = BuildTinyIm(images(1:numIm), HOMEIMAGES, fullfile(HOMEDATA,'tinyIm'),1);
t(4) = toc./numIm;
tic
desc.coHist = BuildColorHist(images(1:numIm), HOMEIMAGES, fullfile(HOMEDATA,'coHist'),1)./3;
t(5) = toc./numIm;

end
