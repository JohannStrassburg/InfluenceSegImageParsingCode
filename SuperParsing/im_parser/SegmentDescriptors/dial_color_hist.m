function [ desc ] = dial_color_hist( im, mask, maskCrop, bb, centers, textons, borders, varargin )
%COLOR_HIST Summary of this function goes here
%   Detailed explanation goes here

%strEl = strel('square',20);
%mask = imdilate(mask,strEl,'same');
mask=borders(:,:,5);

desc = [];
numBins = 8;
binSize = 256/numBins;
binCenters = (binSize-1)/2:binSize:255;
for c = 1:3
    r = im(:,:,c);
    desc = [desc hist(double(r(mask)),binCenters)/sum(mask(:))];
end