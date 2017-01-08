function [ desc ] = color_hist( im, mask, maskCrop, varargin )
%COLOR_HIST Summary of this function goes here
%   Detailed explanation goes here

desc = [];
numBins = 8;
binSize = 256/numBins;
binCenters = (binSize-1)/2:binSize:255;
for c = 1:3
    r = im(:,:,c);
    desc = [desc hist(double(r(maskCrop)),binCenters)/sum(maskCrop(:))];
end