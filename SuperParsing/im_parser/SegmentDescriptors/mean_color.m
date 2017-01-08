function [ desc ] = mean_color( im, mask, maskCrop, bb, varargin )
%MEAN_COLOR Summary of this function goes here
%   Detailed explanation goes here

desc = zeros(3,1);
for c = 1:3
    r = im(:,:,c);
    desc(c) = mean(r(maskCrop));
end