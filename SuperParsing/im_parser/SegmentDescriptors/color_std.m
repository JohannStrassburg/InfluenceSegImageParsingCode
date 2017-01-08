function [ desc ] = color_std( im, mask, maskCrop, varargin )
%COLOR_STD Summary of this function goes here
%   Detailed explanation goes here

desc = zeros(3,1);
for c = 1:3
    r = im(:,:,c);
    desc(c) = std(double(r(maskCrop)));
end