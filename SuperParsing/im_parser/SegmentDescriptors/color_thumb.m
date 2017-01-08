function [ desc ] = color_thumb( im, mask, maskCrop, varargin )
%COLOR_THUMB Summary of this function goes here
%   Detailed explanation goes here

[y x] = find(maskCrop);
im = im(min(y):max(y), min(x):max(x),:);
desc = imresize(im,[8 8]);
