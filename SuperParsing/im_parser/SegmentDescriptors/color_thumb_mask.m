function [ desc ] = color_thumb_mask( im, mask, maskCrop, varargin )
%COLOR_THUMB_MASK Summary of this function goes here
%   Detailed explanation goes here

[y x] = find(maskCrop);
maskCrop = maskCrop(min(y):max(y), min(x):max(x));
im = im(min(y):max(y), min(x):max(x),:);
im = im.*uint8(repmat(maskCrop,[1 1 3]));
desc = imresize(im,[8 8]);