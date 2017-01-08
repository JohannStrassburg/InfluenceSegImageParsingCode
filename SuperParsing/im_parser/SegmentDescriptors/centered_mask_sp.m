function [ desc ] = centered_mask_sp( im, mask, maskCrop, varargin )
%CENTERED_MASK Summary of this function goes here
%   Detailed explanation goes here

[y x] = find(maskCrop);
mask = maskCrop(min(y):max(y), min(x):max(x));
[h,w] = size(mask);
padAmount = fix((h-w)/2);
mask = padarray(mask,[max(-padAmount,0) max(padAmount,0)],'both');
desc = max(imresize(double(mask),[8 8]),0);
desc(desc>1) = 1;desc(desc<0) = 0;