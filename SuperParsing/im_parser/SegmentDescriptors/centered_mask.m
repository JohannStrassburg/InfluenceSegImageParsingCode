function [ desc ] = centered_mask( im, mask, maskCrop, varargin )
%CENTERED_MASK Summary of this function goes here
%   Detailed explanation goes here

[y x] = find(maskCrop);
maskCrop = maskCrop(min(y):max(y), min(x):max(x));
[h,w] = size(maskCrop);
padAmount = fix((h-w)/2);
maskCrop = padarray(maskCrop,[max(-padAmount,0) max(padAmount,0)],'both');
desc = max(imresize(double(maskCrop),[32 32]),0);
desc(desc>1) = 1;desc(desc<0) = 0;
