function [ desc ] = top_height( im, mask, maskCrop, bb, varargin )
%TOP_HEIGHT Summary of this function goes here
%   Detailed explanation goes here

[y,x] = find(maskCrop>0);
desc = min(y+bb(1))/size(mask,1);