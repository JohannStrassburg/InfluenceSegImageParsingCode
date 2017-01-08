function [ desc ] = bottom_height( im, mask, maskCrop, bb, varargin )
%BOTTOM_HEIGHT Summary of this function goes here
%   Detailed explanation goes here

[y,x] = find(maskCrop>0);
desc = max(y+bb(1))/size(mask,1);