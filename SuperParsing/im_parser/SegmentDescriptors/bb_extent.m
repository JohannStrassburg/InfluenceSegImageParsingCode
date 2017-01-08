function [ desc ] = bb_extent( im, mask, maskCrop, bb, varargin )
%BB_EXTENT Summary of this function goes here
%   Detailed explanation goes here

[y x] = find(maskCrop);
desc = [(max(y+bb(1))-min(y+bb(1)))/size(mask,1) (max(x+bb(3))-min(x+bb(3)))/size(mask,2)];