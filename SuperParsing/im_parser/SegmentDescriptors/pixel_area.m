function [ desc ] = pixel_area( im, mask, maskCrop, varargin )
%PIXEL_AREA Summary of this function goes here
%   Detailed explanation goes here
desc = sum(maskCrop(:))/numel(mask);