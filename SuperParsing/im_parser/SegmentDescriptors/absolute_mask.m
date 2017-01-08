function [ desc ] = absolute_mask( im, mask, varargin )
%ABSOLUTE_MASK Summary of this function goes here
%   Detailed explanation goes here

desc = max(imresize(double(mask),[8 8]),0);