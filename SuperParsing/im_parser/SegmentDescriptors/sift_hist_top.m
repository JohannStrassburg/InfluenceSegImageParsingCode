function [ desc ] = sift_hist_top( im, mask, maskCrop, bb, centers, textons, borders, varargin  )
%TOP_TEXT_HIST_M8 Summary of this function goes here
%   Detailed explanation goes here

%[foo, borders] = get_int_and_borders(mask);
mask= borders(:,:,3);
mask = mask(:);
textonIm = textons.sift_textons(:);
textonIm(~mask) = [];

dictionarySize = size(centers.sift_centers,1);
desc = calculate_texton_hist( textonIm, dictionarySize );