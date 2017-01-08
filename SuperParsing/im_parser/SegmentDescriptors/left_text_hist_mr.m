function [ desc ] = left_text_hist_mr( im, mask, maskCrop, bb, centers, textons, borders, varargin )
%LEFT_TEXT_HIST_M8 Summary of this function goes here
%   Detailed explanation goes here

%[foo, borders] = get_int_and_borders(mask);
mask= borders(:,:,1);
mask = mask(:);
textonIm = textons.mr_filter(:);
textonIm(~mask) = [];

dictionarySize = size(centers.mr_resp_centers,1);
desc = calculate_texton_hist( textonIm, dictionarySize );