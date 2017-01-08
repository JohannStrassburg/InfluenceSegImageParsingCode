function [ desc ] = dial_text_hist_mr( im, mask, maskCrop, bb, centers, textons, borders, varargin  )
%INT_TEXT_HIST_MR Summary of this function goes here
%   Detailed explanation goes here

%strEl = strel('square',20);
%mask = imdilate(mask,strEl,'same');
mask=borders(:,:,5);
mask = mask(:);
textonIm = textons.mr_filter(:);
textonIm(~mask) = [];

dictionarySize = size(centers.mr_resp_centers,1);
desc = calculate_texton_hist( textonIm, dictionarySize );