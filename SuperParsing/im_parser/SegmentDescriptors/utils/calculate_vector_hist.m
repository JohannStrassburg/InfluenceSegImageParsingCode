function [ descHist ] = calculate_vector_hist( descVec, centers )
%CALCULATE_VECTOR_HIST Summary of this function goes here
%   Detailed explanation goes here

dist_mat = dist2(descVec, centers);
[min_dist, min_ind] = min(dist_mat, [], 2);

dictionarySize = size(centers,1);
descHist = hist(min_ind,1:dictionarySize)./size(descVec,1);
