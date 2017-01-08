function [ textons ] = sift_textons( im, centers )
%MR_FILTER Summary of this function goes here
%   Detailed explanation goes here
im2 = padarray(im,[7 7],'symmetric','both');
sift = sp_dense_sift(im2);%LMDenseSift(im2);%
flatsift = reshape(sift,[size(sift,1)*size(sift,2) size(sift,3)]);
dist_mat = dist2(flatsift, centers.sift_centers);
[min_dist, min_ind] = min(dist_mat, [], 2);
textons = reshape(min_ind,[size(sift,1) size(sift,2)]);