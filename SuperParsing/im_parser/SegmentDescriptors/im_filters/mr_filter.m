function [ textons ] = mr_filter( im, centers )
%MR_FILTER Summary of this function goes here
%   Detailed explanation goes here
[foo, filt_resp] = FullMR8(im);
dist_mat = dist2(filt_resp', centers.mr_resp_centers);
[min_dist, min_ind] = min(dist_mat, [], 2);
textons = reshape(min_ind,[size(im,1) size(im,2)]);