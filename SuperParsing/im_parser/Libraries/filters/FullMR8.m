function [ featvec, filtvec ] = FullMR8( I )
%FULLMR8 Summary of this function goes here
%   Detailed explanation goes here

if(size(I,3)==3)
    I = rgb2gray(I);
end

I = padarray(I,[25 25],'symmetric','both');

[ featvec, filtvec ] = MR8(I);