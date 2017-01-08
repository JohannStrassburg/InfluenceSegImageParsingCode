function [ spIm, im ] = GenerateSuperPixels( im, k , vx, vy )
%GENERATESUPERPIXELS Summary of this function goes here
%   Detailed explanation goes here

if(~exist('k','var'))
    k = 100;
end
k = k*max(1,(length(im)/640)^2);%
sigma = .8;
if(~exist('vx','var'))
    spIm = SegmentToLabels(segmentmex(im,sigma,k,100));
else
    im = im2single(im);
    %vx = vx./50;vx=vx+.5;
    %vx(vx<0) = 0; vx(vx>1) = 1;
    %vy = vy./50;vy=vy+.5;
    %vy(vy<0) = 0; vy(vy>1) = 1;
    %{
    [a b] = SpatialCues(vx,5);
    vx = max(a,b);
    [a b] = SpatialCues(vy,5);
    vy = max(a,b);
    %}
    %vx = padarray(max(abs(filter2(fspecial('sobel'),vx,'valid')),abs(filter2(fspecial('sobel')',vx,'valid'))),[1 1]);
    %vy = padarray(max(abs(filter2(fspecial('sobel'),vy,'valid')),abs(filter2(fspecial('sobel')',vy,'valid'))),[1 1]);
    %vx(vx<.2) = 0;
    %vy(vy<.2) = 0;
    %maxvval = max(max(vx(:)),max(vy(:)));
    %maskX = (abs(vx([1 1:end-1],:)-vx(1:end,:))>.2)|(abs(vx(:,[1 1:end-1])-vx(:,1:end))>.2);
    %maskY = (abs(vy([1 1:end-1],:)-vy(1:end,:))>.2)|(abs(vy(:,[1 1:end-1])-vy(:,1:end))>.2);
    im(:,:,4) = vx;
    im(:,:,5) = vy;
    im = im*128;
    xdif = (im(1:end-1,:,:)-im(2:end,:,:)).^2;
    ydif = (im(1:end-1,:,:)-im(2:end,:,:)).^2;
    coMean = [sqrt(sum(xdif(:,:,1:3),3)) sqrt(sum(ydif(:,:,1:3),3))];coMean = mean(coMean(:));
    floMean = [sqrt(sum(xdif(:,:,4:5),3)) sqrt(sum(ydif(:,:,4:5),3))];floMean = mean(floMean(:));
    im(:,:,4:5) = im(:,:,4:5).*coMean/floMean;
    %im(:,:,4) = maskX*coMean;
    %im(:,:,5) = maskY*coMean;
    spIm = SegmentToLabels(segmentmex(im,sigma,k,100));
end