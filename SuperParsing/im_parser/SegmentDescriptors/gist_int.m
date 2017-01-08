function [ desc ] = gist_int( im, mask, maskCrop, bb, centers, textons, borders, imAll, varargin  )
%GIST_INT Summary of this function goes here
%   Detailed explanation goes here

numberBlocks = 4;
if ndims(im) == 3
    I = rgb2gray(imAll);
end

[y x] = find(mask);
y1 = min(y); y2 = max(y); x1 = min(x);x2 = max(x);
h = y2-y1+1;w=x2-x1+1;
padAmount = round((h-w)/2);
I = padarray(I,[max(-padAmount,0) max(padAmount,0)],'symmetric','both');
mh = (y1+y2)/2+max(-padAmount,0); mw = (x1+x2)/2+max(padAmount,0);
s = floor(max(h,w)/2);
I = I(fix(mh-s+.5):fix(mh+s),fix(mw-s+.5):fix(mw+s));

G = centers.gist_centers;
[ro co ch] = size(G);
I = imresizecrop(I,[ro co], 'bicubic');
output = prefilt(im2double(I),4);
desc = gistGabor(output,numberBlocks,G);
end
