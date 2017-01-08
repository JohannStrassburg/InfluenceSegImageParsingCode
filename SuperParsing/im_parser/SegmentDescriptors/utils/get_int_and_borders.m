function [ int_mask, border, bb] = get_int_and_borders( mask )
%GET_INT_AND_BORDERS Summary of this function goes here
%   Detailed explanation goes here
if(sum(mask(:)==1) == numel(mask))
    mask(1:end,[1 end]) = 0;
    mask([1 end], 1:end) = 0;
end
strEl = strel('square',5);
int_mask = imerode(mask,strEl,'same');
full_border = double(imdilate(mask,strEl,'same')-int_mask);

[y x] = find(full_border);
[r c] = size(full_border);
top = min(y);
bottom = max(y);
left = min(x);
right = max(x);
yVals = repmat((1:r)',1,c);
xVals = repmat(1:c,r,1);
border = zeros([r c 5]);
border(:,:,1) = abs(xVals-left);
border(:,:,2) = abs(xVals-right);
border(:,:,3) = abs(yVals-top);
border(:,:,4) = abs(yVals-bottom);
[foo, index] = min(border(:,:,1:4),[],3);
border(:,:,1) = (index==1).*full_border;
border(:,:,2) = (index==2).*full_border;
border(:,:,3) = (index==3).*full_border;
border(:,:,4) = (index==4).*full_border;

strEl = strel('square',20);
border(:,:,5) = imdilate(mask,strEl,'same');
[y x] = find(border(:,:,5));
border = border>0;
top = min(y);
bottom = max(y);
left = min(x);
right = max(x);
bb = [top bottom left right];