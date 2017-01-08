function [imgCrop, scaling] = LMobjectnormalizedcrop(img, annotation, j, b, height, width)
%
% Crop object from image using a normalized frame
% [imgCrop, scaling] = LMobjectnormalizedcrop(img, annotation, j, b, height, width)
%  extract object index j from the annotation.
%
%   ------------
%   |     b    |   b = number of boundary pixels
%   |   ----   |   h = height inner bounding box
%   | b |  |   |   w = width inner bounding box
%   |   |h |   |
%   |   ----   |
%   |    w     |
%   ------------
%

[X,Y,t] = getLMpolygon(annotation.object(j).polygon);
numFrames = length(t);

imgCrop = zeros([height+2*b width+2*b size(img,3) numFrames], 'uint8');
for f = t
    f
    if numFrames>1
        i = find(t==f); i = i(1);
        x = X(:,i);
        y = Y(:,i);
    else
        i = 1;
        x = X;
        y = Y;
    end

    %  bb = boundingbox = [xmin ymin xmax ymax]
    bb = [min(x) min(y) max(x) max(y)];

    % 1) Resize image so that object is normalized in size
    scaling = min(height/(bb(4)-bb(2)), width/(bb(3)-bb(1)));
    if numFrames>1
        [foo, I] = LMimscale([], img(:,:,:,f+1), scaling, 'bilinear');
    else
        [foo, I] = LMimscale([], img, scaling, 'bilinear');
    end
    bb = bb*scaling;

    % 2) pad image (just to make sure)
    margin = 10+ceil(b+1 + max([0 -bb(1) -bb(2) bb(4)-size(I,1) bb(3)-size(I,2)]));
    [foo, I] = LMimpad(foo, I, [size(I,1)+2*margin size(I,2)+2*margin], 128);
    bb = bb+margin;
    
     % 2) Crop result
%     [x,y] = getLMpolygon(annotation.object(j).polygon);
     cx = fix((bb(3)+bb(1))/2); cy = fix((bb(4)+bb(2))/2);
     bb = [cx-width/2-b+1 cy-height/2-b+1 cx+width/2+b cy+height/2+b];
     
bb
size(I)
    % Image crop:
    imgCrop(:,:,:,i) = I(bb(2):bb(4), bb(1):bb(3), :);
end

