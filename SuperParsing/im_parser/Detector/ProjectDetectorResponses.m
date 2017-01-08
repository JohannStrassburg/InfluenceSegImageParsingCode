function [ dataTerm dataTermMax] = ProjectDetectorResponses( im, bbs, models)
%PROJECTDETECTORRESPONSES Summary of this function goes here
%   Detailed explanation goes here

[ro co ch] = size(im);
dataTerm = zeros(ro,co);
dataTermMax = zeros(ro,co);
for i = 1:size(bbs,1)
    score = bbs(i,end)+1;
    model = bbs(i,6);
    gtbb = models{model}.gt_box;
    flipped = bbs(i,7);
    if(isfield(models{model}.polygon,'pt'))
        poly.x = str2double({models{model}.polygon.pt.x});
        poly.y = str2double({models{model}.polygon.pt.y});
    else
        poly = models{model}.polygon;
    end
    tbb = bbs(i,1:4);
    
    sx = (tbb(3)-tbb(1))/(gtbb(3)-gtbb(1));
    sy = (tbb(4)-tbb(2))/(gtbb(4)-gtbb(2));
    if(isnan(sx)||isnan(sy)||isinf(sx)||isinf(sy))
        continue;
    end
    tx = tbb(1) - gtbb(1);
    ty = tbb(2) - gtbb(2);
    poly2.x = double(poly.x-gtbb(1)).*sx+tbb(1);
    poly2.y = double(poly.y-gtbb(2)).*sy+tbb(2);
    if(~flipped)
        poly2.x = (co/2)-(poly2.x-(co/2));
    end
    mask = poly2mask(poly2.x,poly2.y,ro,co);
    dataTerm = dataTerm + mask.*score;
    dataTermMax = max(dataTermMax,mask.*score);
    %show(dataTerm,1);
    %{
    figure(1);
    hold on;
    axis([0 co 0 ro]);axis equal;
    rectangle('Position',[gtbb(1) gtbb(2) gtbb(3)-gtbb(1) gtbb(4)-gtbb(2)]);
    fill(poly.x,poly.y,'r');
    rectangle('Position',[tbb(1) tbb(2) tbb(3)-tbb(1) tbb(4)-tbb(2)]);
    fill(poly2.x,poly2.y,'r');
    %}
    
end


end

