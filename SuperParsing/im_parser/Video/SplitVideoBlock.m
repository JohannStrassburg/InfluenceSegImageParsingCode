function [ blockList ] = SplitVideoBlock( integralIm, integralImSq, integralImAll )
%SPLITVIDEOBLOCK Summary of this function goes here
%   Detailed explanation goes here
    [ro co t d] = size(integralIm);
    blockList = SplitVideoBlockRecurse(integralIm, integralImSq, [1 1 1 ro co t],integralImAll);%[1 13 2 105 65 12][95 1 1 115 56 12]
    
    %L = ClusterBlockList(blockList,integralIm, integralImSq);
    
    %{
    cform = makecform('lab2srgb');
    for i = 1:size(integralImAll,3)
        blockndx = find((blockList(:,3)<=i)&(blockList(:,4)>=i));
        show(applycform(reshape(integralImAll(:,:,i,1:3),[size(integralImAll,1) size(integralImAll,2) 3]),cform),1); 
        for bl = blockndx(:)'
            b = blockList(bl,:);
            rectangle('Position',[b(2) b(1) b(5)-b(2) b(4)-b(1)],'EdgeColor','r');
        end
        drawnow;
    end
    %}
end

function [ blockList ] = SplitVideoBlockRecurse( integralIm, integralImSq, b, integralImAll )
    [ro co t] = SizeOfB(b);
    if((t<10)&&(ro<100)&&(co<100))
        if(ro>100||co>100)
            fprintf('%d %d %d\n',ro,co,t);
        end
        blockList = b;
        return;
    end
    n = (ro*co*t);
    eb = CalcError(GetIVal(integralIm,b),GetIVal(integralImSq,b),NumInB(b));
    %ebt = FullCalcError(integralImAll,b);

    gain = zeros(max([ro co t])-1,3);
    %gaint = zeros(max([ro co t])-1,3);
    for i = b(1)+1:b(4)-1
        bt = b; bt(4) = i;
        eb1 = CalcError(GetIVal(integralIm,bt),GetIVal(integralImSq,bt),NumInB(bt));
        %ebt1 = FullCalcError(integralImAll,bt);
        bt = b; bt(1) = i;
        eb2 = CalcError(GetIVal(integralIm,bt),GetIVal(integralImSq,bt),NumInB(bt));
        %ebt2 = FullCalcError(integralImAll,bt);
        %gaint(i,1) = (ebt1+ebt2-ebt);
        gain(i,1) = (eb1+eb2-eb);
    end
    for i = b(2)+1:b(5)-1
        bt = b; bt(5) = i;
        eb1 = CalcError(GetIVal(integralIm,bt),GetIVal(integralImSq,bt),NumInB(bt));
        %ebt1 = FullCalcError(integralImAll,bt);
        bt = b; bt(2) = i;
        eb2 = CalcError(GetIVal(integralIm,bt),GetIVal(integralImSq,bt),NumInB(bt));
        %ebt2 = FullCalcError(integralImAll,bt);
        %gaint(i,2) = (ebt1+ebt2-ebt);
        gain(i,2) = (eb1+eb2-eb);
    end
    for i = b(3)+1:b(6)-1
        bt = b; bt(6) = i;
        eb1 = CalcError(GetIVal(integralIm,bt),GetIVal(integralImSq,bt),NumInB(bt));
        %ebt1 = FullCalcError(integralImAll,bt);
        bt = b; bt(3) = i;
        eb2 = CalcError(GetIVal(integralIm,bt),GetIVal(integralImSq,bt),NumInB(bt));
        %ebt2 = FullCalcError(integralImAll,bt);
        %gaint(i,3) = (ebt1+ebt2-ebt);
        gain(i,3) = (eb1+eb2-eb);
    end

    %figure(1);plot(gain);%figure(2);plot(gaint);%set(gca,'yscale','log');drawnow;
    %show((reshape(integralImAll(:,:,b(3),1:3),[size(integralImAll,1) size(integralImAll,2) 3])),3); rectangle('Position',[b(2) b(1) b(5)-b(2) b(4)-b(1)],'EdgeColor','r');drawnow;
    %show((reshape(integralImAll(:,:,b(6)-1,1:3),[size(integralImAll,1) size(integralImAll,2) 3])),4); rectangle('Position',[b(2) b(1) b(5)-b(2) b(4)-b(1)],'EdgeColor','r');drawnow;
    [v ndx] = min(-abs(gain));
    [v dim] = min(v);
    ndx = ndx(dim);
    
    bt = b; bt(dim+3)= ndx;
    BL1 = SplitVideoBlockRecurse(integralIm, integralImSq,bt,integralImAll);
    bt = b; bt(dim)= ndx;
    BL2 = SplitVideoBlockRecurse(integralIm, integralImSq,bt,integralImAll);
    blockList = [BL1;BL2];
end

function val = GetIVal(integralIm,b)
    val(1,:) = integralIm(b(4),b(5),b(6),:);
    val(2,:) =    -integralIm(b(1),b(5),b(6),:);
    val(3,:) =    -integralIm(b(4),b(2),b(6),:);
    val(4,:) =    -integralIm(b(4),b(5),b(3),:);
    val(5,:) =    +integralIm(b(1),b(2),b(6),:);
    val(6,:) =    +integralIm(b(4),b(2),b(3),:);
    val(7,:) =    +integralIm(b(1),b(5),b(3),:);
    val(8,:) =    -integralIm(b(1),b(2),b(3),:);
    val = sum(val,1);
end
function n = NumInB(b)
    n = prod(SizeOfB(b));
end
function [ro co t] = SizeOfB(b)
    ro = [b(4)-b(1) b(5)-b(2) b(6)-b(3)]; 
    if(nargout>1)
        t = ro(3);co = ro(2);ro=ro(1);
    end
end

function e = CalcError(sfp,sfp2,n)
e = ((sfp.^2)./n)+sfp2-(2.*(sfp.^2)./n);
e= sum(e);
end

function e = FullCalcError(integralImAll,b)
    temp = integralImAll(b(1):b(4)-1,b(2):b(5)-1,b(3):b(6)-1,:);
    st = size(temp);
    meanf = sum(sum(sum(temp,1),2),3);
    meanf = meanf./NumInB(b);
    temp = (temp - repmat(meanf,[st(1:3) 1])).^2;
    e = sum(temp(:));
end