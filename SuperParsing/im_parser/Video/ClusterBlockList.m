function [d] = ClusterBlockList(blockList,integralIm, integralImSq)
    numBlocks = size(blockList,1);
    d = sparse(numBlocks,numBlocks);
    eb = zeros(numBlocks,1);
    for i = 1:numBlocks; 
        eb(i) = CalcError(GetIVal(integralIm,blockList(i,:)),GetIVal(integralImSq,blockList(i,:)),NumInB(blockList(i,:))); 
    end
    for i = 1:numBlocks
        bs = blockList(i+1:end,:);
        b = blockList(i,:);
        a1 = inb(b(1),b(4),bs(:,1))|inb(b(1),b(4),bs(:,4));
        a2 = inb(b(2),b(5),bs(:,2))|inb(b(2),b(5),bs(:,5));
        a3 = inb(b(3),b(6),bs(:,3))|inb(b(3),b(6),bs(:,6));
        ind = find(a1&a2&a3);
        for j = ind(:)'
            bu = CalcError(GetIVal(integralIm,blockList(j+i,:))+GetIVal(integralIm,b),GetIVal(integralImSq,blockList(j+i,:))+GetIVal(integralImSq,b),NumInB(b)+NumInB(blockList(j+i,:)));
            d(i+j,i) = abs(bu-eb(i)-eb(i+j));
            d(i,i+j) = abs(bu-eb(i)-eb(i+j));
        end
    end
    %{
    d = squareform(d);
    Z = linkage(d);
    c = cluster(Z,'cutoff',1000,'criterion','distance');
    L = zeros(size(integralIm(:,:,:,1))-1);
    for i = 1:length(c)
        b = blockList(i,:);
        L(b(1):b(4)-1,b(2):b(5)-1,b(3):b(6)-1) = c(i);
    end
    %}
end
function r = inb(a,b,c)
    r = a<=c & b>=c;
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