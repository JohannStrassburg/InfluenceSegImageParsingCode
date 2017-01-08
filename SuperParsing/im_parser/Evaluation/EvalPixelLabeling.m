function [perPixStats perLabelStats conMat] = EvalPixelLabeling(L,labelList,S,names)
perLabelStats = zeros(length(names),2);
lS = unique(S);
lS(lS<=0) = [];
for l = lS(:)'
    mask = S==l;
    perLabelStats(l,1) = sum(L(mask)==l);
    perLabelStats(l,2) = sum(mask(:));
end
perPixStats(1) = sum(perLabelStats(:,1));
perPixStats(2) = sum(perLabelStats(:,2));
if(nargout>2)
    %needs to be debugged
    conMat = zeros(length(labelList),length(labelList));
    for l1 = lS(:)'
        Lm = L(S==l1);
        lL = unique(Lm);
        for l2 = lL(:)'
            conMat(l1,l2) = sum(Lm==l2);
        end
    end
end


