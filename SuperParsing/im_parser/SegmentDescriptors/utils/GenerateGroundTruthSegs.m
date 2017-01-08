function [imSP, ll, names] = GenerateGroundTruthSegs(groundTruthFile)

load(groundTruthFile);

labels = unique(S);
imSP = zeros(size(S));
curL = 1;
ll = [];
for l = labels(:)'
    cc = bwconncomp(S==l,8);
    for r = 1:cc.NumObjects
        if(numel(cc.PixelIdxList{r})<=20)
            imSP(cc.PixelIdxList{r}) = 0;
        else
            imSP(cc.PixelIdxList{r}) = curL;
            ll(curL) = l;
            curL = curL+1;
        end
    end
end
if(sum(imSP(:)==0)<=20)
    while(sum(imSP(:)==0)>0)
        pixNdx = find(imSP==0);
        for i = pixNdx(:)'
            imSP(i) = -1;
            dmask = imdilate(imSP==-1,[1 1 1;1 0 1;1 1 1]);
            ls = unique(imSP(dmask));
            ls(ls==0) = [];
            if(~isempty(ls))
                imSP(i) = ls(1);
            end
        end
    end
end
imSP(imSP==0) = curL;
ll(curL) = 0;