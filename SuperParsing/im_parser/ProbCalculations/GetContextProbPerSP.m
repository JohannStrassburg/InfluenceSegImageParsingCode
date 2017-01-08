function [probPerLabel] = GetContextProbPerSP(HOMEDATATEST,probPerSP,imSP,contextDesc,spIndexTrain,knntarget,suffix,baseFName,shortLabelNum,labCount,useRatio,descFun,l2l,canSkip,rnn)

curIndexLength = length(spIndexTrain.DBImageIndex);
outfilename = fullfile(HOMEDATATEST,['probPerLabel' suffix],sprintf('%s_count.mat',baseFName));
if(exist(outfilename,'file')&&canSkip)
    load(outfilename);
    if(exist('indexLength','var') && indexLength == curIndexLength)
        if(exist('probPerDescPerLabel','var'))
            if(size(probPerLabel,2)==length(shortLabelNum))
                return;
            end
        end
    end
end

[foo maxL] = max(probPerSP,[],2);
spndx = unique(imSP);
names = fieldnames(contextDesc);
contextForIm = ComputeContexForIm(spndx,maxL,imSP,size(contextDesc.(names{1}),1));

dists = dist2(contextForIm.(descFun)',contextDesc.(descFun)');%hist_isect
rawNNs = [];
for j = 1:size(probPerSP,1)
    dist = dists(j,:);%distind(nns);%
    [rawNNs(j).context.dists ind] = sort(dist);
    if(exist('rnn','var'))
        fstDif = find(rawNNs(j).context.dists>rnn,1);
    else
        fstDif = find(rawNNs(j).context.dists>rawNNs(j).context.dists(knntarget-1),1);
    end
    rawNNs(j).context.dists = rawNNs(j).context.dists(1:(fstDif-1));
    rawNNs(j).context.nns = ind(1:(fstDif-1));
end
[probPerLabel] = GetAllProbPerLabel(HOMEDATATEST,baseFName,spIndexTrain,rawNNs,shortLabelNum,labCount,suffix,useRatio,1,l2l); %#ok<AGROW>

