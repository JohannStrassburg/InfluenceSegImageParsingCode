function [segLabelProb segDescLabelProb] = GetProbPerLabel(spIndex,rawNNs,filteredLabels,filtLabCounts,type,smoothingConst)

descFuns = fieldnames(rawNNs);
segLabelProb = zeros(length(filtLabCounts),1); %ML
segDescLabelProb = zeros(length(filtLabCounts),length(descFuns));

for j = 1:length(descFuns)
    nns = rawNNs.(descFuns{j}).nns;
    if(isfield(spIndex,'label'))
        if(strfind(type,'extreme')==1)
            prob = GetProbPerLabelPerDescSimp(spIndex,nns,filteredLabels,filtLabCounts,type,smoothingConst);
            ps = sort(prob,'descend');
            wbparams = wblfit(ps(2:min(end,6)));
            segDescLabelProb(:,j) = log(wblcdf(prob,wbparams(1),wbparams(2))+.001);%./length(descFuns);
        else
            segDescLabelProb(:,j) = log(GetProbPerLabelPerDescSimp(spIndex,nns,filteredLabels,filtLabCounts,type,smoothingConst))';
        end
    end
    segLabelProb = segLabelProb + segDescLabelProb(:,j);
end
end
