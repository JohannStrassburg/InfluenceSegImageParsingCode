function [descProb, descVotes] = GetProbPerLabelPerDesc(D,spIndex,nns,filteredLabels,filtLabCounts,listFun,hierarchy)

if(~exist('listFun','var'))
    listFun = @GetLabelListNum;
end

if(~exist('hierarchy','var'))
    hierarchy = [];
end
newCount = filtLabCounts;
descProb = ones(length(newCount),1)./sum(newCount);
descVotes = zeros(length(newCount),1);
[labels votes] = GetLabelVotes(D,spIndex,nns,filteredLabels,listFun);

if(~isempty(hierarchy))
    labNumList = filteredLabels(labels);
    translator = [];
    newCount = zeros(length(hierarchy),1);
    for i = 1:length(hierarchy)
        [labNums inds] = intersect(labNumList,hierarchy{i});
        translator(labels(inds)) = i;
        [labNums inds] = intersect(filteredLabels,hierarchy{i});
        newCount(i) = sum(filtLabCounts(inds));
    end
    descProb = ones(length(newCount),1)./sum(newCount);
    descVotes = zeros(length(newCount),1);
    ltemp = translator(labels);
    lfinal = [];
    vfinal = [];
    for i = 1:length(hierarchy)
        ind = find(ltemp==i);
        if(~isempty(ind))
            lfinal = [lfinal i];
            vfinal = [vfinal sum(votes(ind))];
        end
    end
    labels = lfinal;
    votes = vfinal;
end

%for ratio
%{-
tv = sum(votes);
tc = sum(newCount);
descProb = ones(length(newCount),1)./(tv+1);
%}
for i = 1:length(labels)
    descProb(labels(i)) = ((1+votes(i))/(1+tv-votes(i)))*((tc/newCount(labels(i))));
    %descProb(labels(i)) = (votes(i))/(newCount(labels(i)));
    descVotes(labels(i)) = votes(i);
end
end

