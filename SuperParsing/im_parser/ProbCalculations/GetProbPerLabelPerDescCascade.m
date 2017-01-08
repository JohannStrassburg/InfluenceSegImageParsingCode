function [descProb, descVotes] = GetProbPerLabelPerDescCascade(D,spIndex,nns,filteredLabels,filtLabCounts,listFun,combineLabels)

if(~exist('listFun','var'))
    listFun = @GetLabelListNum;
end

if(~exist('combineLabels','var'))
    combineLabels = [];
end

[a b c] = intersect(filteredLabels,combineLabels);
[newFiltL ind] = setdiff(filteredLabels,a);
newCount = filtLabCounts(ind);
newCount(end+1) = sum(filtLabCounts(b));

descProb = ones(length(newCount),1)./sum(newCount);
descVotes = zeros(length(newCount),1);
[labelst votes] = GetLabelVotes(D,spIndex,nns,filteredLabels,listFun);

%fix up labels
[x y z] = intersect(newFiltL,filteredLabels(labelst));
labels(z) = y;
[x y z] = intersect(combineLabels,filteredLabels(labelst));
labels(z) = length(newCount);

%for ratio
%{
tv = sum(votes);
tc = sum(newCount);
descProb = ones(length(newCount),1)./(tv+1);
%}
for i = 1:length(labels)
    %descProb(labels(i)) = ((1+votes(i))/(1+tv-votes(i)))*((tc/newCount(labels(i))));
    descProb(labels(i)) = (votes(i))/(newCount(labels(i)));
    descVotes(labels(i)) = votes(i);
end
end

