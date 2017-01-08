function [descProb, descVotes] = GetWeightedProbPerLabelPerDesc(D,spIndex,nns,weights,filteredLabels,filtLabCounts)
descProb = ones(length(filteredLabels),1)./sum(filtLabCounts);
descVotes = zeros(length(filteredLabels),1);
[labels votes] = GetLabelWeightedVotes(D,spIndex,nns,weights,filteredLabels);
for i = 1:length(labels)
    descProb(labels(i)) = (votes(i))/(filtLabCounts(labels(i)));
    descVotes(labels(i)) = votes(i);
end
end

