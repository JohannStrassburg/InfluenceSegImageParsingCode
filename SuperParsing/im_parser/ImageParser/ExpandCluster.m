function [extendedClust extendedClustWithInitial] = ExpandCluster(labelHist,inds,expandedSize)%DStrain,imageNNs,clusterSize,shortLabNum,linkageType,type,fullDesc)

cDesc = labelHist(inds,:);
meanDesc = mean(cDesc,1);
clustersDists = dist2(meanDesc,labelHist);
[clustersDists sortedInds] = sort(clustersDists,'ascend');
extendedClust = sortedInds(1:min(end,expandedSize));

[a b] = intersect(sortedInds,inds);
sortedInds(b) = [];
extendedClustWithInitial = [inds sortedInds(1:min(end,expandedSize-length(inds)))];
end
