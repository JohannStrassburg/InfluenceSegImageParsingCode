function [ probPerLabel, dataCost, tubeSize ] = CombMaxDataSPSize( spProbs, spData, spSizes, ~ )
%COMBMAX Summary of this function goes here
%   Detailed explanation goes here

for i = 1:length(spProbs)
    if(~isempty(spProbs{i}))
        numLabels = size(spProbs{i},2);
        break;
    end
end
probPerLabel = zeros(size(spProbs,2),numLabels);
dataCost = zeros(size(spProbs,2),numLabels);
tubeSize = zeros(length(spSizes),1);
for i = 1:length(spProbs)
    if(isempty(spProbs{i}))
        continue;
    end
    [a, ind] = min(spData{i}./repmat(spSizes{i},[1 numLabels]),[],1);
    cind = sub2ind(size(spProbs{i}),ind,1:length(ind));
    probPerLabel(i,:) = spProbs{i}(cind);
    dataCost(i,:) = a.*mean(spSizes{i}(ind));
    tubeSize(i) = sum(spSizes{i});
end

end

