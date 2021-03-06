function [ probPerLabel, dataCost, tubeSize ] = CombMax( spProbs, spData, spSizes, ~ )
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
    [a, ind] = max(spProbs{i},[],1);
    probPerLabel(i,:) = spProbs{i}(sub2ind(size(spProbs{i}),ind,1:length(ind)));
    dataCost(i,:) = spData{i}(sub2ind(size(spProbs{i}),ind,1:length(ind)));
    tubeSize(i) = sum(spSizes{i});
end

end

