function [ probPerLabel, dataCost, tubeSize ] = CombMean( spProbs, spData, spSizes, ~ )
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
    tubeSize(i) = sum(spSizes{i});
    probPerLabel(i,:) = mean(spProbs{i},1);
    dataCost(i,:) = mean(spData{i},1);
end

end

