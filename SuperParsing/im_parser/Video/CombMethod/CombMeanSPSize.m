function [ probPerLabel, dataCost, tubeSize ] = CombMeanSPSize( spProbs, spData, spSizes, ~ )
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
    probPerLabel(i,:) = sum(spProbs{i}.*repmat(spSizes{i},[1 numLabels]),1)./sum(spSizes{i});
    dataCost(i,:) = sum(spData{i}.*repmat(spSizes{i},[1 numLabels]),1)./sum(spSizes{i});
end

end