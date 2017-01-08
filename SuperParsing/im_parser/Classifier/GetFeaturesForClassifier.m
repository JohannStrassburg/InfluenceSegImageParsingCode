function [features featureGroup] = GetFeaturesForClassifier(spDescTrain,dimNum,sortNames)
if(~exist('dimNum','var'))
    dimNum = [];
end
if(~exist('sortNames','var'))
    sortNames = 0;
end
%fix for multi level
descNames = fieldnames(spDescTrain);
if(sortNames)
    descNames = sort(descNames);
end
featCount = 0;
for i = 1:length(descNames)
    featCount = featCount + size(spDescTrain.(descNames{i}),2);
end

numSP = size(spDescTrain.(descNames{1}),1);
if(~isempty(dimNum))
    features = zeros(numSP,length(dimNum));
    currentFeatPos = 1;
    for i = 1:length(descNames)
        featSize = size(spDescTrain.(descNames{i}),2);
        nextFeatPos = currentFeatPos+featSize-1;
        ft = spDescTrain.(descNames{i});
        [foo fetInd descInd] = intersect(dimNum,currentFeatPos:nextFeatPos);
        features(:,fetInd) = ft(:,descInd);
        currentFeatPos = nextFeatPos+1;
    end        
else
    features = zeros(numSP,featCount);
    featureGroup = zeros(1,featCount);
    currentFeatPos = 1;
    for i = 1:length(descNames)
        featSize = size(spDescTrain.(descNames{i}),2);
        nextFeatPos = currentFeatPos+featSize-1;
        ft = spDescTrain.(descNames{i});
        features(:,currentFeatPos:nextFeatPos) = ft;
        featureGroup(currentFeatPos:nextFeatPos)=i;
        currentFeatPos = nextFeatPos+1;
    end
end

end