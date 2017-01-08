function [mcc] = TrainMultiClassClassifier(HOMEDATA, HOMELABELSETS, Labels, trainIndexs, trainGlobalDesc, testIndexs, testGlobalDesc, descList)

for ls = 1:length(trainIndexs)
    numIm = length(unique(trainIndexs{ls}.image));
    Y = zeros(numIm,length(Labels{ls}));
    for l = 1:size(Y,2)
        Y(unique(trainIndexs{ls}.image(trainIndexs{ls}.label==l)),l) = 1;
    end
    numDim = 0;
    for d = 1:length(descList)
        numDim = numDim + size(trainGlobalDesc.(descList{d}),2);
    end
    X = zeros(numIm,numDim);
    curDim = 1;
    for d = 1:length(descList)
        descSize = size(trainGlobalDesc.(descList{d}),2);
        X(:,curDim:curDim+descSize-1) = trainGlobalDesc.(descList{d});
        curDim = curDim + descSize;
    end
    B = mnrfit(X,Y);
end