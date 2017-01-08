descList = testParams.SVMDescs;%{'parserHist'};%
svmL1 = [];
level2GlobalSVM = [];
for ls = UseGlobalSVM
    [fo labelTypeName] = fileparts(HOMELABELSETS{ls});
    
    numLabels = length(Labels{ls});
    numTrainIms = length(trainFileList);
    numTestIms = length(testFileList);
    numValIms = length(valFileList);
    
    trainLabels = -1*ones(numTrainIms,numLabels);
    for l = 1:numLabels; trainLabels(trainIndex{ls}.image(trainIndex{ls}.label==l),l) = 1; end
    testLabels = -1*ones(numTestIms,numLabels);
    for l = 1:numLabels; testLabels(testIndex{ls}.image(testIndex{ls}.label==l),l) = 1; end
    valLabels = -1*ones(numValIms,numLabels);
    for l = 1:numLabels; valLabels(valIndex{ls}.image(valIndex{ls}.label==l),l) = 1; end
    
    [svm{ls} a b c] = TrainGlobalSVMBasic(trainGlobalDesc, trainLabels, valGlobalDesc, valLabels, testGlobalDesc, testLabels,  descList);    
    globalSVM{ls} = {svm{ls}};
end
evalSVM;
