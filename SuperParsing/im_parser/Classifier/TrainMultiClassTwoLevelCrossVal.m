descList = testParams.SVMDescs;
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
        
    Level2Training = zeros(size(trainLabels));
    numCrossValidation = 10;
    for cv = 1:numCrossValidation
        testRange = SetupRange(cv,numCrossValidation,numTrainIms);
        testMask = zeros(numTrainIms,1)==1;testMask(testRange) = true;
        trainRange = find(~testMask);
        svm = TrainGlobalSVMBasic(SelectDesc(trainGlobalDesc,trainRange,1), trainLabels(trainRange,:), valGlobalDesc, valLabels, testGlobalDesc, testLabels,  descList);
        Level2Training(testRange,:) = svmRun(svm,SelectDesc(trainGlobalDesc,testRange,1),descList);
    end
    
    svmL1{ls} = TrainGlobalSVMBasic(trainGlobalDesc, trainLabels, valGlobalDesc, valLabels, testGlobalDesc, testLabels,  descList);
    trainSVMDesc.svm = Level2Training;
    valSVMDesc.svm = svmRun(svmL1{ls},valGlobalDesc,descList);
    testSVMDesc.svm = svmRun(svmL1{ls},testGlobalDesc,descList);
    level2GlobalSVM{ls} = TrainGlobalSVMBasic(trainSVMDesc, trainLabels, valSVMDesc, valLabels, testSVMDesc, testLabels,  {'svm'});
    
    sigmaSVM = .5;
    trainSVMDesc.svmsig = 1./(1+exp(-trainSVMDesc.svm./sigmaSVM));
    valSVMDesc.svmsig = 1./(1+exp(-valSVMDesc.svm./sigmaSVM));
    testSVMDesc.svmsig = 1./(1+exp(-testSVMDesc.svm./sigmaSVM));
    %{
    for l = 1:length(Labels{ls})
        fprintf('AUC: %.3f tpr: %.3f fpr: %.3f cr: %.3f - %s\n',auc(l),tpr(l,3),fpr(l,3),cr(l),Labels{ls}{l});
    end
    fprintf('correct rate: %.3f\n\n',mean(cr));
    %}
    
    globalSVM{ls} = {svmL1{ls},level2GlobalSVM{ls}};
end
evalSVM;
