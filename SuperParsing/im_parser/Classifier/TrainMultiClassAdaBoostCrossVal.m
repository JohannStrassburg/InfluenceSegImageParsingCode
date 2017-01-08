descList = testParams.SVMDescs;
globalBoost = cell(size(HOMELABELSETS));
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
        
    %{
    defaultStream = RandStream.getDefaultStream;
    savedState = defaultStream.State;
    Level2Training = zeros(size(trainLabels));
    numCrossValidation = 10;
    for cv = 1:numCrossValidation
        testRange = SetupRange(cv,numCrossValidation,numTrainIms);
        testMask = zeros(numTrainIms,1)==1;testMask(testRange) = true;
        trainRange = find(~testMask);
        defaultStream.State = savedState;
        svm = TrainGlobalSVMBasic(SelectDesc(trainGlobalDesc,trainRange,1), trainLabels(trainRange,:), valGlobalDesc, valLabels, testGlobalDesc, testLabels,  descList);
        Level2Training(testRange,:) = svmRun(svm,SelectDesc(trainGlobalDesc,testRange,1),descList);
    end
    defaultStream.State = savedState;
    svmL1{ls} = TrainGlobalSVMBasic(trainGlobalDesc, trainLabels, valGlobalDesc, valLabels, testGlobalDesc, testLabels,  descList);
    trainSVMDesc.svm = Level2Training;
    valSVMDesc.svm = svmRun(svmL1{ls},valGlobalDesc,descList);
    testSVMDesc.svm = svmRun(svmL1{ls},testGlobalDesc,descList);
    %}
    
    
    tr_error = zeros(size(trainLabels,2),weak_learner_n);
    te_error = zeros(size(trainLabels,2),weak_learner_n);
    weak_learner_n = 21;
    adaboost_model = cell(weak_learner_n,size(trainLabels,2));
    for l = 1:size(trainLabels,2)
        for i=1:weak_learner_n
            adaboost_model{i,l} = ADABOOST_tr(@threshold_tr,@threshold_te,trainSVMDesc.svm,(trainLabels(:,l)>0)+1,i);
            [L_tr,hits_tr] = ADABOOST_te(adaboost_model{i,l},@threshold_te,trainSVMDesc.svm,(trainLabels(:,l)>0)+1);
            tr_error(l,i) = (numTrainIms-hits_tr)/numTrainIms;
            [L_te,hits_te] = ADABOOST_te(adaboost_model{i,l},@threshold_te,testSVMDesc.svm,(testLabels(:,l)>0)+1);
            te_error(l,i) = (numTestIms-hits_te)/numTestIms;
            [L] = ADABOOST_te(adaboost_model{i,l},@threshold_te,valSVMDesc.svm,(valLabels(:,l)>0)+1);
            adaboost_model{i,l}.b = ROCBalance(L(:,2),valLabels(:,l)>0);
        end
    end
    globalBoost{ls} = adaboost_model;
end
