function testMetadataSVM = TrainLinearSVMForMetaData(trainMetadata,testMetadata,trainGlobalDesc, testGlobalDesc,descList)


descStr = [];
for i = 1:length(descList)
    scale = std(trainGlobalDesc.(descList{i}));%ones(size());
    trainData{i,1} = trainGlobalDesc.(descList{i})./repmat(scale,[size(trainGlobalDesc.(descList{i}),1) 1]);
    testData{i,1} = testGlobalDesc.(descList{i})./repmat(scale,[size(testGlobalDesc.(descList{i}),1) 1]);
    str = descList{i}(2:end); str(str>='a'&str<='z') = [];str = [descList{i}(1) str];
    descStr = [descStr str];
end
trainData = cell2mat(trainData');
testData = cell2mat(testData');

options = '';
useLinear = true;
testMetadataSVM = [];

fields = fieldnames(trainMetadata);
for f = 1:length(fields)
    [labels foo trainLabelNums] = unique(trainMetadata.(fields{f}));
    [foo    bar testLabelNums] = unique(testMetadata.(fields{f}));
    
    
    numLabels = length(labels);
    if(numLabels == 2) numLabels = 1; end
    numTrainIms = size(trainData,1);
    numTestIms = size(testData,1);
    
    trainLabels = -1*ones(numTrainIms,numLabels);
    for l = 1:numLabels; trainLabels(trainLabelNums==l,l) = 1; end
    testLabels = -1*ones(numTestIms,numLabels);
    for l = 1:numLabels; testLabels(testLabelNums==l,l) = 1; end
    testResults = -1*ones(numTestIms,numLabels);

    wts = 10000;%[.2 .1 .05 .025];%
    rates = zeros(numLabels,2*length(wts));
    tp = zeros(numLabels,2*length(wts));
    tn = zeros(numLabels,2*length(wts));
    for l = 1:numLabels
        for w = 1:length(wts)
            tl = trainLabels(:,l); tl(tl==-1) = 2;
            [a b] = UniqueAndCounts(tl);
            b = (1./b);%[1; 1];% 
            weights = b(tl);
            %weights = weights./sum(weights);
            options = ['-c ' num2str(wts(w))];
            if(useLinear); weights = weights.^2; model=liblineartrain(weights,trainLabels(:,l),sparse(trainData),options); else model=libsvmtrain(weights,trainLabels(:,l),trainData,options);end
            tel = testLabels(:,l);
            if(useLinear); [p tmp testResults(:,l)] = liblinearpredict(tel,sparse(testData),model);  else [p tmp d] = libsvmpredict(tel,testData,model); end
            rates(l,w) = tmp(1);
            tp(l,w) = sum(p(tel>0)==tel(tel>0))/sum(tel>0);
            tn(l,w) = sum(p(tel<0)==tel(tel<0))/sum(tel<0);
        end
    end
    if(numLabels==1)
        testResults(:,2) = -testResults(:,1);
    end
    [foo testLabelNumsSVM] = max(testResults,[],2);
    testMetadataSVM.(fields{f}) = labels(testLabelNumsSVM);
end