function [SVMOutput] = TrainLinearSVM(HOMEDATA, HOMELABELSETS, Labels, trainIndex, trainGlobalDesc, testIndex, testGlobalDesc, descList)

try for i = 3:100;fclose(i);fprintf('closed %d\n',i);end;catch ERR;end

svm = cell(size(HOMELABELSETS));
descStr = [];
for i = 1:length(descList)
    scale = std(trainGlobalDesc.(descList{i}));%ones(size());
    trainData{i,1} = trainGlobalDesc.(descList{i})./repmat(scale,[size(trainGlobalDesc.(descList{i}),1) 1]);
    testData{i,1} = testGlobalDesc.(descList{i})./repmat(scale,[size(testGlobalDesc.(descList{i}),1) 1]);
    str = descList{i}(2:end); str(str>='a'&str<='z') = [];str = [descList{i}(1) str];
    descStr = [descStr str];
end

useLinear = true;

trainData = cell2mat(trainData');
testData = cell2mat(testData');
if(useLinear); svstr = 'Linear'; else svstr = 'LibSVM'; end
wts = 1;%10.^(-3:3);%[.2 .1 .05 .025];%

saveFile = fullfile(HOMEDATA,'SVM',['SVMTests' svstr '.txt']);
make_dir(saveFile);
fid = fopen(saveFile,'w');
SVMOutput = cell(size(HOMELABELSETS));
for ls = 1:length(HOMELABELSETS)
    HOMELABELSET = HOMELABELSETS{ls};
    [fo labelSet] = fileparts(HOMELABELSET);
    
    numLabels = length(Labels{ls});
    numTrainIms = size(trainData,1);
    numTestIms = size(testData,1);
    SVMOutput{ls} = zeros(numTestIms,numLabels);
    topPN = 0;topC = 0;
    %numValIms = length(valFileList);
    
    trainLabels = -1*ones(numTrainIms,numLabels);
    for l = 1:numLabels; trainLabels(trainIndex{ls}.image(trainIndex{ls}.label==l),l) = 1; end
    testLabels = -1*ones(numTestIms,numLabels);
    for l = 1:numLabels; testLabels(testIndex{ls}.image(testIndex{ls}.label==l),l) = 1; end
    %valLabels = -1*ones(numValIms,numLabels);
    %for l = 1:numLabels; valLabels(valIndex{ls}.image(valIndex{ls}.label==l),l) = 1; end
    
    rates = zeros(numLabels,2*length(wts));
    tp = zeros(numLabels,2*length(wts));
    tn = zeros(numLabels,2*length(wts));
    for l = 1:numLabels
        topPN = 0;
        for w = 1:length(wts)
            tl = trainLabels(:,l); tl(tl==-1) = 2;
            [a b] = UniqueAndCounts(tl);
            b = (1./b);%[1; 1];% 
            weights = b(tl);
            %weights = weights./sum(weights);
            options = ['-q -c ' num2str(wts(w))];
            if(useLinear); weights = weights.^2; model=liblineartrain(weights,trainLabels(:,l),sparse(trainData),options); else model=libsvmtrain(weights,trainLabels(:,l),trainData,options);end
            tel = testLabels(:,l);
            if(useLinear); [p tmp d] = liblinearpredict(tel,sparse(testData),model,'-b 1');  else [p tmp d] = libsvmpredict(tel,testData,model); end
            d = d*model.Label(1);
            rates(l,w) = tmp(1);
            tp(l,w) = sum(p(tel>0)==tel(tel>0))/sum(tel>0);
            tn(l,w) = sum(p(tel<0)==tel(tel<0))/sum(tel<0);
            if(topPN < tp(l,w))%+tn(l,w))
                topPN = tp(l,w);%+tn(l,w);
                topC = wts(w);
                SVMOutput{ls}(:,l) = d;
            end
        end
        
        for w = 1:length(wts)
            b = [1/numTrainIms; 1/numTrainIms];
            weights = b(tl);
            options = ['-q -c ' num2str(wts(w))];
            if(useLinear); model=liblineartrain(weights,trainLabels(:,l),sparse(trainData),options); else model=libsvmtrain(weights,trainLabels(:,l),trainData,options);end
            tel = testLabels(:,l);
            if(useLinear); [p tmp d] = liblinearpredict(tel,sparse(testData),model,'-b 1');  else [p tmp d] = libsvmpredict(tel,testData,model); end
            d = d*model.Label(1);
            rates(l,length(wts)+w) = tmp(1);
            tp(l,length(wts)+w) = sum(p(tel>0)==tel(tel>0))/sum(tel>0);
            tn(l,length(wts)+w) = sum(p(tel<0)==tel(tel<0))/sum(tel<0);
            if(topPN < tp(l,length(wts)+w))%+tn(l,length(wts)+w))
                topPN = tp(l,length(wts)+w);%+tn(l,length(wts)+w);
                topC = wts(w);
                SVMOutput{ls}(:,l) = d;
            end
        end
        fprintf('\n\n%s: %.3f %.3f\n\n',Labels{ls}{l},topC,topPN);
    end
    
    fprintf(fid,'\t\t');
    fprintf(fid,'%.3f Weighted\t',wts);
    fprintf(fid,'%.3f No Weights\t',wts);
    fprintf(fid,'\n');
    
    fprintf(fid,'Accuracy\t\t');
    fprintf(fid,'%.2f\t',mean(rates));
    fprintf(fid,'\n');
    fprintf(fid,'True Positive\t\t');
    fprintf(fid,'%.2f\t',mean(tp)*100);
    fprintf(fid,'\n');
    fprintf(fid,'True Negative\t\t');
    fprintf(fid,'%.2f\t',mean(tn)*100);
    fprintf(fid,'\n');
    
    fprintf(fid,'Accuracy\n');
    for l = 1:numLabels
        tl = trainLabels(:,l); tl(tl==-1) = 2;
        [a b] = UniqueAndCounts(tl);
        fprintf(fid,'%s\t%d\t',Labels{ls}{l},b(1));
        fprintf(fid,'%.2f\t',rates(l,:));
        fprintf(fid,'\n');
    end
    fprintf(fid,'True Positive\n');
    for l = 1:numLabels
        tl = trainLabels(:,l); tl(tl==-1) = 2;
        [a b] = UniqueAndCounts(tl);
        fprintf(fid,'%s\t%d\t',Labels{ls}{l},b(1));
        fprintf(fid,'%.2f\t',tp(l,:)*100);
        fprintf(fid,'\n');
    end
    fprintf(fid,'True Negative\n');
    for l = 1:numLabels
        tl = trainLabels(:,l); tl(tl==-1) = 2;
        [a b] = UniqueAndCounts(tl);
        fprintf(fid,'%s\t%d\t',Labels{ls}{l},b(1));
        fprintf(fid,'%.2f\t',tn(l,:)*100);
        fprintf(fid,'\n');
    end
    
end
fclose(fid);

