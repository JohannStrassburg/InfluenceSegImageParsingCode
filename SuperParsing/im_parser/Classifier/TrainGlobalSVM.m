function [globalSVM globalSVMT globalSVMRaw] = TrainGlobalSVM(HOMEDATA, HOMELABELSETS, Labels, trainIndexs, trainGlobalDesc, testIndexs, testGlobalDesc, descList, testSet, tprThresh)

kernelList = cell(0);%{'gaussian';'intersection';'intersection'};
trainData = cell(0);%{trainGlobalDesc.colorGist;trainGlobalDesc.coHist;trainGlobalDesc.spatialPry};
testData = cell(0);%{testGlobalDesc.colorGist;testGlobalDesc.coHist;testGlobalDesc.spatialPry};
descStr = '';
global trainPrune;
if(~exist('testSet','var'))
    testSet = '';
elseif(isnumeric(testSet))
    testSet = num2str(testSet);
end

if(~exist('tprThresh','var'))
    tprThresh = 0;
end

globalSVM = cell(size(HOMELABELSETS));
globalSVMT = cell(size(HOMELABELSETS));
globalSVMRaw = cell(size(HOMELABELSETS));
for i = 1:length(descList)
    trainData{i,1} = trainGlobalDesc.(descList{i});
    testData{i,1} = testGlobalDesc.(descList{i});
    descStr = [descStr descList{i}(1:3) descList{i}(end-2:end)];
    if(strcmp(descList{i},'colorGist'))
        kernelList{i,1} = 'gaussian';
    else
        kernelList{i,1} = 'intersection';
    end
    %subsample to keep in memory
end
maxIms = 5000;
if(size(trainData{1,1},1)>maxIms)
    usedIms = [];
    i = 1;
    j = 1;
    [ls counts] = UniqueAndCounts(trainIndexs{i}.label);
    [counts ls] = sort(counts);
    while(length(usedIms) < maxIms)
        iminds = unique(trainIndexs{i}.image(trainIndexs{i}.label==ls(j)));
        randind = randperm(length(iminds));
        usedIms = union(usedIms,iminds(randind(1:min(end,25))));
        j = j + 1;
        if(j>length(ls))
            j = 1;
            i = i + 1;
            if(i>length(trainIndexs))
                i = 1;
            end
            [ls counts] = UniqueAndCounts(trainIndexs{i}.label);
            [counts ls] = sort(counts);
        end
    end
    for j = 1:length(trainIndexs)
        trainPrune{j} = PruneIndex(trainIndexs{j}, usedIms, length(usedIms), 0);
        trainData{j,1} = trainData{j,1}(usedIms,:);
    end
else
    trainPrune = trainIndexs;
end

goodCount = 0;
saveFile = cell(size(HOMELABELSETS));
for ls = 1:length(HOMELABELSETS)
    HOMELABELSET = HOMELABELSETS{ls};
    [fo labelSet] = fileparts(HOMELABELSET);
    if(tprThresh>0)
        saveFile{ls} = fullfile(HOMEDATA,'Classifier','SVMValT',[descStr labelSet testSet '-' num2str(maxIms) '-' num2str(100*tprThresh) '.mat']);
    else
        saveFile{ls} = fullfile(HOMEDATA,'Classifier','SVMValT',[descStr labelSet testSet '-' num2str(maxIms) '.mat']);
    end
    if(exist(saveFile{ls},'file'))
        load(saveFile{ls});
        globalSVM{ls} = globalSVMLS;
        if(exist('globalSVMLST','var')); globalSVMT{ls} = globalSVMLST; end
        globalSVMRaw{ls} = globalSVMRawLS;
        goodCount = goodCount+1;
    end
end
if(goodCount==length(HOMELABELSETS))
    return;
end

kerneloptionList=cell(length(trainData),1);
for i=1:length(trainData)
    % estimate the bandwidth for the gaussian kernel
    if strcmp(kernelList{i,1},'gaussian')
        x=trainData{i};
        index=randperm(size(x,1));
        xDis = dist2(x(index(1:100),:), x);
        xSort=sort(xDis,2);
        xSort=xSort(:,10);
        kerneloptionList{i}=mean(xSort);
    else
        kerneloptionList{i}=1;
    end
end
%global ps;
%if(isempty(ps))
    ps  =  zeros(size(trainData{1},1));	
    for i=1:length(trainData)
        pstemp=svmkernel(trainData{i},kernelList{i},kerneloptionList{i});
        ps=ps+pstemp;
    end
%end

for ls = 1:length(HOMELABELSETS)
    HOMELABELSET = HOMELABELSETS{ls};
    [fo labelSet] = fileparts(HOMELABELSET);
    if(exist(saveFile{ls},'file'))
        load(saveFile{ls});
        globalSVM{ls} = globalSVMLS;
        globalSVMT{ls} = globalSVMLST;
        globalSVMRaw{ls} = globalSVMRawLS;
        continue;
    end

    trainIndex = trainPrune{ls};
    imInds = unique(trainIndex.image);
    immap = zeros(max(imInds),1);
    immap(imInds) = 1:length(imInds);
    trainIndex.image = immap(trainIndex.image);
    testIndex = testIndexs{ls};
    labels = unique(trainIndex.label);
    trainIms = unique(trainIndex.image);
    trainIm2Ndx = zeros(max(trainIms),1);trainIm2Ndx(trainIms) = 1:length(trainIms);
    numIms = length(trainIms);
    psTrain = ps(trainIms,:);psTrain = psTrain(:,trainIms);    
    testIms = unique(testIndex.image);
    testIm2Ndx = zeros(max(testIms),1);testIm2Ndx(testIms) = 1:length(testIms);
    numTestIms = length(testIms);
    
    trainDataLS = trainData;
    testDataLS = testData;
    for i = 1:length(trainData)
        trainDataLS{i} = trainData{i}(trainIms,:);
        testDataLS{i} = testData{i}(testIms,:);
    end

    globalSVMLS = cell(max(labels),1);
    globalSVMLST = cell(max(labels),1);
    globalSVMRawLS = cell(max(labels),1);
    bad = 1;
    pfig = ProgressBar(HOMELABELSET);
    for l = labels(:)'
        imageNDX = unique(trainIndex.image(trainIndex.label==l));
        if(isempty(imageNDX) || length(imageNDX) > numIms-10)
            continue;
        end
        bad = 0;
        labeling = -ones(numIms,1);
        labeling(trainIm2Ndx(imageNDX)) = 1;
        imageNDX = unique(testIndex.image(testIndex.label==l));
        testLabeling = -ones(numTestIms,1);
        testLabeling(testIm2Ndx(imageNDX)) = 1;
        [SVMS SVMST SVMSRaw rates ratesT] = TrainOneSVM(labeling,trainDataLS,testLabeling,testDataLS,kernelList,kerneloptionList,psTrain,tprThresh);
        score = rates(:,1)+(1-rates(:,2))+rates(:,3);
        [foo ind] = max(score);[ind] = find(score==foo);
        [foo bar] = min(abs(ind-length(score)/2));ind = ind(bar);
        globalSVMLS{l} = SVMS{ind};
        globalSVMRawLS{l} = SVMSRaw{ind};
        fprintf('%d %s: Correct P: %.2f False P: %.2f AUC: %.2f B: %f Braw: %f\n',ind,Labels{ls}{l},rates(ind,:),globalSVMLS{l}.b,globalSVMRawLS{l}.b);
        score = ratesT(:,1)+(1-ratesT(:,2))+ratesT(:,3);
        [foo ind] = max(score);[ind] = find(score==foo);
        [foo bar] = min(abs(ind-length(score)/2));ind = ind(bar);
        globalSVMLST{l} = SVMST{ind};
        fprintf('%d %s: Correct P: %.2f False P: %.2f AUC: %.2f B: %f test\n',ind,Labels{ls}{l},ratesT(ind,:),globalSVMLST{l}.b);
        ProgressBar(pfig,find(l==labels),length(labels));
    end
    close(pfig);
    if(~bad)
        make_dir(saveFile{ls});
        save(saveFile{ls},'globalSVMLS','globalSVMLST','globalSVMRawLS');
        globalSVM{ls} = globalSVMLS;
        globalSVMT{ls} = globalSVMLST;
        globalSVMRaw{ls} = globalSVMRawLS;
    end
end


function [SVMS SVMST SVMSRaw rates ratesT] = TrainOneSVM(labeling,trainData,testLabeling,testData,kernelList,kerneloptionList,ps,tprThresh)

paraList = [.02 .2 2 20 200];
epsilon = .000001;
verbose = 0;

rates=[];
ratesT=[];
SVMS = cell(length(paraList),1);
SVMST = cell(length(paraList),1);
SVMSRaw = cell(length(paraList),1);
for i=1:length(paraList)
    [xsupList,w,b,pos]=svmclassMK(trainData,labeling,paraList(i),epsilon,kernelList,kerneloptionList,verbose,[],[],ps);
    SVM.xsupList=xsupList;
    SVM.w=w;
    SVM.b=b;
    SVM.pos=pos;
    SVM.kernelList=kernelList;
    SVM.kerneloptionList=kerneloptionList;
    SVMSRaw{i} = SVM;
    [SVMS{i} correctPosRate falsePosRate AUC]=testSVM(testData,testLabeling,SVM,tprThresh);
    rates=[rates;[correctPosRate, falsePosRate, AUC]];
    bothData = cell(size(trainData));
    for f = 1:length(trainData)
        bothData{f} = [trainData{f}; testData{f}];
    end
    bothLabeling = [labeling; testLabeling];
    [SVMST{i} correctPosRate falsePosRate AUC]=testSVM(bothData,bothLabeling,SVM,tprThresh);
    ratesT=[ratesT;[correctPosRate, falsePosRate, AUC]];
end


function [SVM,correctPosRate,falsePosRate,AUC,scores,results] = testSVM(testData,testLabeling,SVM,tprThresh)

correctPosRate = 0;
falsePosRate = 1;
AUC = 1;
scores = [];
results =[];
if(all(testLabeling==1)||all(testLabeling==-1))
    return;
end

iPos=find(testLabeling==1);
iNeg=find(testLabeling==-1);
numPos=size(iPos,1);
numNeg=size(iNeg,1);    

if(tprThresh==0)
    [AUC,tpr,fpr,b]=SVMROCMK(testData,testLabeling,SVM);
else
    [AUC,tpr,fpr,b]=SVMROCMKThresh(testData,testLabeling,SVM,tprThresh);
end
SVM.b=b;

ypred = svmvalMK(testData,SVM.xsupList,SVM.w,SVM.b,SVM.kernelList,SVM.kerneloptionList);

results=ones(size(testLabeling,1),1);
results(ypred<=0)=-1;

if numPos>0
    posItems=results(iPos) + testLabeling(iPos);
    indCorrect=find(posItems>0);
    correctPosRate = size(indCorrect,1)./numPos;
end

if numNeg>0
    negItems=results(iNeg) + testLabeling(iNeg);
    indFalse=find(negItems==0);
    falsePosRate=size(indFalse,1)./numNeg;
end

scores=ypred;

%fprintf('Correct Positive rate=%f\n', correctPosRate);
%fprintf('False Positive rate=%f\n', falsePosRate);

function [AUC,tpr,fpr,b]=SVMROCMKThresh(xtest,ytest,SVM,tprThresh)

ypred = svmvalMK(xtest,SVM.xsupList,SVM.w,SVM.b,SVM.kernelList,SVM.kerneloptionList);

w0=SVM.b;
npos=sum(ytest==1);
nneg=sum(ytest==-1);

ytest=ytest>0;
[ypred,ind] = sort(ypred);
ytest       = ytest(ind);    

tpr = 1-cumsum(ytest)/sum(ytest);
fpr = 1-cumsum(1-ytest)/sum(1-ytest);
tpr = [1 ; tpr ; 0];
fpr = [1 ; fpr ; 0];
n = size(tpr, 1);
AUC = sum((fpr(1:n-1) - fpr(2:n)).*(tpr(2:n)+tpr(1:n-1)))/2;
b=[min(ypred)-1;ypred];
[aux,indice]=min(abs(1-fpr-tpr)); % intersection entre la courbe roc et la diagonale.
[indiceTresh]=find((tprThresh-tpr)>0,1); indiceTresh=indiceTresh-1;% intersection entre la courbe roc et la diagonale.
fprintf('Moving b: %.4f -> %.4f tpr: %.2f -> %.2f\n', w0-b(indice),w0-b(indiceTresh),tpr(indice),tpr(indiceTresh));

b=w0-b(min(indice,indiceTresh)) + eps;
