function [level2GlobalSVM] = TrainMultiClassClassifierSVM(HOMEDATA, HOMELABELSETS, Labels, trainIndexs, trainGlobalDesc, globalSVM, testIndexs, testGlobalDesc, descList)



kernelList = cell(0);%{'gaussian';'intersection';'intersection'};
trainData = cell(0);%{trainGlobalDesc.colorGist;trainGlobalDesc.coHist;trainGlobalDesc.spatialPry};
testData = cell(0);%{testGlobalDesc.colorGist;testGlobalDesc.coHist;testGlobalDesc.spatialPry};
descStr = '';

for ls = 1:length(HOMELABELSETS)
    numTrainIms = max(trainIndexs{ls}.image);
    [foo trainData] = svmShortList(globalSVM{ls},trainGlobalDesc,descList,0);
    [foo testData] = svmShortList(globalSVM{ls},testGlobalDesc,descList,0);
    
    xDis = dist2(trainData, trainData);
    xSort=sort(xDis,2);
    xSort=xSort(:,10);
    kerneloption=1;%mean(xSort);
    ps = svmkernel(trainData,'gaussian',kerneloption);
        
    trainIndex = trainIndexs{ls};
    testIndex = testIndexs{ls};
    
    for l = 1:length(Labels{ls})
        imNdx = unique(trainIndex.image(trainIndex.label==l));
        labelsTrain = -1*ones(numTrainIms,1);labelsTrain(imNdx) = 1;
        imNdx = unique(testIndex.image(testIndex.label==l));
        labelsTest = -1*ones(numTrainIms,1);labelsTest(imNdx) = 1;
        [SVMS SVMST SVMSRaw rates ratesT] = TrainOneSVM(labelsTrain,{trainData},labelsTest,{testData},{'gaussian'},{kerneloption},ps,.9);
        score = rates(:,1)+(1-rates(:,2))+rates(:,3);
        [foo ind] = max(score);[ind] = find(score==foo);
        [foo bar] = min(abs(ind-length(score)/2));ind = ind(bar);
        level2GlobalSVM{ls}{l} = SVMS{ind};
        fprintf('%d %s: Correct P: %.2f False P: %.2f AUC: %.2f B: %f\n',ind,Labels{ls}{l},rates(ind,:), level2GlobalSVM{ls}{l}.b);
        
    end
end



function [SVMS SVMST SVMSRaw rates ratesT] = TrainOneSVM(labeling,trainData,testLabeling,testData,kernelList,kerneloptionList,ps,tprThresh)

paraList = [.0002 .002 .02 .2 2 20 200];
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