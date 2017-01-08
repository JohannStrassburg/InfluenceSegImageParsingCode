function [globalSVM tpr fpr cr AUC] = TrainGlobalSVMBasic(trainDesc, trainLabels, valDesc, valLabels, testDesc, testLabels,  descList, tprThresh)

kernelList = cell(0);%{'gaussian';'intersection';'intersection'};
trainData = cell(0);%{trainGlobalDesc.colorGist;trainGlobalDesc.coHist;trainGlobalDesc.spatialPry};
valData = cell(0);%{trainGlobalDesc.colorGist;trainGlobalDesc.coHist;trainGlobalDesc.spatialPry};
testData = cell(0);%{testGlobalDesc.colorGist;testGlobalDesc.coHist;testGlobalDesc.spatialPry};
descStr = '';
tpr = zeros(size(trainLabels,2),3);
fpr = zeros(size(trainLabels,2),3);
cr = zeros(size(trainLabels,2),1);
AUC = zeros(size(trainLabels,2),1);

if(~exist('tprThresh','var'))
    tprThresh = 0;
end

globalSVM = cell(size(trainLabels,2),1);
for i = 1:length(descList)
    trainData{i,1} = trainDesc.(descList{i});
    valData{i,1} = valDesc.(descList{i});
    testData{i,1} = testDesc.(descList{i});
    if(strcmp(descList{i},'colorGist')|| strcmp(descList{i},'svm')|| strcmp(descList{i},'svmsig'))
        kernelList{i,1} = 'gaussian';
    else
        kernelList{i,1} = 'intersection';
    end
    %subsample to keep in memory
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

ps  =  zeros(size(trainData{1},1));	
for i=1:length(trainData)
    pstemp=svmkernel(trainData{i},kernelList{i},kerneloptionList{i});
    ps=ps+pstemp;
end

paraList = 2;
epsilon = .000001;
verbose = 0;
for l = 1:size(trainLabels,2)
    SVM = [];
    [SVM.xsupList,SVM.w,SVM.b,SVM.pos]=svmclassMK(trainData,trainLabels(:,l),paraList,epsilon,kernelList,kerneloptionList,verbose,[],[],ps);
    SVM.kernelList=kernelList;
    SVM.kerneloptionList=kerneloptionList;
    if(~isempty(valLabels))
        [AUC(l),tpr(l,2),fpr(l,2),SVM.b]=SVMROCMKThresh(valData,valLabels(:,l),SVM,tprThresh);
    else
        [AUC(l),tpr(l,2),fpr(l,2),SVM.b]=SVMROCMKThresh(trainData,trainLabels(:,l),SVM,tprThresh);
    end
    
    if(nargout > 1)
        ypred = svmvalMK(trainData,SVM.xsupList,SVM.w,SVM.b,SVM.kernelList,SVM.kerneloptionList);
        results=ones(size(trainLabels,1),1);results(ypred<=0)=0;
        tpr(l,1) = sum(results(trainLabels(:,l)==1))./sum(trainLabels(:,l)==1);
        fpr(l,1) = sum(results(trainLabels(:,l)==-1))./sum(trainLabels(:,l)==-1);

        ypred = svmvalMK(testData,SVM.xsupList,SVM.w,SVM.b,SVM.kernelList,SVM.kerneloptionList);
        results=ones(size(testLabels,1),1);results(ypred<=0)=0;
        tpr(l,3) = sum(results(testLabels(:,l)==1))./sum(testLabels(:,l)==1);
        fpr(l,3) = sum(results(testLabels(:,l)==-1))./sum(testLabels(:,l)==-1);
        cr(l) = sum(results==(testLabels(:,l)==1))./length(results);
    end
    
    globalSVM{l} = SVM;
end



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
[indiceTresh]=find((tprThresh-tpr)>0,1); indiceTresh=indiceTresh-1;
if(isempty(indiceTresh)) indiceTresh = length(b); end
%fprintf('Moving b: %.4f -> %.4f tpr: %.2f -> %.2f\n', w0-b(indice),w0-b(indiceTresh),tpr(indice),tpr(indiceTresh));

b=w0-b(min(indice,indiceTresh)) + eps;
tpr = tpr(min(indice,indiceTresh));
fpr = fpr(min(indice,indiceTresh));

