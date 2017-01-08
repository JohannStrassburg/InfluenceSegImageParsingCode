function [sl] = svmShortListLPBPSoftMax(ittResult,globalDesc,descList)
labels = unique(ittResult.cluster);labels(labels==0) = [];
numLabels = max(labels);
numDesc = length(descList);

testData = cell(0);
for i = 1:length(descList)
    testData{i,1} = globalDesc.(descList{i});
end
numIms = size(testData{1},1);
sl = zeros(numIms,numLabels);
fs = zeros(numIms,numLabels);

for c = 1:length(ittResult.svm)
    clabels = ittResult.cluster(c,:);clabels(clabels==0) = [];
    if(length(clabels)==1)
        SVM = ittResult.svm{c};
        fs(:,clabels) = svmvalMK(testData,SVM.xsupList,SVM.w,SVM.b,SVM.kernelList,SVM.kerneloptionList);
        sl(:,clabels) = fs(:,clabels)>=0;
    else
        ftemp = zeros(numIms,length(ittResult.svm{c}{1}),length(ittResult.svm{c}));
        for d = 1:length(ittResult.svm{c})
            for i = 1:length(ittResult.svm{c}{1})
                SVM = ittResult.svm{c}{d}{i};
                ftemp(:,i,d) = svmvalMK(testData(d),SVM.xsupList,SVM.w,SVM.b,SVM.kernelList,SVM.kerneloptionList);
                if(all(ftemp(:,i)==ftemp(1,i)))
                    ftemp(:,i) = -10;
                end
            end
        end
        softMaxFS = prod(exp(ftemp)./repmat(sum(exp(ftemp),2),[1 size(ftemp,2) 1]),3);
        [fo ind] = max(softMaxFS,[],2);
        sl(:,clabels) = ittResult.combPattern{c}(ind,:)==1;
        fs(:,clabels) = repmat(fo,[1 length(clabels)]);
    end
end
