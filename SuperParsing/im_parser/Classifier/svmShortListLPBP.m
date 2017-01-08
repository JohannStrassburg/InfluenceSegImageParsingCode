function [sl] = svmShortListLPBP(ittResult,globalDesc,descList)
labels = unique(ittResult.cluster);labels(labels==0) = [];
numLabels = max(labels);

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
        ftemp = zeros(numIms,length(ittResult.svm{c}));
        for i = 1:length(ittResult.svm{c})
            SVM = ittResult.svm{c}{i};
            ftemp(:,i) = svmvalMK(testData,SVM.xsupList,SVM.w,SVM.b,SVM.kernelList,SVM.kerneloptionList);
            if(all(ftemp(:,i)==ftemp(1,i)))
                ftemp(:,i) = -10;
            end
        end
        [fo ind] = max(ftemp,[],2);
        sl(:,clabels) = ittResult.combPattern{c}(ind,:)==1;
        fs(:,clabels) = repmat(fo,[1 length(clabels)]);
    end
end