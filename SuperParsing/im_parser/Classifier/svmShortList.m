function [shortListInds, fs] = svmShortList(globalSVM,globalDesc,descList,thresh)
if(~exist('thresh','var'));thresh = 0;end

kernelList = cell(0);
testData = cell(0);
descStr = '';
for i = 1:length(descList)
    testData{i,1} = globalDesc.(descList{i});
    descStr = [descStr descList{i}(1:3) descList{i}(end-2:end)];
    if(strcmp(descList{i},'colorGist') || strcmp(descList{i},'svm')|| strcmp(descList{i},'svmsig'))
        kernelList{i,1} = 'gaussian';
    else
        kernelList{i,1} = 'intersection';
    end
end

fs = zeros(size(testData{1},1),length(globalSVM));
shortListInds = zeros(size(testData{1},1),length(globalSVM));
for i = 1:length(globalSVM)
    SVM = globalSVM{i};
    if(isempty(SVM))
        fs(i) = 0;
        shortListInds(i) = true;
        continue;
    end
    if(sum(strcmp(SVM.kernelList,kernelList))~=length(kernelList))
        fprintf('ERROR WRONG Kernel List\n');
        %keyboard;
    end
    fs(:,i) = svmvalMK(testData,SVM.xsupList,SVM.w,SVM.b,SVM.kernelList,SVM.kerneloptionList);
    %[class fs(i)] = svmdecision(globalDesc.spatialPry,globalSVM{i}); 
    shortListInds(:,i) = fs(:,i)>=thresh;
end
if(sum(shortListInds)==0)
    [foo ind] = max(fs);
    shortListInds(ind) = 1;
end
if(thresh>0)
    shortListInds = zeros(size(globalSVM));
    [foo ind] = sort(fs,'descend');
    shortListInds(ind(1:min(fix(thresh),length(ind)))) = 1;
end
