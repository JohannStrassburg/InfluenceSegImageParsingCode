function fs = svmRun(svm,globalDesc,descList)
kernelList = cell(0);
testData = cell(0);
descStr = '';
for i = 1:length(descList)
    testData{i,1} = globalDesc.(descList{i});
    descStr = [descStr descList{i}(1:3) descList{i}(end-2:end)];
    if(strcmp(descList{i},'colorGist') || strcmp(descList{i},'svm'))
        kernelList{i,1} = 'gaussian';
    else
        kernelList{i,1} = 'intersection';
    end
end

fs = zeros(size(testData{1},1),length(svm));
if(~iscell(svm))
    svm = {svm};
end
for i = 1:length(svm)
    svm1 = svm{i};
    if(isempty(svm1))
        fs(i) = 0;
        continue;
    end
    if(sum(strcmp(svm1.kernelList,kernelList))~=length(kernelList))
        fprintf('ERROR WRONG Kernel List\n');
        keyboard;
    end
    fs(:,i) = svmvalMK(testData,svm1.xsupList,svm1.w,svm1.b,svm1.kernelList,svm1.kerneloptionList);
end