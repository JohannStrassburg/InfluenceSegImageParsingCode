
%{
globalSVM = cell(size(Labels));
%globalSVMT = cell(size(Labels));
globalSVMRaw = cell(size(Labels));
svmDescs = {'colorGist'};%,'coHist','spatialPry'
UseGlobalSVM = 2:3;
globalSVM = cell(length(Labels),length(svmDescs));
for i = 1:length(svmDescs)
    [globalSVMt]= TrainGlobalSVM(HOMEDATA, HOMELABELSETS(UseGlobalSVM), Labels(UseGlobalSVM), trainIndex(UseGlobalSVM), trainGlobalDesc, valIndex(UseGlobalSVM), valGlobalDesc, svmDescs(i), testSetNum);
    globalSVM(UseGlobalSVM,i) = globalSVMt;
    clear globalSVMt;
end
%}




fid = fopen(fullfile(HOMEDATA,'softMax-row-gist.txt'),'w');

useSVM = globalSVM;

for labelType=1:length(HOMELABELSETS)
    if(~isempty(useSVM{labelType}))
        [fo labelTypeName] = fileparts(HOMELABELSETS{labelType});
        labels = unique(testIndex{labelType}.label);
        numIms = length(testFileList);

        trueLabels = zeros(numIms,max(labels))==1;
        for l = labels(:)'
            imageNDX = unique(testIndex{labelType}.image(testIndex{labelType}.label==l));
            trueLabels(imageNDX,l) = 1;
        end
        svmLabels = zeros(numIms,max(labels))==1;
        topLabels = zeros(numIms,1);
        fs = zeros(numIms,length(labels));%,size(globalSVM,2));
        for i = 1:numIms
            %{
            for j = 1:size(globalSVM,2)
                [svmLabels(i,:) fs(i,:,j)] = svmShortList(useSVM{labelType,j},SelectDesc(testGlobalDesc,i,1),svmDescs(j),1);
                %[foo topLabels(i)] = max(fs);
            end
            %}
            fs(i,:) = svmShortListSoftMax(useSVM(labelType,:),SelectDesc(testGlobalDesc,i,1),svmDescs);
        end
        
        
        %{
        softMaxScore = zeros(numIms,length(labels));
        for i = 1:size(globalSVM,2)
            softMaxScore = softMaxScore + log(softmax(fs(:,:,i)')');
        end
        %}
        
        [foo topLabels] = max(fs,[],2);
        svmLabels = zeros(numIms,max(labels))==1;
        svmLabels(sub2ind(size(svmLabels),(1:numIms)', topLabels)) = 1;

        correct = svmLabels==trueLabels;

        for l = labels(:)'
            fprintf('%.2f  ',100*sum(correct(:,l))./numIms);
            fprintf('%.2f  ',100*sum(correct(trueLabels(:,l),l))./sum(trueLabels(:,l)));
            fprintf('%.2f %s:%s\n',100*sum(correct(~trueLabels(:,l),l))./sum(~trueLabels(:,l)),labelTypeName,Labels{labelType}{l});
        end
        fprintf('%.2f %s\n\n',100*sum(correct(:))./numel(correct),labelTypeName);
        
        [trueTopLabels foo] = find(trueLabels');
        if(numel(trueTopLabels) == numel(topLabels))
            correctTop = topLabels==trueTopLabels;
            fprintf('%.2f %s\n\n',100*sum(correctTop(:))./numel(correctTop),labelTypeName);
            fprintf(fid,'%.2f %s\n\n',100*sum(correctTop(:))./numel(correctTop),labelTypeName);
        end
        
        conMat = zeros(size(svmLabels,2));
        for l1 = 1:size(svmLabels,2)
            for l2 = 1:size(svmLabels,2)
                conMat(l1,l2) = sum(svmLabels(trueLabels(:,l1),l2));
            end
        end
        label = Labels{labelType};
        labelCounts = sum(conMat,2);
        label(labelCounts==0) = [];
        conMat(labelCounts==0,:) = [];
        conMat(:,labelCounts==0) = [];
        labelCounts(labelCounts==0) = [];
        [labelCounts sinds] = sort(labelCounts,'descend');
        conMat = conMat(sinds,:);
        conMat = conMat(:,sinds);
        label = label(sinds);
        conMat = conMat./repmat(labelCounts,[1 size(conMat,2)]);

        fprintf(fid,'Class\t# of Testing Examples\t');
        for j = 1:length(label)
            fprintf(fid,'%s\t',label{j});
        end
        fprintf(fid,'\n');
        for j = 1:length(label)
            fprintf(fid,'%s\t%d\t',label{j},labelCounts(j));
            for k = 1:length(label)
                fprintf(fid,'%.2f\t',conMat(j,k));
            end
            fprintf(fid,'\n');
        end
    end
end
fclose(fid);