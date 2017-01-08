

fid = -1;
%fid = fopen(fullfile(HOMEDATA,'svmconmat.txt'),'w');

useSVM = svm;

useIndex = testIndex;
useFileList = testFileList;
useGlobalDesc = testGlobalDesc;


for labelType=1:length(HOMELABELSETS)
    if(~isempty(useSVM{labelType}))
        [fo labelTypeName] = fileparts(HOMELABELSETS{labelType});
        labels = unique(useIndex{labelType}.label);
        numIms = length(useFileList);

        trueLabels = zeros(numIms,max(labels))==1;
        for l = labels(:)'
            imageNDX = unique(useIndex{labelType}.image(useIndex{labelType}.label==l));
            trueLabels(imageNDX,l) = 1;
        end
        svmLabels = zeros(numIms,max(labels))==1;
        topLabels = zeros(numIms,1);
        %{-
        [svmLabels fs] = svmShortList(useSVM{labelType},useGlobalDesc,testParams.SVMDescs,0);
        [foo topLabels] = max(fs,[],2);
        fprintf('\n');
        
        correct = svmLabels==trueLabels;

        for l = labels(:)'
            fprintf('rate: %.2f  ',sum(correct(:,l))./numIms);
            fprintf('tpr: %.2f  ',sum(correct(trueLabels(:,l),l))./sum(trueLabels(:,l)));
            fprintf('tnr: %.2f %s\n',sum(correct(~trueLabels(:,l),l))./sum(~trueLabels(:,l)),Labels{labelType}{l});
        end
        fprintf('%.2f %s\n\n',100*sum(correct(:))./numel(correct),labelTypeName);
        
        %{
        for i = 1:size(svmLabels,1)
            if(svmLabels(i,1)==1 && trueLabels(i,15) == 1)
                im = imread(fullfile(HOMEIMAGES,testFileList{i}));
                show(im,1);
            end
        end
        %}
        %}
        
        [trueTopLabels foo] = find(trueLabels');
        if(numel(trueTopLabels) == numel(topLabels))
            correctTop = topLabels==trueTopLabels;
            fprintf('%.2f %s\n\n',100*sum(correctTop(:))./numel(correctTop),labelTypeName);
        end
        
        if(exist('level2GlobalSVM','var'))
            svmOutput = [];svmOutput.svm = fs;
            svmOutput.svmsig = 1./(1+exp(-svmOutput.svm./sigmaSVM));
            [svmLabels fsl2] = svmShortList(level2GlobalSVM{labelType},svmOutput,{'svm'},0);
            correct = svmLabels==trueLabels;
            for l = labels(:)'
                fprintf('rate: %.2f  ',sum(correct(:,l))./numIms);
                fprintf('tpr: %.2f  ',sum(correct(trueLabels(:,l),l))./sum(trueLabels(:,l)));
                fprintf('tnr: %.2f %s\n',sum(correct(~trueLabels(:,l),l))./sum(~trueLabels(:,l)),Labels{labelType}{l});
            end
            fprintf('%.2f %s\n\n',100*sum(correct(:))./numel(correct),labelTypeName);

            [trueTopLabels foo] = find(trueLabels');
            if(numel(trueTopLabels) == numel(topLabels))
                correctTop = topLabels==trueTopLabels;
                fprintf('%.2f %s\n\n',100*sum(correctTop(:))./numel(correctTop),labelTypeName);
            end
        end
        
        if(exist('globalBoost','var'))
            for i = 1:size(globalBoost{ls},1)
                if(isempty(globalBoost{ls}{i,1}))continue;end
                for l = 1:size(globalBoost{ls},2)
                    [L,hits_te] = ADABOOST_te(globalBoost{ls}{i,l},@threshold_te,fs,trueLabels(:,l)+1);
                    svmLabels(:,l) = L(:,2)>globalBoost{ls}{i,l}.b;
                end
                fprintf('Boost %d\n',i);
                
                correct = svmLabels==trueLabels;
                for l = labels(:)'
                    fprintf('rate: %.2f  ',sum(correct(:,l))./numIms);
                    fprintf('tpr: %.2f  ',sum(correct(trueLabels(:,l),l))./sum(trueLabels(:,l)));
                    fprintf('tnr: %.2f %s\n',sum(correct(~trueLabels(:,l),l))./sum(~trueLabels(:,l)),Labels{labelType}{l});
                end
                fprintf('%.2f %s\n\n',100*sum(correct(:))./numel(correct),labelTypeName);

                [trueTopLabels foo] = find(trueLabels');
                if(numel(trueTopLabels) == numel(topLabels))
                    correctTop = topLabels==trueTopLabels;
                    fprintf('%.2f %s\n\n',100*sum(correctTop(:))./numel(correctTop),labelTypeName);
                end
            end
        end
        
        %{
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
        %}
    end
end
%fclose(fid);