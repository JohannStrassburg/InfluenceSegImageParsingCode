fid = -1;
%fid = fopen(fullfile(HOMEDATA,'svmconmat.txt'),'w');

useMCC = mcc;

useIndex = trainIndex;
useFileList = trainFileList;
useGlobalDesc = trainGlobalDesc;


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
        [svmLabels] =  mnrval(useMCC,useGlobalDesc.colorGist);
        svmLabels(isnan(svmLabels)) = 1;
        svmLabels(svmLabels>.5) = 1;svmLabels(svmLabels<=.5)=0;
        %[svmLabels fs] = svmShortList(useSVM{labelType},useGlobalDesc,testParams.SVMDescs,0);
        [foo topLabels] = max(fs,[],2);
        fprintf('\n');
        
        correct = svmLabels==trueLabels;

        for l = labels(:)'
            fprintf('rate: %.2f  ',sum(correct(:,l))./numIms);
            fprintf('tpr: %.2f  ',sum(correct(trueLabels(:,l),l))./sum(trueLabels(:,l)));
            fprintf('tnr: %.2f %s\n',sum(correct(~trueLabels(:,l),l))./sum(~trueLabels(:,l)),Labels{labelType}{l});
        end
        fprintf('%.2f %s\n\n',100*sum(correct(trueLabels))./sum(trueLabels(:)),labelTypeName);
        
        [trueTopLabels foo] = find(trueLabels');
        if(numel(trueTopLabels) == numel(topLabels))
            correctTop = topLabels==trueTopLabels;
            fprintf('%.2f %s\n\n',100*sum(correctTop(:))./numel(correctTop),labelTypeName);
        end
    end
end
%fclose(fid);