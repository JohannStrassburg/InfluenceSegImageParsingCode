
useIndex = testIndex;
useFileList = testFileList;
useGlobalDesc = testGlobalDesc;
useSoftMax = 1;

for labelType=1:length(HOMELABELSETS)
    if(~isempty(ittResult{labelType}))
        [fo labelTypeName] = fileparts(HOMELABELSETS{labelType});
        labels = unique(useIndex{labelType}.label);
        numIms = length(useFileList);

        trueLabels = zeros(numIms,max(labels))==1;
        for l = labels(:)'
            imageNDX = unique(useIndex{labelType}.image(useIndex{labelType}.label==l));
            trueLabels(imageNDX,l) = 1;
        end
        for i = 1:length(ittResult{labelType})
            svmLabels = zeros(numIms,max(labels))==1;
            topLabels = zeros(numIms,1);
            if(useSoftMax)
                [svmLabels] = svmShortListLPBPSoftMax(ittResult{labelType}(i),useGlobalDesc,testParams.SVMDescs);
            else
                [svmLabels] = svmShortListLPBP(ittResult{labelType}(i),useGlobalDesc,testParams.SVMDescs);
            end
            fprintf('Itt %d\n',i);

            correct = svmLabels==trueLabels;

            for l = labels(:)'
                fprintf('rate: %.2f  ',sum(correct(:,l))./numIms);
                fprintf('tpr: %.2f  ',sum(correct(trueLabels(:,l),l))./sum(trueLabels(:,l)));
                fprintf('tnr: %.2f %s\n',sum(correct(~trueLabels(:,l),l))./sum(~trueLabels(:,l)),Labels{labelType}{l});
            end
            fprintf('%.2f %s\n\n',100*sum(correct(trueLabels))./sum(trueLabels(:)),labelTypeName);
        end
    end
end
        