function SVMSoftMaxCutoff = rocNormSoftMax(HOMELABELSETS,globalSVM,valIndex)
useSVM = globalSVM;
SVMSoftMaxCutoff = cell(size(HOMELABELSETS));
for labelType=1:length(HOMELABELSETS)
    if(~isempty(useSVM{labelType}))
        [foo ltstr] = fileparts(HOMELABELSETS{labelType});
        saveFile = fullfile(HOMEDATA,'Classifier','SVMValT',[ltstr '-cutoff.mat']);
        if(exist(saveFile,'file'))
            load(saveFile);
            SVMSoftMaxCutoff{labelType} = cutoff;
            continue;
        end
        [fo labelTypeName] = fileparts(HOMELABELSETS{labelType});
        labels = unique(valIndex{labelType}.label);
        numIms = length(valFileList);

        trueLabels = zeros(numIms,max(labels))==1;
        for l = labels(:)'
            imageNDX = unique(valIndex{labelType}.image(valIndex{labelType}.label==l));
            trueLabels(imageNDX,l) = 1;
        end
        svmLabels = zeros(numIms,max(labels))==1;
        topLabels = zeros(numIms,1);
        %{-
        fs = zeros(numIms,length(labels));%,size(globalSVM,2));
        for i = 1:numIms
            fs(i,:) = svmShortListSoftMax(useSVM(labelType,:),SelectDesc(valGlobalDesc,i,1),svmDescs);
        end
        %}
        
        SVMSoftMaxCutoff{labelType} = zeros(size(fs,2),1);
        for l = 1:size(fs,2)
            truel = trueLabels(:,l);
            [fsl,ind] = sort(fs(:,l));
            truel = truel(ind);

            fpr = cumsum(truel)/sum(truel);
            tpr = cumsum(1-truel)/sum(1-truel);
            tpr = [0 ; tpr ; 1];
            fpr = [0 ; fpr ; 1];
            n = size(tpr, 1);
            AUC = sum((fpr(2:n) - fpr(1:n-1)).*(tpr(2:n)+tpr(1:n-1)))/2;
            b=[min(fsl)-1;fsl];
            [aux,indice]=min(abs(1-fpr-tpr)); % intersection entre la courbe roc et la diagonale.
            SVMSoftMaxCutoff{labelType}(l) = b(indice) + eps;
        end
        cutoff = SVMSoftMaxCutoff{labelType};
        save(saveFile,'cutoff');
    end
end