DataDir = fullfile(HOMEDATA,testParams.TestString);
%{-
for ls = 1:length(UseLabelSet)
    labelList = Labels{UseLabelSet(ls)};
    curFold = [];
    i = 1;
    [foo labelSet] = fileparts(HOMELABELSETS{UseLabelSet(ls)});
    suffix = ['k' num2str(K) segSuffix '-CL' num2str(UseClassifier) '_' claParams.init_weight];
    while i <= length(testFileList)
        [fold base] = fileparts(testFileList{i});
        curFold = fold;
        spProbs = cell(0);
        spSizes = cell(0);
        j = i;
        while strcmp(curFold,fold)
            spNdx = unique(parsingData{j}.imSP);
            if(length(spProbs)<max(spNdx)); spProbs{max(spNdx)} = []; spSizes{max(spNdx)} = []; end
            for s = 1:length(spNdx)
                spProbs{spNdx(s)} = [spProbs{spNdx(s)}; parsingData{j}.probPerLabel{ls}(s,:)];
                spSizes{spNdx(s)} = [spSizes{spNdx(s)}; sum(parsingData{j}.imSP(:)==spNdx(s))];
            end
            j = j+1;
            if(j>length(testFileList))
                break;
            end
            [fold base] = fileparts(testFileList{j});
        end
        spMax = zeros(size(spProbs,2),length(labelList));
        spMean = zeros(size(spProbs,2),length(labelList));
        spSoftMax = zeros(size(spProbs,2),length(labelList));
        for k = 1:length(spProbs)
            if(~isempty(spProbs{k}))
                spMax(k,:) = max(spProbs{k},[],1);
                spMean(k,:) = mean(spProbs{k},1);
                spSoftMax(k,:) = prod(softmax(spProbs{k}'),2);
            end
        end
        [foo maxL] = max(spMax,[],2);
        [meanX meanL] = max(spMean,[],2);
        [softMaxX softMaxL] = max(spSoftMax,[],2);
        if(any(softMaxL~=meanL))
            fprintf('They aren''t the same\n');
            keyboard;
        end
        for k = i:j-1 
            [fold base] = fileparts(testFileList{k});

            outfile = fullfile(DataDir,testParams.MRFFold,labelSet,['Max' suffix],fold,[base '.mat']);make_dir(outfile);
            L = maxL(parsingData{k}.imSP);
            Lsp = maxL(unique(parsingData{k}.imSP));
            save(outfile,'L','labelList','Lsp');

            outfile = fullfile(DataDir,testParams.MRFFold,labelSet,['Mean' suffix],fold,[base '.mat']);make_dir(outfile);
            L = meanL(parsingData{k}.imSP);
            Lsp = meanL(unique(parsingData{k}.imSP));
            save(outfile,'L','labelList','Lsp');

            outfile = fullfile(DataDir,testParams.MRFFold,labelSet,['SoftMax' suffix],fold,[base '.mat']);make_dir(outfile);
            L = softMaxL(parsingData{k}.imSP);
            Lsp = softMaxL(unique(parsingData{k}.imSP));
            save(outfile,'L','labelList','Lsp');
        end
        i = j;
    end
end
%}
