
for ls = 2:length(Labels)
    [foo labelSet] = fileparts(HOMELABELSETS{ls});
    trIndex = trainIndex{ls};
    teIndex = testIndex{ls};
    
    [Ls Cts] = UniqueAndCounts(trIndex.label);
    numL = length(Ls);
    
    %{
    trLabels = trIndex.label';
    retSetSPDesc = LoadSegmentDesc(trainFileList,trIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(1),testParams.segSuffix);
    trfeatures = GetFeaturesForClassifier(retSetSPDesc);
    clear retSetSPDesc;
    rp = randperm(length(trLabels));
    rp2 = randperm(size(trfeatures,2));
    trfeatures = trfeatures(:,:);
    trLabels = trLabels(:);
    
    teLabels = teIndex.label';
    retSetSPDesc = LoadSegmentDesc(testFileList,teIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(1),testParams.segSuffix);
    tefeatures = GetFeaturesForClassifier(retSetSPDesc);
    clear retSetSPDesc;
    tefeatures = tefeatures(:,:);
    %}
    
    extra_options.print_verbose_tree_progression = 1;
    extra_options.classwt = ones(1,numL);
    model = classRF_train(trfeatures,trLabels,0,0,extra_options);
    [teLabelsPred a b c] = classRF_predict(tefeatures,model);
    
    ppr = sum(teIndex.spSize(teLabelsPred==teLabels))./sum(teIndex.spSize);
    fprintf('%.2f: Per-pixel\n%.2f: Per-sp\n',ppr,sum(teLabelsPred==teLabels)./length(teLabels));
    lRate = zeros(size(Ls));
    for l = Ls(:)'
        lRate(l) = sum(teIndex.spSize(teLabels==l)'.*(teLabelsPred(teLabels==l)==l))./sum(teIndex.spSize(teLabels==l));
        fprintf('%.2f: %s\n',lRate(l),Labels{ls}{l});
    end
end


