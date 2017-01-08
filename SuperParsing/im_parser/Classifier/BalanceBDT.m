
if(~exist('clout','var'))
    clouttemp = cell(size(valFileList));%,size(HOMELABELSETS));
    gtlabeltemp = cell(size(valFileList));%,size(HOMELABELSETS));
    for i = 1:length(valFileList)
        clouttemp{i} = cell(size(HOMELABELSETS));
        gtlabeltemp{i} = cell(size(HOMELABELSETS));
    end
    
    tic
    %matlabpool(5);
    parfor i = 1:length(valFileList)
        fprintf('%d of %d\n',i,length(valFileList));
        im = imread(fullfile(HOMEIMAGES,valFileList{i}));
        [fold file ext] = fileparts(valFileList{i});
        [valImSPDesc imSP] = LoadSegmentDesc(valFileList(i),[],HOMEDATA,claParams.segmentDescriptors,K);
        features = GetFeaturesForClassifier(valImSPDesc);

        for ls = 1:length(HOMELABELSETS)
            prob = test_boosted_dt_mc(classifiers{ls}, features);
            ndx = find(valIndex{ls}.image == i);
            sps = valIndex{ls}.sp(ndx);
            labels = valIndex{ls}.label(ndx);
            prob = prob(sps,:);
            clouttemp{i}{ls} = prob;
            gtlabeltemp{i}{ls} = labels;
        end
    end
    
    clout = cell(size(HOMELABELSETS));
    gtlabel = cell(size(HOMELABELSETS));
    for i = 1:length(valFileList)
        for ls = 1:length(HOMELABELSETS)
            clout{ls} = [clout{ls}; clouttemp{i}{ls}];
            gtlabel{ls} = [gtlabel{ls} gtlabeltemp{i}{ls}];
        end
    end
end


for ls = 1:length(HOMELABELSETS)
    for l = 1:length(Labels{ls})
        fprintf('%s ',Labels{ls}{l});
        classifiers{ls}.h0(l) = ROCBalance(clout{ls}(:,l),gtlabel{ls}'==l);
    end
end