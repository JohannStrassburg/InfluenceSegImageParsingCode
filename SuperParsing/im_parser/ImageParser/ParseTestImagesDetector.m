function timing = ParseTestImagesDetector(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex,trainCounts,labelPenality,Labels,classifiers,globalSVM,testParams,fullSPDesc)

DataDir = fullfile(HOMETESTDATA,testParams.TestString);
if(~exist('fullSPDesc','var'))
    fullSPDesc = cell(length(HOMELABELSETS),length(testParams.K));
end
probType = 'ratio';
if(isfield(testParams,'probType'))
    probType = testParams.probType;
end

if(~isfield(testParams,'segSuffix'))
    testParams.segSuffix = '';
end
%close all;
pfig = ProgressBar('Parsing Images');
range = 1:length(testFileList);
if(isfield(testParams,'range'))
    range = testParams.range;
end
timing = zeros(length(testFileList),4);
glSuffix = '';
for i = range
    im = imread(fullfile(HOMEDATA,'..','Images',testFileList{i}));
    [ro co ch] = size(im);
    [folder file] = fileparts(testFileList{i});
    baseFName = fullfile(folder,file);
    
    timing(i,1)=toc;  [retInds rank]= FindRetrievalSet(trainGlobalDesc,SelectDesc(testGlobalDesc,i,1),DataDir,baseFName,testParams,glSuffix);  timing(i,1)=toc-timing(i,1);
    
    svmstr = '';
    
    shortListMask = cell(size(Labels));
    for labelType=1:length(HOMELABELSETS)
        if(~isempty(globalSVM{labelType}))
            svmstr = testParams.SVMType;
            if(isfield(testParams,'SVMSoftMaxCutoff'))
                fs = svmShortListSoftMax(globalSVM(labelType,:),SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs);
                shortListInds = fs>=testParams.SVMSoftMaxCutoff{labelType};
                if(all(~shortListInds))
                    [foo ind] = max(fs);
                    shortListInds(ind) = 1;
                end
            elseif(strcmp(testParams.SVMType,'SVMPerf'))
                dataFile = fullfile(HOMELABELSETS{labelType},folder,[file '.mat']);
                load(dataFile);
                shortListInds = zeros(size(names));
                ind = unique(S(:));
                ind(ind<1) = [];
                shortListInds(ind)=true;
            elseif(strcmp(testParams.SVMType,'SVMTop10'))
                [~, ind] = sort(trainCounts{labelType},'descend');
                shortListInds = zeros(size(trainCounts{labelType}));
                shortListInds(ind(1:min(end,10)))=true;
            elseif(strcmp(testParams.SVMType,'SVMLPBP'))
                svmstr = [testParams.SVMType num2str(testParams.svmLPBPItt)];
                [shortListInds] = svmShortListLPBPSoftMax(globalSVM{labelType}(testParams.svmLPBPItt),SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs);
            elseif(length(globalSVM{labelType}) ==2)
                svmOutput = [];
                [shortListInds svmOutput.svm]  = svmShortList(globalSVM{labelType}{1},SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,0);
                [shortListInds fs] = svmShortList(globalSVM{labelType}{2},svmOutput,{'svm'},0);
            else
                shortListInds = svmShortList(globalSVM{labelType},SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,0);
            end
        elseif(isfield(testParams,'SVMOutput'))
            shortListInds = testParams.SVMOutput{labelType}(i,:)>0;
            minSL = 1;
            if(sum(shortListInds)<minSL)
                [a inds] = sort(testParams.SVMOutput{labelType}(i,:),'descend');
                shortListInds(inds(1:minSL)) = true;
            end
            svmstr = [testParams.SVMType 'Min' num2str(minSL)];
            Labels{labelType}(shortListInds)
        else
            shortListInds = ones(size(Labels{labelType}));
        end
        shortListMask{labelType} = shortListInds==1;
    end
    
    if(isfield(testParams,'RetrievalMetaMatch'))
        if(strcmp('GroundTruth',testParams.RetrievalMetaMatch))
            svmstr = [svmstr 'RetGT'];
            metaFields = fieldnames(testParams.TrainMetadata);
            totalMask = ones(size(retInds))==1;
            for f = 1:length(metaFields)
                mask = strcmp(testParams.TrainMetadata.(metaFields{f})(retInds),testParams.TestMetadata.(metaFields{f}){i});
                totalMask = mask&totalMask;
            end
            retInds = retInds(totalMask);
        end
        if(strcmp('SVM',testParams.RetrievalMetaMatch))
            svmstr = [svmstr 'RetSVM'];
            metaFields = fieldnames(testParams.TrainMetadata);
            totalMask = ones(size(retInds))==1;
            for f = 1:length(metaFields)
                mask = strcmp(testParams.TrainMetadata.(metaFields{f})(retInds),testParams.TestMetadataSVM.(metaFields{f}){i});
                totalMask = mask&totalMask;
            end
            retInds = retInds(totalMask);
        end
    end
    
    FGSet = 0;Kndx = 1;
    classifierStr = repmat('0',[1 length(HOMELABELSETS)]);
    preStr = '';
    probSuffix = sprintf('K%d',testParams.K(Kndx));
    clear imSP testImSPDesc;
    [testImSPDesc imSP] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K,testParams.segSuffix);
    probPerLabel = cell(size(HOMELABELSETS));
    dataCost = cell(size(probPerLabel));
    for labelType=1:length(HOMELABELSETS)
        [foo labelSet] = fileparts(HOMELABELSETS{labelType});
        if(strcmp(labelSet,'LabelsForgroundBK'))
            FGSet = labelType;
        end
        if(isempty(classifiers{labelType}))
            if(testParams.retSetSize == length(trainFileList) )
                retSetIndex = trainIndex{labelType,Kndx};
            else
                [retSetIndex descMask] = PruneIndex(trainIndex{labelType,Kndx},retInds,testParams.retSetSize,testParams.minSPinRetSet);
            end
            TrainAndRunDetector(HOMEDATA,testFileList(i),trainFileList(unique(retSetIndex.image)),testParams);
            suffix = sprintf('R%dK%dTNN%d',testParams.retSetSize,testParams.K(Kndx),testParams.targetNN);%nn%d  ,testParams.targetNN
            suffix = [suffix testParams.globalDescSuffix];
            suffix = [suffix glSuffix];
            probSuffix = [suffix '-sc' myN2S(testParams.smoothingConst) probType];
            labelNums = 1:length(trainCounts{labelType});
            probPerLabel{labelType} = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,probSuffix,retSetIndex,[],labelNums,trainCounts{labelType,Kndx},probType,testParams.smoothingConst,1); %#ok<AGROW>
            if(~isempty(probPerLabel{labelType}) && size(probPerLabel{labelType},1)~=size(testImSPDesc.sift_hist_dial,1))
                suffix = [suffix 'added'];
                probPerLabel{labelType} = [];
            end
            if(isempty(probPerLabel{labelType}))
                rawNNs = DoRNNSearch(testImSPDesc,[],fullfile(DataDir,labelSet),baseFName,suffix,testParams,Kndx);
                if(isempty(rawNNs))
                    if(testParams.retSetSize >= length(trainFileList) )
                        if(isempty(fullSPDesc{labelType,Kndx}))
                            fullSPDesc{labelType,Kndx} = LoadSegmentDesc(trainFileList,retSetIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(Kndx),testParams.segSuffix);
                        end
                        retSetSPDesc = fullSPDesc{labelType,Kndx};
                    else
                        if(isempty(fullSPDesc{labelType,Kndx}))
                            retSetSPDesc = LoadSegmentDesc(trainFileList,retSetIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(Kndx),testParams.segSuffix);
                        else
                            retSetSPDesc = [];
                            for dNdx = 1:length(testParams.segmentDescriptors)
                                retSetSPDesc.(testParams.segmentDescriptors{dNdx}) = fullSPDesc{labelType,Kndx}.(testParams.segmentDescriptors{dNdx})(descMask,:);
                            end
                        end
                    end
                    %timing(i,2)=toc;  rawNNs = DoRNNSearch(testImSPDesc,retSetSPDesc,fullfile(DataDir,labelSet),baseFName,suffix,testParams,Kndx);  timing(i,2)=toc-timing(i,2);
                    [rawNNs timing(i,2) totalMatches] = DoRNNSearch(testImSPDesc,retSetSPDesc,fullfile(DataDir,labelSet),baseFName,suffix,testParams,Kndx);
                    fprintf('%03d: %d\n',i',totalMatches);
                end
                [probPerLabel{labelType} timing(i,3)] = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,probSuffix,retSetIndex,rawNNs,labelNums,trainCounts{labelType,Kndx},probType,testParams.smoothingConst,1);
            end
            
            %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
            probPerLabel{labelType}(:,~shortListMask{labelType}) = min(probPerLabel{labelType}(:))-1;
            dataCost{labelType} = testParams.maxPenality*(1-1./(1+exp(-(testParams.BConst(2)*probPerLabel{labelType}+testParams.BConst(1)))));
            dataCost{labelType}(:,~shortListMask{labelType}) = testParams.maxPenality;
            %for k = 1:size(probPerLabel{labelType},2)
            %    temp = mnrval(testParams.BConst,probPerLabel{labelType}(:,k));
            %    dataCost{labelType}(:,k) = testParams.maxPenality*(1-temp(:,1));
            %end
        else
            probCacheFile = fullfile(DataDir,'ClassifierOutput',[labelSet testParams.CLSuffix],[baseFName  '.mat']);
            classifierStr(labelType) = '1';
            if(~exist(probCacheFile,'file'))
                features = GetFeaturesForClassifier(testImSPDesc);
                prob = test_boosted_dt_mc(classifiers{labelType}, features);
                make_dir(probCacheFile);save(probCacheFile,'prob');
            else
                clear prob;
                load(probCacheFile);
            end
            probPerLabel{labelType} = prob;
            if(size(prob,2) == 1 && length(Labels{labelType}) ==2)
                probPerLabel{labelType}(:,2) = -prob;
            end
            if(labelType == FGSet)
                probPerLabel{labelType}(:,1) = probPerLabel{labelType}(:,1);
                probPerLabel{labelType}(:,2) = probPerLabel{labelType}(:,2);
            end
            
            %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
            probPerLabel{labelType}(:,~shortListMask{labelType}) = min(probPerLabel{labelType}(:))-1;
            dataCost{labelType} = testParams.maxPenality*(1-1./(1+exp(-(testParams.BConstCla(2)*probPerLabel{labelType}+testParams.BConstCla(1)))));
            dataCost{labelType}(:,~shortListMask{labelType}) = testParams.maxPenality;
            %for k = 1:size(probPerLabel{labelType},2)
            %    temp = mnrval(testParams.BConstCla,probPerLabel{labelType}(:,k));
            %    dataCost{labelType}(:,k) = testParams.maxPenality*(1-temp(:,1));
            %end
        end
    end
    
    dataCostPix = cell(size(probPerLabel));
    if(testParams.PixelMRF)
        spTransform = imSP;%SPtoSkel(imSP);%
        spTransform(spTransform==0) = size(dataCost{1},1)+1;
        for labelType=1:length(HOMELABELSETS)
            temp = dataCost{labelType};
            temp(end+1,:) = 0;
            dataCostPix{labelType} = temp(spTransform,:);
        end
    end
    
    adjFile = fullfile(HOMETESTDATA,'Descriptors',sprintf('SP_Desc_k%d%s',testParams.K,testParams.segSuffix),'sp_adjacency',[baseFName '.mat']);
    load(adjFile);
    useLabelSets = 1:length(HOMELABELSETS);
    
    meanSPColors = zeros(size(probPerLabel{1},1),ch);
    imFlat = reshape(im,[ro*co ch]);
    for spNdx = 1:size(meanSPColors,1)
        meanSPColors(spNdx,:) = mean(imFlat(imSP(:)==spNdx,:),1);
    end
    
    endStr = [];
    preStr = [preStr probSuffix glSuffix testParams.CLSuffix];
    for labelType=1:length(HOMELABELSETS)
        if(testParams.weightBySize)
            [foo spSize] = UniqueAndCounts(imSP);
            dataCost{labelType} = dataCost{labelType}.*repmat(spSize(:),[1 size(dataCost{labelType},2)])./mean(spSize);
        end
        dataCost{labelType} = int32(dataCost{labelType});
        dataCostPix{labelType} = int32(dataCostPix{labelType});
    end
    %run mrf
    %{-
    for labelSmoothingInd = 1:length(testParams.LabelSmoothing)
        if(iscell(testParams.LabelSmoothing))
            labelSmoothing = testParams.LabelSmoothing{labelSmoothingInd};
            lsStr = myN2S(labelSmoothingInd,3);
        else
            labelSmoothing = repmat(testParams.LabelSmoothing(labelSmoothingInd),[length(dataCost) 3]);
            lsStr = myN2S(labelSmoothing(1),3);
        end
        for interLabelSmoothing  = testParams.InterLabelSmoothing
            interLabelSmoothingMat = repmat(interLabelSmoothing,size(labelPenality,1));
            for lPenNdx = 1:length(testParams.LabelPenality)
                if((sum(sum(labelSmoothing==0))==numel(labelSmoothing))&&lPenNdx>1); continue; end
                for ilPenNdx = 1:length(testParams.InterLabelPenality)
                    if(all(interLabelSmoothingMat(:)==0)&&ilPenNdx>1); continue; end
                    ilsStr = myN2S(interLabelSmoothing,3);
                    
                    etNdx = 1; epNdx = 1;
                    if(~testParams.PixelMRF)
                        endStr = sprintf('Seg WbS%s', num2str(testParams.weightBySize));
                    else
                        endStr = sprintf('Pix E%s EP%d Cn%d',testParams.edgeType{etNdx},testParams.edgeParam(epNdx),testParams.connected);
                    end
                    testName = sprintf('%s C%s %s S%s IS%s P%s IP%s %s%s',preStr,classifierStr,testParams.NormType,lsStr,ilsStr,...
                        testParams.LabelPenality{lPenNdx}(1:3),testParams.InterLabelPenality{ilPenNdx}(1:3),svmstr,endStr);
                                                
                    %testName = ['Pixel ' testName];
                    interLabelSmoothingMattemp = interLabelSmoothingMat;
                    if(FGSet>0 && interLabelSmoothingMat(FGSet,FGSet) < 1 && length(HOMELABELSETS)>1)
                        interLabelSmoothingMattemp(FGSet,:) = 1;
                        interLabelSmoothingMattemp(:,FGSet) = 1;
                    end

                    smoothingMatrix = BuildSmoothingMatrix(labelPenality,labelSmoothing(:,1),interLabelSmoothingMattemp,testParams.LabelPenality{lPenNdx},testParams.InterLabelPenality{ilPenNdx});
                    mrfParams = [];
                    mrfParams.labelSubSetsWeight = 0;
                    mrfParams.edgeType = testParams.edgeType{etNdx};
                    mrfParams.edgeParam = testParams.edgeParam(epNdx);
                    mrfParams.maxPenality = testParams.maxPenality;
                    mrfParams.connected = testParams.connected;
                    timing(i,4)=toc;
                    if(~testParams.PixelMRF)
                        [Ls Lsps] = MultiLevelSegMRF(fullfile(DataDir,testParams.MRFFold),HOMELABELSETS(useLabelSets),testName,baseFName,Labels,imSP,adjPairs,dataCost,smoothingMatrix,0,0);
                    else
                        [Ls Lsps] = MultiLevelPixMRF(fullfile(DataDir,testParams.MRFFold),HOMELABELSETS(useLabelSets),testName,baseFName,Labels,imSP,im,dataCostPix,smoothingMatrix,0,mrfParams);
                    end
                    timing(i,4)=toc-timing(i,4);
                    %for j=1:length(HOMELABELSETS);DrawImLabels(im,Ls{j},[rand([length(Labels{j}) 3]); [0 0 0]],Labels{j},[],1,0,j);end;show(im,7);
                end
            end
        end
    end
    %}
    ProgressBar(pfig,find(i==range),length(range));
end
close(pfig);

end


