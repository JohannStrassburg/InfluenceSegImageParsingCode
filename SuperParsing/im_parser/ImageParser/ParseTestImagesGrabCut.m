function timing = ParseTestImagesGrabCut(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testGlobalDesc,labelPenality,Labels,classifiers,globalSVM,testParams)

DataDir = fullfile(HOMETESTDATA,testParams.TestString);
close all;
pfig = ProgressBar('Parsing Images');
range = 1:length(testFileList);
if(isfield(testParams,'range'))
    range = testParams.range;
end
timing = zeros(length(testFileList),4);
for i = range
    im = imread(fullfile(HOMEDATA,'..','Images',testFileList{i}));
    [folder file] = fileparts(testFileList{i});
    baseFName = fullfile(folder,file);
    
    timing(i,1)=toc;  [retInds rank]= FindRetrievalSet(trainGlobalDesc,SelectDesc(testGlobalDesc,i,1),DataDir,baseFName,testParams);  timing(i,1)=toc-timing(i,1);
    
    shortListMask = cell(size(Labels));
    svmstr = '';
    for labelType=1:length(HOMELABELSETS)
        if(~isempty(globalSVM{labelType}))
            svmstr = testParams.SVMType;
            shortListInds = svmShortList(globalSVM{labelType},SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,testParams.SLThresh);
        else
            shortListInds = ones(size(Labels{labelType}));
        end
        shortListMask{labelType} = shortListInds==1;
    end
    
    FGSet = 0;
    classifierStr = repmat('0',[1 length(HOMELABELSETS)]);
    preStr = '';%num2str(testParams.K);
    clear imSP testImSPDesc;
    [testImSPDesc imSP] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K);
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
            suffix = sprintf('R%dK%d',testParams.retSetSize,testParams.K(Kndx));
            labelNums = 1:length(trainCounts{labelType});
            probPerLabel{labelType,Kndx} = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,suffix,retSetIndex,[],labelNums,trainCounts{labelType,Kndx},probType,1); %#ok<AGROW>
            if(isempty(probPerLabel{labelType,Kndx}))
                rawNNs = DoRNNSearch(testImSPDesc,[],fullfile(DataDir,labelSet),baseFName,suffix,testParams,Kndx);
                if(isempty(rawNNs))
                    if(testParams.retSetSize == length(trainFileList) )
                        if(isempty(fullSPDesc{labelType,Kndx}))
                            fullSPDesc{labelType,Kndx} = LoadSegmentDesc(trainFileList,retSetIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(Kndx));
                        end
                        retSetSPDesc = fullSPDesc{labelType,Kndx};
                    else
                        if(isempty(fullSPDesc{labelType,Kndx}))
                            retSetSPDesc = LoadSegmentDesc(trainFileList,retSetIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(Kndx));
                        else
                            retSetSPDesc = [];
                            for dNdx = 1:length(testParams.segmentDescriptors)
                                retSetSPDesc.(testParams.segmentDescriptors{dNdx}) = fullSPDesc{labelType,Kndx}.(testParams.segmentDescriptors{dNdx})(descMask,:);
                            end
                        end
                    end
                    timing(i,2)=toc;  rawNNs = DoRNNSearch(testImSPDesc,retSetSPDesc,fullfile(DataDir,labelSet),baseFName,suffix,testParams,Kndx);  timing(i,2)=toc-timing(i,2);
                end
                timing(i,3)=toc;  probPerLabel{labelType,Kndx} = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,suffix,retSetIndex,rawNNs,labelNums,trainCounts{labelType,Kndx},probType,1);  timing(i,3)=toc-timing(i,3);
            end

            %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
            probPerLabel{labelType,Kndx}(:,~shortListMask{labelType}) = min(probPerLabel{labelType,Kndx}(:))-1;
            for k = 1:size(probPerLabel{labelType,Kndx},2)
                temp = mnrval(testParams.BConst,probPerLabel{labelType,Kndx}(:,k));
                dataCost{labelType,Kndx}(:,k) = int32(testParams.maxPenality*(1-temp(:,1)));
            end
        else
            probCacheFile = fullfile(DataDir,'ClassifierOutput',labelSet,[baseFName '.mat']);
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
                probPerLabel{labelType}(:,1) = probPerLabel{labelType}(:,1) - testParams.FGShift/2;
                probPerLabel{labelType}(:,2) = probPerLabel{labelType}(:,2) + testParams.FGShift/2;
            end
            
            %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
            probPerLabel{labelType}(:,~shortListMask{labelType}) = min(probPerLabel{labelType}(:))-1;
            for k = 1:size(probPerLabel{labelType},2)
                temp = mnrval(testParams.BConstCla,probPerLabel{labelType}(:,k));
                dataCost{labelType}(:,k) = int32(testParams.maxPenality*(1-temp(:,1)));
            end
        end
    end
    
    
    %Add Forground Background Classes
    dataCostPix = cell(size(probPerLabel));
    ModLabels = Labels;
    ModLabelPenality = labelPenality;
    BKPenality = 100;
    useLabelSets = 1:length(HOMELABELSETS);
    preStr = '';
    if(~testParams.PixelMRF)
        endStr = sprintf('SL%sfx Seg',myN2S(testParams.SLThresh)); 
    else
        endStr = sprintf('SL%sfx Pix l%s-%s GCF%d',myN2S(testParams.SLThresh),myN2S(testParams.GrabCutl1),myN2S(testParams.GrabCutl2),testParams.GrabCutFine);
    end
    if(FGSet)
        [foo spFGL] = min(dataCost{FGSet},[],2);
        L = spFGL(imSP);
        inputMask = L==2;
        [outputMaskFine outputMaskCorse] = gcmask(HOMETESTDATA,baseFName,im,inputMask,testParams.GrabCutl1,testParams.GrabCutl2,testParams.FGShift);
        outputMaskFine = outputMaskFine+1;outputMaskCorse = outputMaskCorse+1;
        if(testParams.GrabCutFine)
            FGLabeling = outputMaskFine;
        else
            FGLabeling = outputMaskCorse;
        end
        for labelType=1:length(HOMELABELSETS)
            if(FGSet~=labelType)
                if(testParams.PixelMRF)
                    dataCostPix{labelType} = zeros([size(FGLabeling) size(dataCost{labelType},2)+1]);
                    for l = 1:size(dataCost{labelType},2)
                        dcp = dataCost{labelType}(imSP(:),l);
                        if(testParams.fgFixed)
                            dcp(FGLabeling==1) = testParams.maxPenality;
                            dataCost{labelType}(spFGL==1,l) = testParams.maxPenality;
                        end
                        dataCostPix{labelType}(:,:,l) = reshape(dcp,size(FGLabeling));
                    end
                    if(testParams.fgFixed)
                        dataCostPix{labelType}(:,:,end) = (FGLabeling-1)*testParams.maxPenality;
                    else
                        dataCostPix{labelType}(:,:,end) = (FGLabeling-1)*testParams.maxPenality;
                    end
                    dataCostPix{labelType} = reshape(dataCostPix{labelType}, [size(imSP,1)*size(imSP,2) size(dataCostPix{labelType},3)]);
                else
                    if(testParams.fgFixed)
                        for l = 1:size(dataCost{labelType},2)
                            dataCost{labelType}(spFGL==1,l) = testParams.maxPenality;
                        end
                        dataCost{labelType}(:,end+1) = (spFGL-1)*testParams.maxPenality;
                    else
                        dataCost{labelType}(:,end+1) = dataCost{FGSet}(:,1);
                    end
                end
                ModLabels{labelType} = [Labels{labelType}(:)' {'Unlabeled'}];
            else
                if(testParams.PixelMRF)
                    dataCostPix{labelType} = zeros([size(FGLabeling) 2]);
                    dataCostPix{labelType}(:,:,1) = (FGLabeling-1)*testParams.maxPenality;
                    dataCostPix{labelType}(:,:,2) = testParams.maxPenality-dataCostPix{labelType}(:,:,1);
                    dataCostPix{labelType} = reshape(dataCostPix{labelType}, [size(imSP,1)*size(imSP,2) size(dataCostPix{labelType},3)]);
                end
            end
        end
        if(isfield(testParams,'BKPenality'))
            BKPenality = testParams.BKPenality;
        end
        for lt1 = 1:length(HOMELABELSETS)
            for lt2 = 1:length(HOMELABELSETS)
                if(lt1~=FGSet && lt2~=FGSet)
                	ModLabelPenality{lt1,lt2} = [labelPenality{lt1,lt2} ones(size(labelPenality{lt1,lt2},1),1);ones(1,size(labelPenality{lt1,lt2},2)) 0];
                end
                if(lt1==FGSet&&lt2~=FGSet)
                    ModLabelPenality{lt1,lt2} = BKPenality*[ones(1,size(labelPenality{lt1,lt2},2)) 0;zeros(1,size(labelPenality{lt1,lt2},2)) 1];
                end
                if(lt1~=FGSet&&lt2==FGSet)
                    ModLabelPenality{lt1,lt2} = BKPenality*[ones(1,size(labelPenality{lt1,lt2},1)) 0;zeros(1,size(labelPenality{lt1,lt2},1)) 1]';
                end
            end
        end
        if(testParams.ExcludeFB)
            dataCost(FGSet) = [];
            ModLabelPenality(FGSet,:) = [];
            ModLabelPenality(:,FGSet) = [];
            useLabelSets(FGSet) = [];
            ModLabels(FGSet) = [];
        end
        preStr = sprintf('FS%.2f BK%d ',testParams.FGShift,BKPenality);
        endStr = [endStr sprintf(' FgFix%d',testParams.fgFixed)];
    end
    
    adjFile = fullfile(HOMETESTDATA,'Descriptors',sprintf('SP_Desc_k%d',testParams.K),'sp_adjacency',[baseFName '.mat']);
    load(adjFile);

    %run mrf
    %{-
    for labelSmoothingInd = 1:length(testParams.LabelSmoothing)
        if(iscell(testParams.LabelSmoothing))
            labelSmoothing = testParams.LabelSmoothing{labelSmoothingInd};
            lsStr = myN2S(labelSmoothingInd);
        else
            labelSmoothing = repmat(testParams.LabelSmoothing(labelSmoothingInd),[length(dataCost) 3]);
            lsStr = myN2S(labelSmoothing(1));
        end
        for interLabelSmoothing  = testParams.InterLabelSmoothing
            interLabelSmoothingMat = repmat(interLabelSmoothing,size(ModLabelPenality,1));
            for lPenNdx = 1:length(testParams.LabelPenality)
                if((sum(sum(labelSmoothing==0))==numel(labelSmoothing))&&lPenNdx>1); continue; end
                for ilPenNdx = 1:length(testParams.InterLabelPenality)
                    if(all(interLabelSmoothingMat(:)==0)&&ilPenNdx>1); continue; end
                    ilsStr = myN2S(interLabelSmoothing);
                    testName = sprintf('%sC%s %s S%s IS%s P%s IP%s %s%s',preStr,classifierStr,testParams.NormType,lsStr,ilsStr,...
                        testParams.LabelPenality{lPenNdx}(1:3),testParams.InterLabelPenality{ilPenNdx}(1:3),svmstr,endStr);
                    smoothingMatrix = BuildSmoothingMatrix(ModLabelPenality,labelSmoothing(:,1),interLabelSmoothingMat,testParams.LabelPenality{lPenNdx},testParams.InterLabelPenality{ilPenNdx});
                    if(~testParams.PixelMRF)
                        [L Lsp] = MultiLevelSegMRF(DataDir,HOMELABELSETS(useLabelSets),testName,baseFName,ModLabels,imSP,adjPairs,dataCost,smoothingMatrix);
                    else
                        [L Lsp] = MultiLevelPixMRF(DataDir,HOMELABELSETS(useLabelSets),testName,baseFName,ModLabels,imSP,im,dataCostPix,smoothingMatrix);
                    end
                    %for j=1:length(HOMELABELSETS); show(L{j},j,0);end
                    
                    if(FGSet>0 && interLabelSmoothingMat(FGSet,FGSet) < 1)
                        interLabelSmoothingMat(FGSet,:) = 1;
                        interLabelSmoothingMat(:,FGSet) = 1;
                        testName = [testName ' FBForce'];
                        smoothingMatrix = BuildSmoothingMatrix(ModLabelPenality,labelSmoothing(:,1),interLabelSmoothingMat,testParams.LabelPenality{lPenNdx},testParams.InterLabelPenality{ilPenNdx});
                        if(~testParams.PixelMRF)
                            [L Lsp] = MultiLevelSegMRF(DataDir,HOMELABELSETS(useLabelSets),testName,baseFName,ModLabels,imSP,adjPairs,dataCost,smoothingMatrix);
                        else
                            [L Lsp] = MultiLevelPixMRF(DataDir,HOMELABELSETS(useLabelSets),testName,baseFName,ModLabels,imSP,im,dataCostPix,smoothingMatrix);
                        end
                        %for j=1:length(HOMELABELSETS); show(L{j},j+length(HOMELABELSETS),0);end
                    end
                end
            end
        end
    end
    ProgressBar(pfig,find(i==range),length(range));
end
close(pfig);

end

function [str] = myN2S(num,prec)
    if(~exist('prec','var'))
        prec = 2;
    end
    if(num<1)
        str = sprintf('%%.%df',prec);
        str = sprintf(str,num);
    else
        str = sprintf('%%0%dd',prec);
        str = sprintf(str,num);
    end
end

