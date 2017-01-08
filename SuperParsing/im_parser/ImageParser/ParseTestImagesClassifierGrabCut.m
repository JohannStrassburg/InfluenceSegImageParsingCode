function timing = ParseTestImagesClassifierGrabCut(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testGlobalDesc,labelPenality,Labels,classifiers,globalSVM,testParams)

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
        if(~isempty(classifiers{labelType}))
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
            probPerLabel{labelType}(:,~shortListMask{labelType}) = min(probPerLabel{labelType}(:))-1;
            
            %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
            for k = 1:size(probPerLabel{labelType},2)
                temp = mnrval(testParams.BConst,probPerLabel{labelType}(:,k));
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
            dataCost{FGSet} = int32(zeros(size(dataCost{FGSet})));
        end
    end
    preStr = sprintf('FS%.2f BK%d',testParams.FGShift,BKPenality);
    if(~testParams.PixelMRF)
        endStr = sprintf('SL%sfx Seg FgFix%d',myN2S(testParams.SLThresh),testParams.fgFixed);
    else
        endStr = sprintf('SL%sfx Pix l%s-%s GCF%d FBFix%d',myN2S(testParams.SLThresh),myN2S(testParams.GrabCutl1),myN2S(testParams.GrabCutl2),testParams.GrabCutFine,testParams.fgFixed);
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
                    testName = sprintf('%s %s S%s IS%s P%s IP%s %s%s',preStr,testParams.NormType,lsStr,ilsStr,...
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

