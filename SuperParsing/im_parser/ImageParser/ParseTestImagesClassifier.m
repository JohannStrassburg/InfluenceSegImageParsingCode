function timing = ParseTestImagesClassifier(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testGlobalDesc,labelPenality,Labels,classifiers,globalSVM,testParams)

DataDir = fullfile(HOMETESTDATA,testParams.TestString);
close all;
pfig = ProgressBar('Parsing Images');
range = 1:length(testFileList);
timing = zeros(length(testFileList),4);
for i = range
    im = imread(fullfile(HOMEDATA,'..','Images',testFileList{i}));
    [folder file] = fileparts(testFileList{i});
    baseFName = fullfile(folder,file);
    
    shortListMask = cell(size(Labels));
    svmstr = '';
    for labelType=1:length(HOMELABELSETS)
        if(~isempty(globalSVM{labelType}))
            svmstr = 'SVMSLRP1.5 ';
            shortListInds = svmShortList(globalSVM{labelType},SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,testParams.SLThresh);
        else
            shortListInds = ones(size(Labels{labelType}));
        end
        shortListMask{labelType} = shortListInds==1;
    end
    
    FGSet = 0;
    classifierStr = repmat('0',[1 length(HOMELABELSETS)]);
    Kstr = num2str(testParams.K);
    clear imSP testImSPDesc;
    [testImSPDesc imSP] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K);
    probPerLabel = cell(size(HOMELABELSETS));
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
                probPerLabel{labelType} = probPerLabel{labelType} + testParams.FGShift;
            end
            probPerLabel{labelType}(:,~shortListMask{labelType}) = min(probPerLabel{labelType}(:))-1;
        end
    end
    %Add Forground Background Classes

    %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
    dataCost = cell(size(probPerLabel));
    dataCostW = cell(size(probPerLabel));
    for j = 1:numel(probPerLabel)
        if(strcmp(testParams.NormType,'Bmulti'))
            temp = mnrval(testParams.Bmulti{j},probPerLabel{j});
            dataCost{j} = int32(testParams.maxPenality*(1-temp));
        elseif(strcmp(testParams.NormType,'B'))
            for k = 1:size(probPerLabel{j},2)
                temp = mnrval(testParams.B{j}(:,k),probPerLabel{j}(:,k));
                dataCost{j}(:,k) = int32(testParams.maxPenality*(1-temp(:,1)));
            end
        else
            for k = 1:size(probPerLabel{j},2)
                temp = mnrval(testParams.BConst,probPerLabel{j}(:,k));
                dataCost{j}(:,k) = int32(testParams.maxPenality*(1-temp(:,1)));
            end
        end
    end
    ModLabels = Labels;
    ModLabelPenality = labelPenality;
    BKPenality = 100;
    endStr = '';
    if(FGSet)
        [foo spL] = max(probPerLabel{FGSet},[],2);
        L = spL(imSP);
        inputMask = L==2;
        [outputMaskFine outputMaskCorse] = gcmask(DataDir,baseFName,im,inputMask,testParams.GrabCutl1,testParams.GrabCutl2);
        outputMaskFine = outputMaskFine+1;outputMaskCorse = outputMaskCorse+1;
        if(testParams.GrabCutFine)
            FGLabeling = outputMaskFine;
        else
            FGLabeling = outputMaskCorse;
        end
        for labelType=1:length(HOMELABELSETS)
            if(FGSet~=labelType)
                dataCost{labelType} = [dataCost{labelType} repmat(max(min(dataCost{labelType},[],2))+1,[size(dataCost{labelType},1) 1])];
                ModLabels{labelType} = [Labels{labelType}(:)' {'Unlabeled'}];
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
        Kstr = sprintf('%s FS%.2f BK%d',Kstr,testParams.FGShift,BKPenality);
    end
    endStr = sprintf(' SL%sfx',myN2S(testParams.SLThresh));
    
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
            if(FGSet>0 && interLabelSmoothingMat(FGSet,FGSet) < 1)
                interLabelSmoothingMat(FGSet,:) = 1;
                interLabelSmoothingMat(:,FGSet) = 1;
            end
            for lPenNdx = 1:length(testParams.LabelPenality)
                if((sum(sum(labelSmoothing==0))==numel(labelSmoothing))&&lPenNdx>1); continue; end
                for ilPenNdx = 1:length(testParams.InterLabelPenality)
                    if(all(interLabelSmoothingMat(:)==0)&&ilPenNdx>1); continue; end
                    ilsStr = myN2S(interLabelSmoothing);
                    testName = sprintf('K%s %s S%s IS%s P%s IP%s %s%s',Kstr,testParams.NormType,lsStr,ilsStr,...
                        testParams.LabelPenality{lPenNdx}(1:3),testParams.InterLabelPenality{ilPenNdx}(1:3),svmstr,endStr);
                    smoothingMatrix = BuildSmoothingMatrix(ModLabelPenality,labelSmoothing(:,1),interLabelSmoothingMat,testParams.LabelPenality{lPenNdx},testParams.InterLabelPenality{ilPenNdx});
                    tic
                    [L Lsp] = MultiLevelSegMRF(DataDir,HOMELABELSETS,testName,baseFName,ModLabels,imSP,adjPairs,dataCostW,smoothingMatrix);
                    timing(i,4) = toc;
                    %for j=1:length(HOMELABELSETS); show(L{j},j);end
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

