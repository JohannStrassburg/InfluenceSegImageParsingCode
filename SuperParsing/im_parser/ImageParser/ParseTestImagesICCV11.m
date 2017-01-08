function timing = ParseTestImages(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex,trainCounts,labelPenality,Labels,classifiers,globalSVM,testParams,fullSPDesc)

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
    
    shortListMask = cell(size(Labels));
    svmstr = '';
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
            elseif(strcmp(testParams.SVMType,'SVMLPBP'))
                svmstr = [testParams.SVMType num2str(testParams.svmLPBPItt)];
                [shortListInds] = svmShortListLPBPSoftMax(globalSVM{labelType}(testParams.svmLPBPItt),SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs);
            elseif(length(globalSVM{labelType}) ==2)
                svmOutput = [];
                [shortListInds svmOutput.svm]  = svmShortList(globalSVM{labelType}{1},SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,testParams.SLThresh);
                [shortListInds fs] = svmShortList(globalSVM{labelType}{2},svmOutput,{'svm'},0);
            else
                shortListInds = svmShortList(globalSVM{labelType},SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,testParams.SLThresh);
            end
        else
            shortListInds = ones(size(Labels{labelType}));
        end
        shortListMask{labelType} = shortListInds==1;
    end
    
    FGSet = 0;Kndx = 1;
    classifierStr = repmat('0',[1 length(HOMELABELSETS)]);
    preStr = '';%num2str(testParams.K);
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
            suffix = sprintf('R%dK%d',testParams.retSetSize,testParams.K(Kndx));%nn%d  ,testParams.targetNN
            suffix = [suffix glSuffix];
            labelNums = 1:length(trainCounts{labelType});
            probPerLabel{labelType} = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,suffix,retSetIndex,[],labelNums,trainCounts{labelType,Kndx},probType,1); %#ok<AGROW>
            if(~isempty(probPerLabel{labelType}) && size(probPerLabel{labelType},1)~=size(testImSPDesc.gist_int,1))
                suffix = [suffix 'added'];
                probPerLabel{labelType} = [];
            end
            probPerLabel{labelType} = [];
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
                    timing(i,2)=toc;  rawNNs = DoRNNSearch(testImSPDesc,retSetSPDesc,fullfile(DataDir,labelSet),baseFName,suffix,testParams,Kndx);  timing(i,2)=toc-timing(i,2);
                end
                timing(i,3)=toc;  probPerLabel{labelType} = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,suffix,retSetIndex,rawNNs,labelNums,trainCounts{labelType,Kndx},probType,1);  timing(i,3)=toc-timing(i,3);
            end
            
            %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
            probPerLabel{labelType}(:,~shortListMask{labelType}) = min(probPerLabel{labelType}(:))-1;
            dataCost{labelType} = testParams.maxPenality*(1-1./(1+exp(-(testParams.BConst(2)*probPerLabel{labelType}+testParams.BConst(1)))));
            %for k = 1:size(probPerLabel{labelType},2)
            %    temp = mnrval(testParams.BConst,probPerLabel{labelType}(:,k));
            %    dataCost{labelType}(:,k) = testParams.maxPenality*(1-temp(:,1));
            %end
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
                probPerLabel{labelType}(:,1) = probPerLabel{labelType}(:,1);
                probPerLabel{labelType}(:,2) = probPerLabel{labelType}(:,2);
            end
            
            %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
            probPerLabel{labelType}(:,~shortListMask{labelType}) = min(probPerLabel{labelType}(:))-1;
            dataCost{labelType} = testParams.maxPenality*(1-1./(1+exp(-(testParams.BConstCla(2)*probPerLabel{labelType}+testParams.BConstCla(1)))));
            %for k = 1:size(probPerLabel{labelType},2)
            %    temp = mnrval(testParams.BConstCla,probPerLabel{labelType}(:,k));
            %    dataCost{labelType}(:,k) = testParams.maxPenality*(1-temp(:,1));
            %end
        end
    end
    
    %{
    minProb = 0;maxProb = 0;
    for labelType=1:length(HOMELABELSETS)
        maxProb = max(maxProb,max(probPerLabel{labelType}(:)));
        minProb = min(minProb,min(probPerLabel{labelType}(:)));
    end
    for labelType=1:length(HOMELABELSETS)
        for k = 1:size(probPerLabel{labelType},2)
            dataCost{labelType}(:,k) = testParams.maxPenality*(probPerLabel{labelType}(:,k)-maxProb)/(minProb-maxProb);
        end
    end
    %}
    
    %Add Forground Background Classes
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
    
    %{
    make_dir(fullfile(DataDir,'MinDataPenality','ImageWeb',folder,'sdf'));
    make_dir(fullfile(DataDir,'MinDataPenality','Ims',folder,'sdf'));
    imFID = fopen(fullfile(DataDir,'MinDataPenality','ImageWeb',folder, [file '.htm']),'w');
    fprintf(imFID,'\n<table border="0">\n');
    fprintf(imFID,'\t<tr>\n');
    show(im,1);
    set(gcf,'PaperPositionMode','auto');
    hold off;
    outfile = fullfile(DataDir,'MinDataPenality','Ims',folder, [file '.jpg']);
    print(outfile,'-djpeg','-r96');
    fprintf(imFID,'<td><img height="%d" src="%s"> </td>',min(ro,400),[ '../../Ims/' folder '/' file '.jpg']);% width="400"
    for labelType=1:length(HOMELABELSETS)
        [foo labelSet] = fileparts(HOMELABELSETS{labelType});
        show(reshape(min(dataCostPix{labelType},[],2),size(imSP)),1);
        set(gcf,'PaperPositionMode','auto');
        hold off;
        outfile = fullfile(DataDir,'MinDataPenality','Ims',folder,[file '-' labelSet '.png']);
        print(outfile,'-dpng','-r96');
    	fprintf(imFID,'<td><img height="%d" src="%s"><br>%s  </td>',min(ro,400),[ '../../Ims/' folder '/' file '-' labelSet '.png'],labelSet);% width="400"
    end
    fprintf(imFID,'\t</tr>\n</table><br>');
    fclose(imFID);
    %}
    
    ModLabels = Labels;
    ModLabelPenality = labelPenality;
    BKPenality = 100;
    useLabelSets = 1:length(HOMELABELSETS);
    preStr = '';%num2str(testParams.targetNN);
    if(~testParams.PixelMRF)d
        endStr = sprintf('SL%sfx Seg',myN2S(testParams.SLThresh));
    else
        endStr = sprintf('Pix');
    end
    if(FGSet)
        [foo spFGL] = min(dataCost{FGSet},[],2);
        L = spFGL(imSP);
        FGLabeling = L==2;
        %load(fullfile(HOMEDATA,'ColorModelMasks',testParams.colorMaskFile,[baseFName '.mat']));
        %FGLabeling = mask;
        %FGLabeling = FGLabeling+1;
        for labelType=1:length(HOMELABELSETS)
            if(FGSet~=labelType)
                if(testParams.PixelMRF)
                    if(testParams.fgFixed)
                        dataCostPix{labelType}(FGLabeling==1,:) = testParams.maxPenality;
                        dataCostPix{labelType}(:,end+1) = (FGLabeling(:)-1)*testParams.maxPenality;
                    else
                        dataCostPix{labelType}(:,end+1) = dataCostPix{FGSet}(:,1);%testParams.maxPenality/2;%
                    end
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
                    if(testParams.fgFixed)
                        dataCostPix{labelType} = zeros([size(FGLabeling) 2]);
                        dataCostPix{labelType}(:,:,1) = (FGLabeling-1)*testParams.maxPenality;
                        dataCostPix{labelType}(:,:,2) = testParams.maxPenality-dataCostPix{labelType}(:,:,1);
                        dataCostPix{labelType} = reshape(dataCostPix{labelType}, [size(imSP,1)*size(imSP,2) size(dataCostPix{labelType},3)]);
                    end
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
        preStr = sprintf('BK%d ',BKPenality);
        endStr = [endStr sprintf(' FgFix%d',testParams.fgFixed)];
    end
    
    adjFile = fullfile(HOMETESTDATA,'Descriptors',sprintf('SP_Desc_k%d%s',testParams.K,testParams.segSuffix),'sp_adjacency',[baseFName '.mat']);
    load(adjFile);
    
    
    meanSPColors = zeros(size(probPerLabel{1},1),ch);
    imFlat = reshape(im,[ro*co ch]);
    for spNdx = 1:size(meanSPColors,1)
        meanSPColors(spNdx,:) = mean(imFlat(imSP(:)==spNdx,:),1);
    end
    
    endStr = [endStr ' WbS' num2str(testParams.weightBySize)];
    preStr = [preStr 'R' num2str(testParams.retSetSize) glSuffix];
    for labelType=1:length(HOMELABELSETS)
        if(testParams.weightBySize)
            [foo spSize] = UniqueAndCounts(imSP);
            dataCost{labelType} = dataCost{labelType}.*repmat(spSize(:),[1 size(dataCost{labelType},2)])./mean(spSize);
        end
        dataCost{labelType} = int32(dataCost{labelType});
        %dataCostPix{labelType} = int32(dataCostPix{labelType});
    end
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
                    ilsStr = myN2S(interLabelSmoothing,3);
                    %I'm checking this in but this needs to be fixed. There is too much color model dependance going on here
                    for cmNdx = 1:length(testParams.colorModel)
                        colorModel = zeros(size(HOMELABELSETS));
                        if(FGSet>0);
                            colorModel(FGSet) = testParams.colorModel(cmNdx);
                            cmStr = myN2S(testParams.colorModel(cmNdx));
                        else
                            cmStr = '';
                            if(isfield(testParams,'colorMaskFile'))cmStr = testParams.colorMaskFile; end
                        end
                        for etNdx = 1:length(testParams.edgeType)
                            for epNdx = 1:length(testParams.edgeParam)
                                if(~strcmp(testParams.edgeType{etNdx},'norm') && epNdx > 1)
                                    continue;
                                end
                                for sdNdx = 1:length(testParams.smoothData)
                                    for cltNdx = 1:length(testParams.clType)
                                        for ncNdx = 1:length(testParams.numClusters)
                                            for fgdNdx = 1:length(testParams.fgDataWeight)
                                                testName = sprintf('%s C%s %s CM%s %s %s NC%s S%s IS%s P%s IP%s E%s EP%d SD%d %s%s FW%s Cn%d',preStr,classifierStr,testParams.NormType,cmStr,testParams.colorSpace,testParams.clType{cltNdx}, myN2S(testParams.numClusters(ncNdx),2),lsStr,ilsStr,...
                                                    testParams.LabelPenality{lPenNdx}(1:3),testParams.InterLabelPenality{ilPenNdx}(1:3),testParams.edgeType{etNdx},testParams.edgeParam(epNdx),testParams.smoothData(sdNdx),svmstr,endStr, myN2S(testParams.fgDataWeight(fgdNdx),3),testParams.connected);
                                                
                                                %testName = ['Pixel ' testName];
                                                interLabelSmoothingMattemp = interLabelSmoothingMat;
                                                if(FGSet>0 && interLabelSmoothingMat(FGSet,FGSet) < 1 && length(HOMELABELSETS)>1)
                                                    interLabelSmoothingMattemp(FGSet,:) = 1;
                                                    interLabelSmoothingMattemp(:,FGSet) = 1;
                                                end
                                                
                                                %show(reshape(dataCostPix{2}(:,1),size(imSP)),6)
                                                %show(reshape(dataCostPix{2}(:,3),size(imSP)),7)
                                                %show(reshape(dataCostPix{1}(:,1),size(imSP)),8)
                                                %show(reshape(dataCostPix{1}(:,2),size(imSP)),9)
                                                
                                                sz = testParams.smoothData(sdNdx);
                                                if(sz>0)
                                                    g = fspecial('gauss', [sz sz], sqrt(sz));
                                                end
                                                dataCostPixTemp = dataCostPix;
                                                for labelType=1:length(HOMELABELSETS)
                                                    if(sz>0)
                                                        for l = 1:size(dataCostPixTemp{labelType},2)
                                                            temp = dataCostPixTemp{labelType}(:,l);
                                                            temp = reshape(temp,size(imSP));
                                                            temp = imfilter(temp,g,'symmetric');%if(l==1);show(temp,20);end
                                                            temp = reshape(temp,[numel(imSP) 1]);
                                                            dataCostPixTemp{labelType}(:,l) = temp;
                                                        end
                                                    end
                                                    dataCostPixTemp{labelType} = int32(dataCostPixTemp{labelType});
                                                end
                                                
                                                smoothingMatrix = BuildSmoothingMatrix(ModLabelPenality,labelSmoothing(:,1),interLabelSmoothingMattemp,testParams.LabelPenality{lPenNdx},testParams.InterLabelPenality{ilPenNdx});
                                                mrfParams = [];
                                                mrfParams.colorModel = colorModel;
                                                mrfParams.labelSubSetsWeight = 0;
                                                mrfParams.edgeType = testParams.edgeType{etNdx};
                                                mrfParams.edgeParam = testParams.edgeParam(epNdx);
                                                mrfParams.colorSpace = testParams.colorSpace;
                                                mrfParams.clType = testParams.clType{cltNdx};
                                                mrfParams.numClusters = testParams.numClusters(ncNdx);
                                                mrfParams.maxPenality = testParams.maxPenality;
                                                mrfParams.connected = testParams.connected;
                                                mrfParams.fgDataWeight = testParams.fgDataWeight(fgdNdx);
                                                if(~testParams.PixelMRF)
                                                    [Ls Lsps] = MultiLevelSegMRF(fullfile(DataDir,testParams.MRFFold),HOMELABELSETS(useLabelSets),testName,baseFName,ModLabels,imSP,adjPairs,dataCost,smoothingMatrix,0,0,colorModel,meanSPColors);
                                                else
                                                    [Ls Lsps] = MultiLevelPixMRF(fullfile(DataDir,testParams.MRFFold),HOMELABELSETS(useLabelSets),testName,baseFName,ModLabels,imSP,im,dataCostPixTemp,smoothingMatrix,0,mrfParams);
                                                end
                                                %for j=1:length(HOMELABELSETS);DrawImLabels(im,Ls{j},[rand([length(ModLabels{j}) 3]); [0 0 0]],ModLabels{j},[],1,0,j);end;show(im,7);
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    %}
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

