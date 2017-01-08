function ParseTestVideos(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testFileListLabeled,labelPenality,Labels,classifiers,testParams)

DataDir = fullfile(HOMETESTDATA,testParams.TestString);
HOMEDATADESC = fullfile(HOMEDATA,'Descriptors',sprintf('SP_Desc_k%d%s',testParams.K,testParams.segSuffix));
if(~isfield(testParams,'segSuffix'))
    testParams.segSuffix = '';
end
%close all;
%pfig = ProgressBar('Parsing Images');
range = 1:length(testFileList);
if(isfield(testParams,'range'))
    range = testParams.range;
end
glSuffix = '';
nsuffix = '';
csuffix = '';
testDirList = (FileList2DirList(testFileList(range)));
testDirListLabeled = (FileList2DirList(testFileListLabeled));
testDirs = unique(testDirList);
for d = 1:length(testDirs)
    testDir = testDirs{d};
    testFileNdx = find(strcmp(testDir,testDirList));
    
    %spProbs = testParams.spProbs;
    %spData = testParams.spData;
    %spSizes = testParams.spSizes;
    %{-
    spProbs = cell(length(HOMELABELSETS),1);
    spData = cell(length(HOMELABELSETS),1);
    spSizes = cell(0);
    allAdjPairs = [];
    
    pfig = ProgressBar('Parsing Images');
    for i = testFileNdx(:)'
        [folder base] = fileparts(testFileList{i});
        baseFName = fullfile(folder,base);
        classifierStr = repmat('0',[1 length(HOMELABELSETS)]);
        for ls = 1:length(classifiers); classifierStr(ls) = num2str(~isempty(classifiers{ls}));end
        outSPName = fullfile(HOMEDATADESC,'super_pixels',folder,[base '.mat']);
        load(outSPName); %superPixels;
        imSP = superPixels;
        spNdx = unique(imSP);
        if(length(spSizes)<max(spNdx))
            spSizes{max(spNdx)} = []; 
            for j = 1:length(spProbs); spProbs{j}{max(spNdx)} = []; end; 
            for j = 1:length(spData); spData{j}{max(spNdx)} = []; end;
        end
        for s = 1:length(spNdx); spSizes{spNdx(s)} = [spSizes{spNdx(s)}; sum(imSP(:)==spNdx(s))]; end
        
        outSPName = fullfile(HOMEDATADESC,'sp_adjacency',folder,[base '.mat']);
        load(outSPName); %adjPairs
        allAdjPairs = [allAdjPairs; adjPairs];
        
        probPerLabel = cell(size(HOMELABELSETS));
        dataCost = cell(size(probPerLabel));
        for ls=1:length(HOMELABELSETS)
            [~, labelSet] = fileparts(HOMELABELSETS{ls});
            if(isempty(classifiers{ls}))
                Kndx = 1;
                nsuffix = sprintf('R%dK%d%s',testParams.retSetSize,testParams.K(Kndx),glSuffix);%nn%d  ,testParams.targetNN
                outfilename = fullfile(DataDir,labelSet,['probPerLabel' nsuffix],folder,[base '.mat']);
                p = load(outfilename);
                probPerLabel{ls} = p.probPerLabel;
                %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
                dataCost{ls} = testParams.maxPenality*(1-(1./(1+exp(-(testParams.BConst(1)+probPerLabel{ls}.*testParams.BConst(2))))));
            else
                csuffix = testParams.clSuffix;
                probCacheFile = fullfile(DataDir,['ClassifierOutput' csuffix],labelSet,[baseFName '.mat']);
                load(probCacheFile);
                probPerLabel{ls} = prob;
                if(size(prob,2) == 1 && length(Labels{ls}) ==2)
                    probPerLabel{ls}(:,2) = -prob;
                end
                %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
                dataCost{ls} = testParams.maxPenality*(1-(1./(1+exp(-(testParams.BConstCla(1)+probPerLabel{ls}.*testParams.BConstCla(2))))));
            end
            for s = 1:length(spNdx)
                spProbs{ls}{spNdx(s)} = [spProbs{ls}{spNdx(s)}; probPerLabel{ls}(s,:)]; 
                spData{ls}{spNdx(s)} = [spData{ls}{spNdx(s)}; dataCost{ls}(s,:)]; 
            end
            suffix = [classifierStr '-' nsuffix csuffix];
            if(any(strcmp(testFileList{i},testFileListLabeled)))
                outfile = fullfile(DataDir,testParams.MRFFold,labelSet,['Base-' suffix],folder,[base '.mat']);make_dir(outfile);
                if(~exist(outfile,'file'))
                    [~, Lsp] = max(probPerLabel{ls},[],2);
                    LspMap = zeros(spNdx(end),1);
                    LspMap(spNdx) = Lsp;
                    L = LspMap(imSP);
                    labelList = Labels{ls};
                    save(outfile,'L','labelList','Lsp');
                end
            end
        end
        ProgressBar(pfig,find(i==range),length(range));
    end
    close(pfig);
    %}
    
    
    pfig = ProgressBar('Combining Tubes. Method:');
    testFileLabeledNdx = find(strcmp(testDir,testDirListLabeled));
    for c = 1:length(testParams.CombMethods)
        ProgressBar(pfig,c,length(testParams.CombMethods));
        probPerLabel = cell(size(HOMELABELSETS));
        dataCost = cell(size(HOMELABELSETS));
        dataCost2 = cell(size(HOMELABELSETS));
        
        endstr = '';
        for ls=1:length(HOMELABELSETS)
            [probPerLabel{ls} dataCost{ls} tubeSize] = feval(testParams.CombMethods{c},spProbs{ls},spData{ls},spSizes);
            if(testParams.postDataCost)
                endstr = [endstr ' pdc'];
                if(isempty(classifiers{ls}))
                    dataCost{ls} = testParams.maxPenality*(1-(1./(1+exp(-(testParams.BConst(1)+probPerLabel{ls}.*testParams.BConst(2))))));
                else
                    dataCost{ls} = testParams.maxPenality*(1-(1./(1+exp(-(testParams.BConstCla(1)+probPerLabel{ls}.*testParams.BConstCla(2))))));
                end
            end
        end
        
        usedSPMask = tubeSize>0;
        usedSPInd = find(usedSPMask);
        spInd2Used = zeros(size(tubeSize));
        spInd2Used(usedSPInd) = 1:length(usedSPInd);
        allAdjPairsUsed = spInd2Used(allAdjPairs);
        for ls=1:length(HOMELABELSETS)
            dataCost{ls} = dataCost{ls}(usedSPMask,:);
            if(testParams.weightBySize)
                dataCost{ls} = dataCost{ls}.*repmat(tubeSize(usedSPMask),[1 size(dataCost{ls},2)])./mean(tubeSize(usedSPMask));
                endstr = [endstr ' wbs'];
            end
            dataCost{ls} = int32(dataCost{ls});
            probPerLabel{ls} = probPerLabel{ls}(usedSPMask,:);
        end
        
        for labelSmoothingInd = 1:length(testParams.LabelSmoothing)
            labelSmoothing = repmat(testParams.LabelSmoothing(labelSmoothingInd),length(dataCost));
            lsStr = myN2S(labelSmoothing(1),3);
            for interLabelSmoothing  = testParams.InterLabelSmoothing
                interLabelSmoothingMat = repmat(interLabelSmoothing,size(labelPenality,1));
                for lPenNdx = 1:length(testParams.LabelPenality)
                    if((sum(sum(labelSmoothing==0))==numel(labelSmoothing))&&lPenNdx>1); continue; end
                    for ilPenNdx = 1:length(testParams.InterLabelPenality)
                        if(all(interLabelSmoothingMat(:)==0)&&ilPenNdx>1); continue; end
                        ilsStr = myN2S(interLabelSmoothing,3);

                        testName = sprintf('%s S%s IS%s P%s IP%s%s',suffix,lsStr,ilsStr,testParams.LabelPenality{lPenNdx}(1:3),testParams.InterLabelPenality{ilPenNdx}(1:3),endstr);

                        smoothingMatrix = BuildSmoothingMatrix(labelPenality,labelSmoothing,interLabelSmoothingMat,testParams.LabelPenality{lPenNdx},testParams.InterLabelPenality{ilPenNdx});
                        [Ls Lsps] = MultiLevelSegMRF(fullfile(DataDir,testParams.MRFFold),HOMELABELSETS,testName,[],Labels,[],allAdjPairsUsed,dataCost,smoothingMatrix,0,0);
                        
                        for i = testFileLabeledNdx(:)'
                            [folder base] = fileparts(testFileListLabeled{i});
                            outSPName = fullfile(HOMEDATADESC,'super_pixels',folder,[base '.mat']);
                            load(outSPName);
                            imSP = spInd2Used(superPixels);
                            spNdx = unique(imSP);
                            for ls=1:length(HOMELABELSETS)
                                [~, labelSet] = fileparts(HOMELABELSETS{ls});
                                outfile = fullfile(DataDir,testParams.MRFFold,labelSet,[testParams.CombMethods{c} '-' testName],folder,[base '.mat']);make_dir(outfile);
                                Lsp = Lsps{ls}(spNdx);
                                L = Lsps{ls}(imSP);
                                labelList = Labels{ls};
                                save(outfile,'L','labelList','Lsp');
                            end
                        end
                    end
                end
            end
        end
    end
    close(pfig);
    
    %{
    for i = testFileLabeledNdx(:)'
        [folder base] = fileparts(testFileListLabeled{i});
        baseFName = fullfile(folder,base);
        outSPName = fullfile(HOMEDATADESC,'super_pixels',folder,[base '.mat']);
        load(outSPName);
        imSP = superPixels;
        spNdx = unique(imSP);
        for ls=1:length(HOMELABELSETS)
            [~, labelSet] = fileparts(HOMELABELSETS{ls});
            for c = 1:length(testParams.CombMethods)
                %fprintf('%s\n',testParams.CombMethods{c});
                probPerLabel = feval(testParams.CombMethods{c},spProbs{ls}(spNdx),spData{ls}(spNdx),spSizes(spNdx));
                outfile = fullfile(DataDir,testParams.MRFFold,labelSet,[testParams.CombMethods{c} '-' suffix],folder,[base '.mat']);make_dir(outfile);
                [~, Lsp] = max(probPerLabel,[],2);
                LspMap = zeros(spNdx(end),1);
                LspMap(spNdx) = Lsp;
                L = LspMap(imSP);
                %show(L,1);
                labelList = Labels{ls};
                save(outfile,'L','labelList','Lsp');
            end
        end
    end
    %}
end
%close(pfig);

end


