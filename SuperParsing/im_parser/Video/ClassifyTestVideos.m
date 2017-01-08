function [timing] = ClassifyTestVideos(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex,trainCounts,Labels,classifiers,testParams,fullSPDesc)

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
glSuffix = '';
timing = zeros(length(testFileList),4);


testDirList = (FileList2DirList(testFileList(range)));
testDirs = unique(testDirList);
for d = 1:length(testDirs)
    testDir = testDirs{d};
    testFileNdx = find(strcmp(testDir,testDirList));
    for i = testFileNdx(:)'
        %im = imread(fullfile(HOMEDATA,'..','Images',testFileList{i}));[ro co ch] = size(im);
        [folder file] = fileparts(testFileList{i});
        baseFName = fullfile(folder,file);
        classifierStr = repmat('0',[1 length(HOMELABELSETS)]);
        probPerLabel = cell(size(HOMELABELSETS));
        dataCost = cell(size(probPerLabel));
        for labelType=1:length(HOMELABELSETS)
            [foo labelSet] = fileparts(HOMELABELSETS{labelType});
            if(isempty(classifiers{labelType}))
                Kndx = 1;
                if(testParams.retSetSize >= length(trainFileList) )
                    retSetIndex = trainIndex{labelType,Kndx};
                else
                    timing(i,1)=toc;  [retInds rank]= FindRetrievalSet(trainGlobalDesc,SelectDesc(testGlobalDesc,i,1),DataDir,baseFName,testParams,glSuffix);  timing(i,1)=toc-timing(i,1);
                    [retSetIndex descMask] = PruneIndex(trainIndex{labelType,Kndx},retInds,testParams.retSetSize,testParams.minSPinRetSet);
                end
                suffix = sprintf('R%dK%d',testParams.retSetSize,testParams.K(Kndx));%nn%d  ,testParams.targetNN
                suffix = [suffix glSuffix];
                labelNums = 1:length(trainCounts{labelType});
                probPerLabel{labelType} = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,suffix,retSetIndex,[],labelNums,trainCounts{labelType,Kndx},probType,1); %#ok<AGROW>
                if(isempty(probPerLabel{labelType}))
                    if(testParams.busyFile)
                        busyFile = fullfile(DataDir,'BusyFile',[baseFName '.mat']);
                        if(exist(busyFile,'file')) continue; end
                        a = 1; make_dir(busyFile); save(busyFile,'a');
                    end
                    [testImSPDesc] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K,testParams.segSuffix);
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
                        timing(i,2)=toc;  rawNNs = DoRNNSearch(testImSPDesc,retSetSPDesc,fullfile(DataDir,labelSet),baseFName,suffix,testParams,Kndx,0);  timing(i,2)=toc-timing(i,2);
                    end
                    timing(i,3)=toc;  probPerLabel{labelType} = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,suffix,retSetIndex,rawNNs,labelNums,trainCounts{labelType,Kndx},probType,1);  timing(i,3)=toc-timing(i,3);
                end
                %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
                dataCost{labelType} = testParams.maxPenality*(1-(1./(1+exp(-(testParams.BConst(1)+probPerLabel{labelType}.*testParams.BConst(2))))));
            else
                probCacheFile = fullfile(DataDir,['ClassifierOutput' testParams.clSuffix],labelSet,[baseFName '.mat']);
                classifierStr(labelType) = '1';
                if(~exist(probCacheFile,'file'))
                    if(testParams.busyFile)
                        busyFile = fullfile(DataDir,'BusyFile',[baseFName '.mat']);
                        if(exist(busyFile,'file')) continue; end
                        a = 1; make_dir(busyFile); save(busyFile,'a');
                    end
                    [testImSPDesc] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K,testParams.segSuffix);
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
                %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
                dataCost{labelType} = testParams.maxPenality*(1-(1./(1+exp(-(testParams.BConstCla(1)+probPerLabel{labelType}.*testParams.BConstCla(2))))));
            end
        end
        if(testParams.busyFile&&exist(busyFile,'file'))
            delete(busyFile);
        end
        ProgressBar(pfig,find(i==range),length(range));
    end
end
close(pfig);

end


