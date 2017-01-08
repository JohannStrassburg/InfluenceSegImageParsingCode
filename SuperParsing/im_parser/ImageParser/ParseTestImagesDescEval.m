function [probPerLabelPerDesc imSPs] = ParseTestImagesDescEval(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex,trainCounts,labelPenality,Labels,classifiers,globalSVM,testParams,fullSPDesc)

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
range = length(testFileList):-1:1;
if(isfield(testParams,'range'))
    range = testParams.range;
end
timing = zeros(length(testFileList),4);
glSuffix = '';
probPerLabelPerDesc = cell(max(range),1);
imSPs = cell(max(range),1);
for i = range
    im = imread(fullfile(HOMEDATA,'..','Images',testFileList{i}));
    [ro co ch] = size(im);
    [folder file] = fileparts(testFileList{i});
    baseFName = fullfile(folder,file);
    
    timing(i,1)=toc;  [retInds rank]= FindRetrievalSet(trainGlobalDesc,SelectDesc(testGlobalDesc,i,1),DataDir,baseFName,testParams,glSuffix);  timing(i,1)=toc-timing(i,1);
    
    FGSet = 0;Kndx = 1;
    classifierStr = repmat('0',[1 length(HOMELABELSETS)]);
    preStr = '';%num2str(testParams.K);
    clear imSP testImSPDesc;
    [testImSPDesc imSP] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K,testParams.segSuffix);
    imSPs{i} = imSP;
    probPerLabel = cell(size(HOMELABELSETS));
    probPerLabelPerDesc{i} = cell(size(HOMELABELSETS));
    dataCost = cell(size(probPerLabel));
    for labelType=1:length(HOMELABELSETS)
        [foo labelSet] = fileparts(HOMELABELSETS{labelType});
        if(testParams.retSetSize == length(trainFileList) )
            retSetIndex = trainIndex{labelType,Kndx};
        else
            [retSetIndex descMask] = PruneIndex(trainIndex{labelType,Kndx},retInds,testParams.retSetSize,testParams.minSPinRetSet);
        end
        suffix = sprintf('R%dK%dTNN%d',testParams.retSetSize,testParams.K(Kndx),testParams.targetNN);%nn%d  ,testParams.targetNN
        suffix = [suffix testParams.globalDescSuffix];
        suffix = [suffix glSuffix];
        probSuffix = [suffix '-sc' myN2S(testParams.smoothingConst) probType];
        labelNums = 1:length(trainCounts{labelType});
        [probPerLabel{labelType} timing(i,3) probPerLabelPerDesc{i}{labelType}] = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,probSuffix,retSetIndex,[],labelNums,trainCounts{labelType,Kndx},probType,testParams.smoothingConst,1); %#ok<AGROW>
        if(~isempty(probPerLabel{labelType}) && size(probPerLabel{labelType},1)~=size(testImSPDesc.gist_int,1))
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
            [probPerLabel{labelType} timing(i,3) probPerLabelPerDesc{i}{labelType}] = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,probSuffix,retSetIndex,rawNNs,labelNums,trainCounts{labelType,Kndx},probType,testParams.smoothingConst,1);
        end
        pplpd = probPerLabelPerDesc{i}{labelType};
        labelList = Labels{labelType};
        for d = 1:size(pplpd,3)
            saveFile = fullfile(DataDir,'Round00',labelSet,[probSuffix '-' testParams.segmentDescriptors{d}],folder,[file '.mat']);make_dir(saveFile);
            [foo Lsp] = max(pplpd(:,:,d),[],2);
            L = Lsp(imSP);
            save(saveFile,'L','labelList','Lsp');
        end
    end
    %}
    ProgressBar(pfig,find(i==range),length(range));
end
close(pfig);

for labelType=1:length(HOMELABELSETS)
    labelList = Labels{labelType};
    [foo labelSet] = fileparts(HOMELABELSETS{labelType});
    [conMats perPixelRates] =  EvaluateTests(HOMEDATA,HOMELABELSETS(labelType),{testParams.TestString},'Round00',[],[],'Round00');
    usedDesc = [];
    unUsedDesc = 1:length(testParams.segmentDescriptors);
    [sortDescList descOrder] = sort(testParams.segmentDescriptors);
    lastRate = 0;
    for r = 1:length(testParams.segmentDescriptors)
        [rate ndx] = sort(perPixelRates{1},'descend');
        descNdx = unUsedDesc(descOrder(ndx));
        fprintf('Round %d: \n',r);
        for j = 1:length(descNdx);fprintf('%0.3f %0.3f: %s\n',rate(j), rate(j)-lastRate, testParams.segmentDescriptors{descNdx(j)});end
        lastRate = rate(1);
        usedDesc = [usedDesc descNdx(1)];
        unUsedDesc(unUsedDesc==descNdx(1)) = [];
        [sortDescList descOrder] = sort(testParams.segmentDescriptors(unUsedDesc));
        for i = range
            [folder file] = fileparts(testFileList{i});
            imSP = imSPs{i};
            pplpd = probPerLabelPerDesc{i}{labelType};
            ppl = sum(pplpd(:,:,usedDesc),3);
            saveFile = fullfile(DataDir,[labelSet '-R'],sprintf('Round%2d-%s',r,testParams.segmentDescriptors{usedDesc(end)}),labelSet,probSuffix,folder,[file '.mat']);make_dir(saveFile);
            [foo Lsp] = max(ppl,[],2);
            L = Lsp(imSP);
            save(saveFile,'L','labelList','Lsp');
            for d = unUsedDesc
                saveFile = fullfile(DataDir,[labelSet '-R'],sprintf('Round%2d-%s',r,testParams.segmentDescriptors{usedDesc(end)}),labelSet,[probSuffix '-' testParams.segmentDescriptors{d}],folder,[file '.mat']);make_dir(saveFile);
                [foo Lsp] = max(ppl+pplpd(:,:,d),[],2);
                L = Lsp(imSP);
                save(saveFile,'L','labelList','Lsp');
            end
        end
        [conMats perPixelRates] =  EvaluateTests(HOMEDATA,HOMELABELSETS(labelType),{fullfile(testParams.TestString,[labelSet '-R'])},sprintf('Round%2d-%s',r,testParams.segmentDescriptors{usedDesc(end)}),[],[],sprintf('Round%2d-%s',r,testParams.segmentDescriptors{usedDesc(end)}));
        perPixelRates{1}(1) = [];
    end
end
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

