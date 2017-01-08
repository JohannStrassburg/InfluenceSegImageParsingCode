function [LabelPixels LabelSPs] = MultiLevelSegMRF(HOMEDATA,HOMELABELSETS,testName,baseFName,Labels,imSP,adjPairs,dataCost,smoothingMatrix,labelSubSets,labelSubSetsWeight)

if(~exist('canskip','var'))
    canskip = 1;
end

numLabelSets = length(dataCost);
LabelPixels = cell(numLabelSets,1);
LabelSPs = cell(numLabelSets,1);
goodFiles = 0;
outFileNames = cell(numLabelSets,1);
for i = 1:numLabelSets
    [foo LabelFolder] = fileparts(HOMELABELSETS{i});
    if(~isempty(baseFName))
        outFileNames{i} = fullfile(HOMEDATA,LabelFolder,testName,sprintf('%s.mat',baseFName));
        if(exist(outFileNames{i},'file')&&canskip)
            load(outFileNames{i});
            LabelPixels{i} = L;
            LabelSPs{i} = Lsp;
            if(exist('labelList','var'))
                if(size(dataCost{i},2)==length(labelList))
                    goodFiles = goodFiles + 1;
                end
            end
        end
    end
end
if(goodFiles == numLabelSets)
    return;
end

%reduce the label set to only plausable labels
usedLs = cell(size(dataCost));
allUsedLs = [];
current = 0;
for ls = 1:numLabelSets
    [foo Lmin] = min(dataCost{ls},[],2);
    usedLs{ls} = unique([Lmin; 1; size(dataCost{ls},2)]);
    allUsedLs = [allUsedLs; current+usedLs{ls}(:)];
    current = current + size(dataCost{ls},2);
    dataCost{ls} = dataCost{ls}(:,usedLs{ls});
end
coDataCost = dataCost;
smoothingMatrix = smoothingMatrix(allUsedLs,:);
smoothingMatrix = smoothingMatrix(:,allUsedLs);
numSP = size(dataCost{1},1);

if(exist('labelSubSets','var') && labelSubSetsWeight>0)
    currentStartLabel = 0;
    flatlabelSubSets = cell(0);
    for ls = 1:numLabelSets
        for i = 1:length(labelSubSets{ls})
            [foo subSetLocalNdx] = intersect(usedLs{ls},labelSubSets{ls}{i});
            if(~isempty(subSetLocalNdx))
                flatlabelSubSets{end+1} = int32(subSetLocalNdx+currentStartLabel);
            end
        end
        currentStartLabel = currentStartLabel + length(usedLs{ls});
    end
end

%create the adjacency graph between all nodes
sparseSmooth = sparse(adjPairs(:,1),adjPairs(:,2),ones(size(adjPairs,1),1),numSP*numLabelSets,numSP*numLabelSets);
for i = 1:numLabelSets-1
    sparseSmooth = sparseSmooth + sparse(adjPairs(:,1)+numSP*i,adjPairs(:,2)+numSP*i,ones(size(adjPairs,1),1),numSP*numLabelSets,numSP*numLabelSets);
end
s = ones(numSP,1);
for i = 1:numLabelSets
    for j = i+1:numLabelSets
        a = (i-1)*numSP+1:i*numSP;
        b = (j-1)*numSP+1:j*numSP;
        sparseSmooth = sparseSmooth + sparse(a,b,s,numSP*numLabelSets,numSP*numLabelSets);
        sparseSmooth = sparseSmooth + sparse(b,a,s,numSP*numLabelSets,numSP*numLabelSets);
    end
end
sparseSmooth(sparseSmooth>0) = 1;
%compute scaling to correct for int rounding errors
minTarget = 100;
minDataCost = minTarget;
for ls = 1:numLabelSets
    minDataCost = min(minDataCost,min(dataCost{ls}(:)));
end
multiplier = max(1,minTarget/minDataCost);
dataInt = cell(size(dataCost));
numInt = 0;
for ls = 1:numLabelSets
    if(isinteger(dataCost{ls}))
        dataInt{ls} = int32(dataCost{ls}');
        numInt = numInt + 1;
    else
        dataInt{ls} = int32(ceil(dataCost{ls}'*multiplier));
    end
end
if(numInt==numLabelSets)
    multiplier = 1;
end

%find initial labeling
Lin = zeros(numSP*numLabelSets,1);
current = 0;
for ls = 1:numLabelSets
    [foo Lin((ls-1)*numSP+1:numSP*ls)] = min(dataInt{ls},[],1);
    Lin((ls-1)*numSP+1:numSP*ls) = Lin((ls-1)*numSP+1:numSP*ls)+current;
    current = current + size(dataInt{ls},1);
end
numTotalLabs = current;
scInt = int32(ceil(smoothingMatrix*multiplier));
if(exist('flatlabelSubSets','var'))
    labelSubSetsWeight = int32(labelSubSetsWeight*multiplier);
end
numItt = 5;
Lout = cell(numItt,1);
Energy = zeros(size(Lout));
dataIntTmp = dataInt;
for i=1:numItt
    graph = GCO_Create(numLabelSets*numSP,length(scInt));
    GCO_SetVerbosity(graph,0)
    current = 0;
    for ls = 1:numLabelSets
        numL = size(dataInt{ls},1);
        dataIntTmp{ls} = (dataInt{ls} + int32(coDataCost{ls}'))./2;
        for l = 1:numL
            GCO_SetDataCost(graph,int32([(ls-1)*numSP+1:numSP*(ls); dataIntTmp{ls}(l,1:numSP)]),l+current);
        end
        current = current+numL;
    end
    if(exist('flatlabelSubSets','var'))
        for j = 1:length(flatlabelSubSets)
            GCO_SetLabelCost(graph,labelSubSetsWeight,flatlabelSubSets{j})
        end
    end
    GCO_SetSmoothCost(graph,scInt);
    stemp = ceil(sparseSmooth);
    GCO_SetNeighbors(graph,stemp);
    GCO_SetLabeling(graph,Lin);
    GCO_SetLabelOrder(graph,1:numTotalLabs);%randperm(numTotalLabs))
    
    if(exist('flatlabelSubSets','var'))
        GCO_Expansion(graph);
    else
        %GCO_Expansion(graph)
    	GCO_Swap(graph);
    end
    Lout{i} = GCO_GetLabeling(graph);
    Energy(i) = GCO_ComputeEnergy(graph);
    current = 0;
    GCO_Delete(graph);
end
   
if(exist('colorModel','var')&& any(colorModel>0))
    Energy = Energy(numItt);
    Lout = Lout{numItt};
else
    [foo ndx] = min(Energy);
    Energy = Energy(ndx);
    Lout = Lout{ndx};
end

if(sum(scInt(:))==0 && (~exist('flatlabelSubSets','var') || labelSubSetsWeight==0) && (~exist('colorModel','var') || all(colorModel==0)) )
    energy = 0;
    for ls = 1:numLabelSets
        [foo] = min(dataInt{ls});
        energy = energy + sum(foo);
    end
    if(sum(Lout~=Lin)>0)
        current = 0;
        inEnergy = 0;
        outEnergy = 0;
        for ls = 1:numLabelSets
            inEnergy = inEnergy+sum(dataInt{ls}(sub2ind(size(dataInt{ls}),min(size(dataInt{ls},1),max(1,double(Lin((ls-1)*numSP+1:numSP*ls)')-current)),1:size(dataInt{ls},2))));
            outEnergy = outEnergy+sum(dataInt{ls}(sub2ind(size(dataInt{ls}),min(size(dataInt{ls},1),max(1,double(Lout((ls-1)*numSP+1:numSP*ls)')-current)),1:size(dataInt{ls},2))));
            current = current + size(dataInt{ls},1);
        end
        fprintf('!!!!!!!!!!!!!!ERROR Graph Cut Broken %d: %.1f vs. %.1f!!!!!!!!!!!!!!!!!\n',sum(Lout~=Lin), inEnergy, outEnergy);
        %Lout=Lin;
        %keyboard;
    end
end

current = 0;
for ls = 1:numLabelSets
    L = Lout(1+(ls-1)*numSP:ls*numSP);
    L = L-current;
    %correct for glitches in the graph cut code
    labelsFound = unique(L);
    inds = find(labelsFound<1|labelsFound>length(usedLs{ls}));
    if(~isempty(inds))
        for lfound = labelsFound(inds(:)')'
            L(L==lfound) = 1;
            fprintf('Label from other labels set used!!!\n');
        end
    end
    
    L = usedLs{ls}(L);
    Lsp = L;
    if(~isempty(imSP))
        L = L(imSP);
        if(~isempty(baseFName))
            labelList = Labels{ls};
            make_dir(outFileNames{ls});
            save(outFileNames{ls},'L','labelList','Lsp');
        end
        LabelPixels{ls} = L;
    end
    LabelSPs{ls} = Lsp;
    current = current+length(usedLs{ls});
end

end
