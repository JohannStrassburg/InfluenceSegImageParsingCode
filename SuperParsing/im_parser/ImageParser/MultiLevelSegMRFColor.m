function [LabelPixels LabelSPs] = MultiLevelSegMRFColor(HOMEDATA,HOMELABELSETS,testName,baseFName,Labels,imSP,adjPairs,dataCost,smoothingMatrix,labelSubSets,labelSubSetsWeight,colorModel,meanSPColors)

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
if(goodFiles == numLabelSets)
    return;
end

%reduce the label set to only plausable labels
usedLs = cell(size(dataCost));
[ro co] = size(imSP);
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
    

if(exist('colorModel','var'))
    clusters = cell(size(dataCost));
    weights = cell(size(dataCost));
    covs = cell(size(dataCost));
    for ls = find(colorModel)
        dc = dataCost{ls}';
        [foo Linit] = min(dc);
        
        clusters{ls} = cell(size(dc,1),1);
        weights{ls} = cell(size(dc,1),1);
        covs{ls} = cell(size(dc,1),1);
        [coDataCost{ls} clusters{ls} weights{ls} covs{ls}] = UpdateColorModel(Linit,meanSPColors,clusters{ls},weights{ls},covs{ls});
        coDataCost{ls} = colorModel(ls)*coDataCost{ls};
        for ls2 = 1:numLabelSets
            if(ls2==ls);continue;end
            coDataCost{ls2}(:,end) = coDataCost{ls}(:,1);
        end
        %{
        mask = (coDataCost{ls}(:,1)+double(dataCost{ls}(:,1)))>coDataCost{ls}(:,2)+double(dataCost{ls}(:,2));show(mask(imSP),6);
        mask = (coDataCost{ls}(:,1)>(coDataCost{ls}(:,2)));show(mask(imSP),7);
        mask = (double(dataCost{ls}(:,1)))>double(dataCost{ls}(:,2));show(mask(imSP),8);
        mask = dataCost{ls}(:,1);show(mask(imSP),9);
        mask = coDataCost{ls}(:,1);show(mask(imSP),10);
        %}
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
    for ls = 1:numLabelSets
        L = Lout{i}(1+(ls-1)*numSP:ls*numSP);
        L = L-current;
        %show(L(imSP),ls);
        current = current+length(usedLs{ls});
        if(exist('colorModel','var') && i~=numItt)
            if(colorModel(ls))
                [coDataCost{ls} clusters{ls} weights{ls} covs{ls}] = UpdateColorModel(L,meanSPColors,clusters{ls},weights{ls},covs{ls});
                coDataCost{ls} = colorModel(ls)*coDataCost{ls};
                for ls2 = 1:numLabelSets
                    if(ls2==ls);continue;end
                    coDataCost{ls2}(:,end) = coDataCost{ls}(:,1);
                end
                %{
                mask = (coDataCost{ls}(:,1)+double(dataCost{ls}(:,1)))>coDataCost{ls}(:,2)+double(dataCost{ls}(:,2));show(mask(imSP),6);
                mask = (coDataCost{ls}(:,1)>(coDataCost{ls}(:,2)));show(mask(imSP),7);
                mask = (double(dataCost{ls}(:,1)))>double(dataCost{ls}(:,2));show(mask(imSP),8);
                mask = dataCost{ls}(:,1);show(mask(imSP),9);
                mask = coDataCost{ls}(:,1);show(mask(imSP),10);
                %}
            end
        end
    end
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
    L = L(imSP);
    labelList = Labels{ls};
    make_dir(outFileNames{ls});
    save(outFileNames{ls},'L','labelList','Lsp');
    LabelPixels{ls} = L;
    LabelSPs{ls} = Lsp;
    current = current+length(usedLs{ls});
end

end


function [coDataCost clusters weights covs] = UpdateColorModel(Linit,meanSPColors,clusters,weights,covs)
    ch = size(meanSPColors,2);
    numClusters = 5;
    for l = 1:length(clusters)
        lcolors = meanSPColors(Linit==l,:);
        if(isempty(lcolors))
            rndx = randperm(size(meanSPColors,1));
            lcolors = meanSPColors(rndx(1:(numClusters)),:);
        end
        if(isempty(clusters{l}))
            if(size(lcolors,1)==1)
                lId = 1;
                kclusters = lcolors;
                cn =1;
            else
                for cn = numClusters:-1:1; 
                    try 
                        [lId kclusters] = kmeans(lcolors, cn,'Online','off'); 
                        [a b] = UniqueAndCounts(lId);
                        if(all(b>ch)); 
                            break; 
                        end
                    catch; end; 
                end;
            end
            clusters{l} = kclusters';
            weights{l}  = zeros(1,cn);
            covs{l}  = zeros(ch, ch, cn);
        else
            [dist, lId] = ClustDistMembership(lcolors, clusters{l}, covs{l}, weights{l});
            cn = size(clusters{l},2);
        end
        
        for k=1:cn
            relColors = lcolors(lId==k,:);        %% Colors belonging to cluster k
            clusters{l}(:,k) = mean(relColors,1)';
            covs{l}(:,:,k) = cov(relColors);
            weights{l}(1,k) = sum(lId==k) / length(lId);
        end
        [dist, ind] = ClustDistMembership(meanSPColors, clusters{l}, covs{l}, weights{l});
        coDataCost(:,l) = dist;
    end
    %{
    avg = mean(coDataCost(:)); stdev = std(coDataCost(:)); 
    b = [-avg/stdev; 1/stdev];
    temp = mnrval(b,coDataCost(:));
    coDataCost = reshape(temp(:,1),size(coDataCost));
    %}
    coDataCost = coDataCost./repmat(sum(coDataCost,2),[1 2]);
end


function [FDist, FInd] = ClustDistMembership(MeanColors, FCClusters, FCovs, FWeights)
% CLUSTDISTMEMBERSHIP - Calcuates FG and BG Distances
% Authors - Mohit Gupta, Krishnan Ramnath
% Affiliation - Robotics Institute, CMU, Pittsburgh
% 2006-05-15

NumFClusters = size(FCClusters,2);
numULabels = size(MeanColors,1);

FDist = zeros(numULabels,1);
FInd = zeros(numULabels,1);

Ftmp = zeros(numULabels, NumFClusters);

for k = 1:NumFClusters
    M = FCClusters(:,k);
    CovM = FCovs(:,:,k);
    W = FWeights(1,k);

    V = MeanColors - repmat(M',numULabels,1);
    Ftmp(:,k) = -log((W / sqrt(det(CovM))) * (eps+exp(-( sum( ((V * inv(CovM)) .* V),2) /2))));

end 
Ftmp = real(Ftmp);
Ftmp(Ftmp<=0) = max(Ftmp(:));
[FDist, FInd] = min(Ftmp,[],2);
end