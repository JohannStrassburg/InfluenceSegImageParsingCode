function [LabelPixels LabelSPs] = MultiLevelPixMRFColor(HOMEDATA,HOMELABELSETS,testName,baseFName,Labels,imSP,im,dataCost,smoothingMatrix,labelSubSets,params)

if(~exist('canskip','var'))
    canskip = 1;
end
pixDataCost = 0;
if(size(dataCost{1},1)==numel(imSP))
    pixDataCost = 1;
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
            numLabels = size(dataCost{i},2);
            if(numLabels==length(labelList))
                goodFiles = goodFiles + 1;
            end
        end
    end
end
if(goodFiles == numLabelSets)
    return;
end

%reduce the label set to only plausable labels
[ro co ch] = size(im);
usedLs = cell(size(dataCost));
allUsedLs = [];
current = 0;
coDataCost = cell(size(dataCost));
for ls = 1:numLabelSets
    [foo Lmin] = min(dataCost{ls},[],2);
    usedLs{ls} = unique([Lmin; size(dataCost{ls},2)]);
    if(length(usedLs{ls}) == 1)
        dcT = dataCost{ls};
        dcT(:,usedLs{ls}) = max(dcT(:));
        [foo Lmin] = min(dcT,[],2);
        usedLs{ls} = unique([Lmin; usedLs{ls}]);
    end
    allUsedLs = [allUsedLs; current+usedLs{ls}(:)];
    current = current + size(dataCost{ls},2);
    dataCost{ls} = dataCost{ls}(:,usedLs{ls});
    if(~pixDataCost)
        coDataCost{ls} = zeros(numel(imSP),size(dataCost{ls},2));%dataCost{ls}(imSP(:),:);%
    else
        coDataCost{ls} = zeros(size(dataCost{ls}));%dataCost{ls};%
    end
end
smoothingMatrix = smoothingMatrix(allUsedLs,:);
smoothingMatrix = smoothingMatrix(:,allUsedLs);
numSP = length(unique(imSP));
numPix = numel(imSP);

if(exist('labelSubSets','var') && params.labelSubSetsWeight>0)
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


edgeFile = fullfile(HOMEDATA,'..','..','BSEdata','Images',[baseFName '.JPEG.pbs']); 
%{-
if(isfield(params,'edgeType') && strcmp(params.edgeType,'bse')  && exist(edgeFile,'file'))
    edgeData = read_array(edgeFile);
    minVal = .5; div = (1-minVal)/.95;
    vC = 1-edgeData{1};
    vC = vC(2:end-1,:);
    vC = vC.^4;
    %vC = .05+max(0,vC-minVal)/div;
    hC = 1-edgeData{2};
    hC = hC(:,2:end-1);
    hC = hC.^4;
    %hC = .05+max(0,hC-minVal)/div;
else
%}
    if(isfield(params,'edgeParam'))
        [hCS vCS] = SpatialCues(im2double(im),params.edgeParam);
    else
        [hCS vCS] = SpatialCues(im2double(im));
    end
    if(isfield(params,'edgeType') && strcmp(params.edgeType,'canny'))
        mask = edge(rgb2gray(im),'canny');
        vCS(mask<1) = 0;
        hCS(mask<1) = 0;
    end
    %hCS = min(hCS,vCS);
    %vCS = min(hCS,vCS);
    hCS = hCS(:,1:end-1); vCS = vCS(1:end-1,:);
    vC = exp(-vCS*5);
    hC = exp(-hCS*5);
    %b = [-.05*50; 50];
    %hCL = mnrval(b,hCS(:));hCL = reshape(hCL(:,2),[ro co-1]);
    %vCL = mnrval(b,vCS(:));vCL = reshape(vCL(:,2),[ro-1 co]);
end
%show(vC,3);show(hC,2);
%vC(:,2:end) = min(vC(:,2:end),hC(2:end,:));
%hC(2:end,:) = min(vC(:,2:end),hC(2:end,:));
vC = (vC-min(vC(:)))*(100/(max(vC(:))-min(vC(:))));
hC = (hC-min(hC(:)))*(100/(max(hC(:))-min(hC(:))));
%show(round(vC),5);show(round(hC),4);
%show(vC,3);show(hC,4);
%show(im,5);

if(isfield(params,'colorModel'))
    sz = 5;
    g = fspecial('gauss', [sz sz], sqrt(sz));
    imFlat = im;
    %imFlat = imfilter(im,g);
    if(isfield(params,'colorSpace'))
        if(strcmp(params.colorSpace,'lab'))
            c = makecform('srgb2lab');
            imFlat = applycform(imFlat,c);
        end
        if(strcmp(params.colorSpace,'l'))
            c = makecform('srgb2lab');
            imFlat = applycform(imFlat,c);
            imFlat = imFlat(:,:,1);
        end
        if(strcmp(params.colorSpace,'ab'))
            c = makecform('srgb2lab');
            imFlat = applycform(imFlat,c);
            imFlat = imFlat(:,:,2:3);
        end
        if(strcmp(params.colorSpace,'hsv'))
            imFlat = rgb2hsv(imFlat);
        end
        if(strcmp(params.colorSpace,'h'))
            imFlat = rgb2hsv(imFlat);
            imFlat = imFlat(:,:,1);
        end
        if(strcmp(params.colorSpace,'hs'))
            imFlat = rgb2hsv(imFlat);
            imFlat = imFlat(:,:,1:2);
        end
        if(strcmp(params.colorSpace,'hv'))
            imFlat = rgb2hsv(imFlat);
            imFlat = imFlat(:,:,[1 3]);
        end
    end
    [ro co ch] = size(imFlat);
    imFlat = double(reshape(imFlat,[ro*co ch]));
    divVal = max(1,1+(max(params.colorModel)/params.maxPenality));
    clusters = cell(size(dataCost));
    weights = cell(size(dataCost));
    covs = cell(size(dataCost));
    for ls = find(params.colorModel)
        dc = dataCost{ls}';
        if(~pixDataCost)
            dc = dc(:,imSP(:));
        end
        [foo Linit] = min(dc);
        
        clusters{ls} = cell(size(dc,1),1);
        weights{ls} = cell(size(dc,1),1);
        covs{ls} = cell(size(dc,1),1);
        [coDataCost{ls} clusters{ls} weights{ls} covs{ls}] = UpdateColorModel(Linit,imFlat,clusters{ls},weights{ls},covs{ls},params);
        coDataCost{ls} = params.colorModel(ls)*coDataCost{ls};
        
        for ls2 = 1:numLabelSets
            if(ls2==ls);continue;end
            coDataCost{ls2}(:,end) = coDataCost{ls}(:,1);
            coDataCost{ls2}(:,1:end-1) = repmat(coDataCost{ls}(:,2),[1 size(coDataCost{ls2},2)-1]);
        end
        %{
        mask = (coDataCost{ls}(:,1)+double(dataCost{ls}(:,1)))>coDataCost{ls}(:,2)+double(dataCost{ls}(:,2));show(reshape(mask,[ro co]),6);
        mask = (coDataCost{ls}(:,1)>(coDataCost{ls}(:,2)));show(reshape(mask,[ro co]),7);
        mask = (double(dataCost{ls}(:,1)))>double(dataCost{ls}(:,2));show(reshape(mask,[ro co]),8);
        show(reshape(dataCost{ls}(:,1),[ro co]),9);
        show(reshape(coDataCost{ls}(:,1),[ro co]),10);
        %}
    end
end


nodeMap = 1:numPix; nodeMap = reshape(nodeMap,size(imSP));
tmp1 = nodeMap(1:end-1,:);tmp2=nodeMap(2:end,:);
adjPairs = [tmp1(:) tmp2(:)];val = vC(:);
tmp1 = nodeMap(:,1:end-1);tmp2=nodeMap(:,2:end);
adjPairs = [adjPairs; [tmp1(:) tmp2(:)]];val = [val; hC(:)];
%{-
if(isfield(params,'connected') && params.connected==8)
    vCT = vC(:,1:end-1);hCT = hC(1:end-1,:);
    tmp1 = nodeMap(1:end-1,1:end-1);tmp2=nodeMap(2:end,2:end);
    adjPairs = [adjPairs; [tmp1(:) tmp2(:)]];val = [val; (hCT(:)+vCT(:))./2.8284271];
    vCT = vC(:,1:end-1);hCT = hC(2:end,:);
    tmp1 = nodeMap(2:end,1:end-1);tmp2=nodeMap(1:end-1,2:end);
    adjPairs = [adjPairs; [tmp1(:) tmp2(:)]];val = [val; (hCT(:)+vCT(:))./2.8284271];
end
%}

%create the adjacency graph between all nodes
sparseSmooth = sparse(adjPairs(:,1),adjPairs(:,2),val,numPix*numLabelSets,numPix*numLabelSets);
for i = 1:numLabelSets-1
    sparseSmooth = sparseSmooth + sparse(adjPairs(:,1)+numPix*i,adjPairs(:,2)+numPix*i,val,numPix*numLabelSets,numPix*numLabelSets);
end
s = ones(numPix,1);
for i = 1:numLabelSets
    for j = i+1:numLabelSets
        a = (i-1)*numPix+1:i*numPix;
        b = (j-1)*numPix+1:j*numPix;
        sparseSmooth = sparseSmooth + sparse(a,b,s,numPix*numLabelSets,numPix*numLabelSets);
        %sparseSmooth = sparseSmooth + sparse(b,a,s,numPix*numLabelSets,numPix*numLabelSets);
    end
end

%compute scaling to correct for int rounding errors
minTarget = 10;
minDataCost = minTarget;
for ls = 1:numLabelSets
    minDataCost = min(minDataCost,min(dataCost{ls}(:)));
end
multiplier = 1;%max(1,minTarget/minDataCost);
dataInt = cell(size(dataCost));

erImSP = zeros(size(imSP));
se = strel('disk',3);
for spNdx = 1:numSP
    erImSP(imerode(imSP==spNdx,se))=spNdx;
end
numInt = 0;
%show(erImSP)
for ls = 1:numLabelSets
    if(isinteger(dataCost{ls}))
        tmp = int32(dataCost{ls}');
        numInt = numInt + 1;
    else
        tmp = int32(ceil(dataCost{ls}'*multiplier));
    end
    if(pixDataCost)
        dataInt{ls} = tmp;
    else
        dataInt{ls} = tmp(:,imSP(:));
    end
end
if(numInt==numLabelSets)
    multiplier = 1;
end


%find initial labeling
Lin = zeros(numPix*numLabelSets,1);
current = 0;energy = 0;
for ls = 1:numLabelSets
    [E Lin((ls-1)*numPix+1:numPix*ls)] = min(dataInt{ls},[],1);
    energy = energy + sum(E);
    Lin((ls-1)*numPix+1:numPix*ls) = Lin((ls-1)*numPix+1:numPix*ls)+current;
    current = current + size(dataInt{ls},1);
end
numTotalLabs = current;
scInt = int32(ceil(smoothingMatrix*multiplier));
if(exist('flatlabelSubSets','var'))
    params.labelSubSetsWeight = int32(params.labelSubSetsWeight*multiplier);
end
numItt = 2;
if(isfield(params,'colorModel'))
    numItt = 4;
end

current = 0;
stemp = round(sparseSmooth);
Linit = [];
for ls = 1:numLabelSets
	numL = size(dataInt{ls},1);
    if(params.colorModel(ls)>0)
        dataIntTmp = (dataInt{ls} + int32(coDataCost{ls}'))./divVal;
    else
        dataIntTmp = dataInt{ls};
    end
    graph = GCO_Create(numPix,numL);
    GCO_SetVerbosity(graph,1);
    GCO_SetDataCost(graph,dataIntTmp);
    GCO_SetSmoothCost(graph,scInt((1:numL)+current,(1:numL)+current));
    GCO_SetNeighbors(graph,stemp((1:numPix)+numPix*(ls-1),(1:numPix)+numPix*(ls-1)));
    GCO_Expansion(graph);
    %GCO_Swap(graph);
    L = GCO_GetLabeling(graph);
    GCO_Delete(graph);
    show(reshape(L,size(imSP)),ls);drawnow;
    
    if(isfield(params,'colorModel') && params.colorModel(ls))
        for i = 2:numItt
            [coDataCost{ls} clusters{ls} weights{ls} covs{ls}] = UpdateColorModel(L,imFlat,clusters{ls},weights{ls},covs{ls},params);
            coDataCost{ls} = params.colorModel(ls)*coDataCost{ls};
            for ls2 = 1:numLabelSets
                if(ls2==ls);continue;end
                coDataCost{ls2}(:,end) = coDataCost{ls}(:,1);
                coDataCost{ls2}(:,1:end-1) = repmat(coDataCost{ls}(:,2),[1 size(coDataCost{ls2},2)-1]);
            end
            graph = GCO_Create(numPix,numL);
            GCO_SetVerbosity(graph,1);
            dataIntTmp = (dataInt{ls} + int32(coDataCost{ls}'))./divVal;
            GCO_SetDataCost(graph,dataIntTmp);
            GCO_SetSmoothCost(graph,scInt((1:numL)+current,(1:numL)+current));
            GCO_SetNeighbors(graph,stemp((1:numPix)+numPix*(ls-1),(1:numPix)+numPix*(ls-1)));
            GCO_Expansion(graph);
            %GCO_Swap(graph);
            L = GCO_GetLabeling(graph);
            GCO_Delete(graph);
            show(reshape(L,size(imSP)),ls);drawnow;
        end
    end
    
    Linit = [Linit; L+current];
    current = current+numL;
end

numItt = 2;
Lout = cell(numItt,1);
Lout{1} = Linit;
Energy = zeros(size(Lout));
Energy(1) = Inf;
dataIntTmp = dataInt;
scOrgInt = scInt;
for i=2:numItt
    graph = GCO_Create(numLabelSets*numPix,length(scInt));
    GCO_SetVerbosity(graph,1);
    current = 0;
    Lmaxl = [];initialE = 0;
    dataAll = 10000*ones(numTotalLabs,numPix*length(dataInt));
    for ls = 1:numLabelSets
        numL = size(dataInt{ls},1);
        if(params.colorModel(ls)>0)
            fgweight = 1;
            if(isfield(params,'fgDataWeight'))
                fgweight = params.fgDataWeight;
            end
            dataIntTmp{ls} = fgweight*(dataInt{ls} + int32(coDataCost{ls}'))./divVal;
            scInt((1:numL)+current,(1:numL)+current) = fgweight*scOrgInt((1:numL)+current,(1:numL)+current);
        else
            dataIntTmp{ls} = dataInt{ls};
        end
        dataAll((1:numL)+current,(1:numPix)+(ls-1)*numPix) = dataIntTmp{ls};
        for l = 1:numL
            GCO_SetDataCost(graph,int32([(ls-1)*numPix+1:numPix*(ls); dataIntTmp{ls}(l,:)]),l+current);
        end
        [e l] = min(dataIntTmp{ls});
        Lmaxl = [Lmaxl l+current];initialE = initialE+sum(e);
        current = current+numL;
    end
    %GCO_SetDataCost(graph,dataAll);
    GCO_SetLabeling(graph,Lout{i-1});
    if(exist('flatlabelSubSets','var'))
        for j = 1:length(flatlabelSubSets)
            GCO_SetLabelCost(graph,params.labelSubSetsWeight,flatlabelSubSets{j})
        end
    end
    GCO_SetSmoothCost(graph,scInt);
    stemp = round(sparseSmooth);
    GCO_SetNeighbors(graph,stemp);
    %GCO_SetLabeling(graph,Lin);
    GCO_SetLabelOrder(graph,numTotalLabs:-1:1);%randperm(numTotalLabs)
    GCO_Expansion(graph);
    %GCO_Swap(graph);
    Lout{i} = GCO_GetLabeling(graph);
    Energy(i) = GCO_ComputeEnergy(graph);
    current = 0;
    for ls = 1:numLabelSets
        L = Lout{i}(1+(ls-1)*numPix:ls*numPix);
        L = L-current;
        %show(reshape(L,size(imSP)),ls+5);drawnow;
        current = current+length(usedLs{ls});
        if(isfield(params,'colorModel') && i~=numItt)
            if(params.colorModel(ls))
                [coDataCost{ls} clusters{ls} weights{ls} covs{ls}] = UpdateColorModel(L,imFlat,clusters{ls},weights{ls},covs{ls},params);
                coDataCost{ls} = params.colorModel(ls)*coDataCost{ls};
                for ls2 = 1:numLabelSets
                    if(ls2==ls);continue;end
                    coDataCost{ls2}(:,end) = coDataCost{ls}(:,1);
                    coDataCost{ls2}(:,1:end-1) = repmat(coDataCost{ls}(:,2),[1 size(coDataCost{ls2},2)-1]);
                end
                %{
                mask = (coDataCost{ls}(:,1)+double(dataCost{ls}(:,1)))>coDataCost{ls}(:,2)+double(dataCost{ls}(:,2));show(reshape(mask,[ro co]),6);
                mask = (coDataCost{ls}(:,1)>(coDataCost{ls}(:,2)));show(reshape(mask,[ro co]),7);
                mask = (double(dataCost{ls}(:,1)))>double(dataCost{ls}(:,2));show(reshape(mask,[ro co]),8);
                show(reshape(dataCost{ls}(:,1),[ro co]),9);
                show(reshape(coDataCost{ls}(:,1),[ro co]),10);
                %}
            end
        end
    end
    GCO_Delete(graph);
end

if(isfield(params,'colorModel')&& any(params.colorModel>0))
    Energy = Energy(numItt);
    Lout = Lout{numItt};
else
    ndx = find(Energy==min(Energy));
    Energy = Energy(max(ndx));
    Lout = Lout{max(ndx)};
end

if(~isfield(params,'labelSubSetsWeight'))
    params.labelSubSetsWeight=0;
end

if(sum(scInt(:))==0 && params.labelSubSetsWeight==0)
    if(sum(Lout~=Lin)>0)
        fprintf('!!!!!!!!!!!!!!ERROR Graph Cut Broken %d!!!!!!!!!!!!!!!!!\n',sum(Lout~=Lin));
        %keyboard;
    end
end

current = 0;
for ls = 1:numLabelSets
    L = Lout(1+(ls-1)*numPix:ls*numPix);
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
    current = current+length(usedLs{ls});
    
    PI = regionprops(imSP,'PixelIdxList');
    for spNdx = 1:max(imSP(:));
        spLabels = L(PI(spNdx).PixelIdxList);
        [a b] = UniqueAndCounts(spLabels);
        [foo mxNdx] = max(b);
        Lsp(spNdx) = a(mxNdx);
    end
    L = reshape(L,size(imSP));
    %show(L,ls,0);
    labelList = Labels{ls};
    make_dir(outFileNames{ls});
    save(outFileNames{ls},'L','labelList','Lsp');
    LabelPixels{ls} = L;
    LabelSPs{ls} = Lsp;
end

end


function [coDataCost clusters weights covs] = UpdateColorModel(Linit,imFlat,clusters,weights,covs,params)
    ch = size(imFlat,2);
    numClusters = params.numClusters;
    for l = 1:length(clusters)
        lcolors = imFlat(Linit==l,:);
        if(isempty(lcolors))
            rndx = randperm(size(imFlat,1));
            lcolors = imFlat(rndx(1:(10*numClusters)),:);
        end
        if(isempty(clusters{l}))
            if(strcmp(params.clType,'kmeans'))
                for cn = numClusters:-1:1; try [lId kclusters] = kmeans(lcolors, cn,'Online','off'); break; catch; end; end;%'Display','iter',
            elseif(strcmp(params.clType,'gmm'))
                for cn = numClusters:-1:1; try 
                        options = statset('Display','final'); 
                        obj = gmdistribution.fit(lcolors,cn,'Options',options); 
                        lId = cluster(obj,lcolors);
                        break; 
                    catch; end; 
                end;%'Display','iter',
            end
            clusters{l} = zeros(ch,cn);%kclusters';
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
        [dist, ind] = ClustDistMembership(imFlat, clusters{l}, covs{l}, weights{l});
        coDataCost(:,l) = dist;
    end
    %avg = mean(coDataCost(:)); stdev = std(coDataCost(:)); 
    %b = [-avg/stdev; 1/stdev];
    %temp = mnrval(b,coDataCost(:));
    %coDataCost = reshape(temp(:,1),size(coDataCost));
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