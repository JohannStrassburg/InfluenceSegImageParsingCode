function [segmentLabelIndex, labels, counts] = LoadSegmentLabelIndex_Video(fileList,excludeLabels,HOMELABELSET,HOMEDESC,segFolderName)

HOMESUPERPIX = fullfile(HOMEDESC,segFolderName,'super_pixels');
segFolderName = [segFolderName];
saveFile = fullfile(HOMELABELSET,segFolderName,'segIndex-video.mat');
doSave = 1; 
if(exist(saveFile,'file'))
    load(saveFile);
    images = unique(segmentLabelIndex.image);
    if(isfield(segmentLabelIndex,'spSize'))
        if(max(images) == length(fileList))
            return;
        else
            doSave = 0;
        end
    end
end
%{-
segmentLabelIndex.label = [];
segmentLabelIndex.sp = [];
segmentLabelIndex.spSize = [];
segmentLabelIndex.image = [];
segmentLabelIndex.fold = [];
%}
i = 0;
while(~exist('labels','var'))
    i = i+1;
    [fold name] = fileparts(fileList{i});
    segFile = fullfile(HOMELABELSET,fold,[name '.mat']);
    if(~exist(segFile,'file'));continue;end
    load(segFile);
    labels=names;
end
excludeMask = zeros(size(labels))==1;
for j = 1:length(labels)
    excludeMask(j) = sum(strcmp(excludeLabels,labels{j}))>0;
end
excludeLabels = find(excludeMask);
foldList = cell(0);%{'01','05','06','16'};%
%{-
pfig = ProgressBar;
for i= 1:length(fileList)
    [fold name] = fileparts(fileList{i});
    foldNdx = find(strcmp(fold,foldList));
    if(isempty(foldNdx));
        foldList{end+1} = fold;
        foldNdx = length(foldList);
    end
    spFile = fullfile(HOMESUPERPIX,fold,[name '.mat']);
    load(spFile);
    S = [];
    segFile = fullfile(HOMELABELSET,fold,[name '.mat']);
    if(exist(segFile,'file')); load(segFile); end
    indexFile = fullfile(HOMELABELSET,segFolderName,fold,[name '.mat']);
    index = [];
    if(exist(indexFile,'file'))
        load(indexFile);
        if(mod(i,100)==0)
            ProgressBar(pfig,i,length(fileList));
        end
    end
    if(~isfield(index,'spSize'))
        if(isempty(S))
            index = GetIndexForImageEmpty(superPixels);
        else 
            index = GetIndexForImage(S,superPixels,excludeLabels);
        end
        make_dir(indexFile);save(indexFile,'index');
        ProgressBar(pfig,i,length(fileList));
    end
    
    segmentLabelIndex.label = [segmentLabelIndex.label index.label];
    segmentLabelIndex.sp = [segmentLabelIndex.sp index.sp];
    segmentLabelIndex.image = [segmentLabelIndex.image ones(size(index.sp))*i];
    segmentLabelIndex.spSize = [segmentLabelIndex.spSize index.spSize];
    segmentLabelIndex.fold = [segmentLabelIndex.fold ones(size(index.sp))*foldNdx];
    
end
close(pfig);
%}
pfig = sp_progress_bar('Transfering Labels');
foldsNdx = unique(segmentLabelIndex.fold);
for i = foldsNdx(:)'
    ndx = find(segmentLabelIndex.fold==i);
    labeledNdx = segmentLabelIndex.label(ndx)~=0;
    spNums = unique(segmentLabelIndex.sp(ndx(labeledNdx)));
    tic
    for j = 1:length(spNums)
        spNum = spNums(j);
        spNumNdx = ndx(segmentLabelIndex.sp(ndx)==spNum);
        ls = segmentLabelIndex.label(spNumNdx);
        spSize = segmentLabelIndex.spSize(spNumNdx);
        votes = zeros(size(labels));
        for k = find(ls(:)'>0)
            votes(ls(k)) = votes(ls(k))+spSize(k);
        end
        [foo l] = max(votes);
        segmentLabelIndex.label(spNumNdx) = l;
        sp_progress_bar(pfig,i,max(foldsNdx),j,length(spNums),['Folder: ' foldList{i}]);
    end
end
close(pfig);
rmNdx = segmentLabelIndex.label==0;
segmentLabelIndex.label(rmNdx) = [];
segmentLabelIndex.sp(rmNdx) = [];
segmentLabelIndex.spSize(rmNdx) = [];
segmentLabelIndex.image(rmNdx) = [];
segmentLabelIndex.fold(rmNdx) = [];

[l ctemp] = UniqueAndCounts(segmentLabelIndex.label);
[l ind] = sort(l);
ctemp = ctemp(ind);
counts = zeros(size(labels));
counts(l) = ctemp;
%labels = labels(l); % do not add back without doing the same for counts
if(doSave)
    make_dir(saveFile);save(saveFile,'segmentLabelIndex', 'labels', 'counts');
end
end

function [index] = GetIndexForImageEmpty(superPixels)
    index.sp = unique(superPixels)';
    index.label=zeros(size(index.sp));
    index.spSize=zeros(size(index.sp));
    for i = 1:length(index.sp)
        index.spSize(i) = sum(superPixels(:)==index.sp(i));
    end
end
function [index] = GetIndexForImage(S,superPixels,excludeLabels)
    spInd = unique(superPixels);
    index.label=[];
    index.sp=[];
    index.spSize=[];
    for i = spInd(:)'
        mask = superPixels==i;
        numPix = sum(mask(:));
        [ls counts] = UniqueAndCounts(S(mask));
        ls(counts<numPix*.5) = [];
        counts(counts<numPix*.5) = [];
        counts(ls<1) = [];
        ls(ls<1) = [];
        for j = 1:length(excludeLabels)
            counts(ls==excludeLabels(j)) = [];
            ls(ls==excludeLabels(j)) = [];
        end
        if(~isempty(ls))
            [foo ind] = max(counts);
            index.label(end+1) = ls(ind);
            index.sp(end+1) = i;
            index.spSize(end+1) = sum(mask(:));
        end
    end
end
