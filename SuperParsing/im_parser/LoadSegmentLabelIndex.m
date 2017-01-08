function [segmentLabelIndex, labels, counts] = LoadSegmentLabelIndex(fileList,excludeLabels,HOMELABELSET,HOMEDESC,segFolderName)

HOMESUPERPIX = fullfile(HOMEDESC,segFolderName,'super_pixels');
segFolderName = [segFolderName];
saveFile = fullfile(HOMELABELSET,segFolderName,'segIndex.mat');
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


segmentLabelIndex.label = [];
segmentLabelIndex.sp = [];
segmentLabelIndex.spSize = [];
segmentLabelIndex.image = [];


[fold name] = fileparts(fileList{1});
segFile = fullfile(HOMELABELSET,fold,[name '.mat']);
load(segFile);
labels=names;
excludeMask = zeros(size(labels))==1;
for j = 1:length(labels)
    excludeMask(j) = sum(strcmp(excludeLabels,labels{j}))>0;
end
excludeLabels = find(excludeMask);

pfig = ProgressBar;
for i= 1:length(fileList)
    
    [fold name] = fileparts(fileList{i});
    segFile = fullfile(HOMELABELSET,fold,[name '.mat']);
    load(segFile);
    spFile = fullfile(HOMESUPERPIX,fold,[name '.mat']);
    load(spFile);
    indexFile = fullfile(HOMELABELSET,segFolderName,fold,[name '.mat']);
    index = [];
    if(exist(indexFile,'file'))
        load(indexFile);
        if(mod(i,100)==0)
            ProgressBar(pfig,i,length(fileList));
        end
    end
    if(~isfield(index,'spSize'))
        index = GetIndexForImage(S,superPixels,excludeLabels);
        make_dir(indexFile);save(indexFile,'index');
        ProgressBar(pfig,i,length(fileList));
    end
    
    segmentLabelIndex.label = [segmentLabelIndex.label index.label];
    segmentLabelIndex.sp = [segmentLabelIndex.sp index.sp];
    segmentLabelIndex.image = [segmentLabelIndex.image ones(size(index.sp))*i];
    segmentLabelIndex.spSize = [segmentLabelIndex.spSize index.spSize];
    
end
close(pfig);
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
