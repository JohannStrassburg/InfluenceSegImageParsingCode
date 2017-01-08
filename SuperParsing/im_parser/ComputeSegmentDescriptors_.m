function [descFuns timing] = ComputeSegmentDescriptors(fileList, HOMEIMAGES, HOMEDATA, HOMECENTERS, HOMECODE, canSkip, range, K, segSuffix, descFuns)
%GENERATESUPPIXDESC Summary of this function goes here
%   Detailed explanation goes here
% spIndex:
%  spIndex.DBImageIndex: Image in the db corresponding to the SP
%  spIndex.DBSegmentIndexList: Cell of segments with OS > .5 with SP
%  spIndex.ImageSPIndex: SP index in the image
%  spIndex.SegmentCenterList: Cell of vetors from center of SP to center of the segment
%  spIndex.SPSize: height and width in pixels of the super pixel
close all;
funDir = 'SegmentDescriptors';
centerFuns = GetFunList(fullfile(HOMECODE,funDir,'calculate_centers'));
filterFuns = GetFunList(fullfile(HOMECODE,funDir,'im_filters'));
if(~exist('descFuns','var'))
    descFuns = GetFunList(fullfile(HOMECODE,funDir));
end
if(~exist('segSuffix','var'))
    segSuffix = '';
end
useBusyFile = false;
HOMECENTERS = fullfile(HOMECENTERS,'Descriptors','Global');
poolstarted = 0;
% get the centers for histogram based descriptors
centers = [];
for k = 1:length(centerFuns)
    outFileName = fullfile(HOMECENTERS,'Dictionaries',sprintf('%s.mat',centerFuns{k}));
    if(exist(outFileName,'file'))
        load(outFileName);
    else
        dic = feval(centerFuns{k}, fileList, HOMEIMAGES, 100);
        make_dir(outFileName);
        save(outFileName,'dic');
    end
    centers.(centerFuns{k}) = dic;
end

if(~exist('K','var'))
    K=200;
end

HOMEDATA = fullfile(HOMEDATA,'Descriptors',sprintf('SP_Desc_k%d%s',K,segSuffix));

if(~exist('range','var'))
    range = 1:length(fileList);
end

tmOff = length(descFuns);
count = 0;

pfig = ProgressBar('Pre-computing Segment Descriptors');
totalSuperPixels=0;
for i = range
%try
    %{-
    if(isempty(fileList{i}))
        continue;
    end
    [dirN base] = fileparts(fileList{i});
    baseFName = fullfile(dirN,base);
    
    descMask = zeros(size(descFuns))==1;
    descInds = 1:length(descFuns);
    for k = descInds
        descFun = descFuns{k};
        outFileName = fullfile(HOMEDATA,descFun,sprintf('%s.mat',baseFName));
        if(exist(outFileName,'file') && canSkip)
            continue;
        end
        descMask(k) = 1;
    end
    descInds = descInds(descMask);
    descs= cell(size(descFuns));
    needTextons = false;
    for filtCount = 1:length(filterFuns)
    	outTxtonName = fullfile(HOMECENTERS,'Textons',filterFuns{filtCount},sprintf('%s.mat',baseFName));
        if(~exist(outTxtonName,'file'))
            needTextons = true;
            break;
        end
    end
    textons = [];
    adjFile = fullfile(HOMEDATA,'sp_adjacency',sprintf('%s.mat',baseFName));
    
    %for each superpixel compute descriptor
    if(~isempty(descInds) || ~exist(adjFile,'file') || needTextons)
        filename = fullfile(HOMEIMAGES,fileList{i});
        im = imread(filename);
        [row col ch] = size(im);
        count = count+1;
        %force it to have 3 channels
        if(ch==1)
            im = repmat(im,[1 1 3]);
        end
    
        %mark file as busy so other threads will skip over it
        busyFile = fullfile(HOMEDATA,'busyFile',sprintf('%s.mat',baseFName));
        if(exist(busyFile,'file'))
            continue;
        end
        if(useBusyFile)
            make_dir(busyFile);save(busyFile,'busyFile');
        end
    
        st = toc;
        % Get Super Pixels for image
        outSPName = fullfile(HOMEDATA,'super_pixels',sprintf('%s.mat',baseFName));
        if(exist(outSPName,'file'))
            load(outSPName);
        else
            if(strcmp(segSuffix,'_vidSeg'))
                if(exist(busyFile,'file')) delete(busyFile); end
                fprintf('ERROR: No Superpixel file for: %s\n',fullfile(dirN,base));
                continue;
            end
            %tic
            if(K>0)
                superPixels = GenerateSuperPixels(im,K);
            else
                superPixels = GenerateGroundTruthSegs(fullfile(HOMEIMAGES,'..','LabelsSemantic',[baseFName '.mat']));
            end
            make_dir(outSPName);save(outSPName,'superPixels');
        end
        superPixInd = unique(superPixels);
        totalSuperPixels = totalSuperPixels + length(superPixInd);
        
        %find the adjacency graph between superpixels
        if(~exist(adjFile,'file'))
            adjPairs = FindSPAdjacnecy(superPixels);
            make_dir(adjFile);save(adjFile,'adjPairs');
        end
        % Get textons for Image
        
        
    end
%}
%{
catch ERR
    %ERR
    %ERR.message
    %for j = 1:length(ERR.stack)
    %    ERR.stack(j)
    %end
    if(exist('busyFile','var'))
        if(exist(busyFile,'file'))
            delete(busyFile);
        end
    end
    close(pfig);
    throw(ERR);
end
%}
if(mod(i,100)==0)
	ProgressBar(pfig,find(i==range),length(range));
end
%fprintf('Finished file: %d of %d %.2f%% done Time: %.2f/%.2f\n',find(i==range),length(range),100*find(i==range)/length(range),toc/60,toc*length(range)/find(i==range)/60);
    
end
close(pfig);
if(poolstarted)
    matlabpool close;
end

function [ funList ] = GetFunList( funDir )
fileList = dir(fullfile(funDir,'*.m'));
funList = cell(length(fileList),1);
for i = 1:length(fileList)
    [foo funList{i}] = fileparts(fileList(i).name);
end
