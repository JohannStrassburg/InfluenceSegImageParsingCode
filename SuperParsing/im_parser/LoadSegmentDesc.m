function [segDescs, superPixels, unLabeledSPDesc] = LoadSegmentDesc(fileList,index,HOMEDATA,descFuns,K,segSuffix)


HOMEDATA = fullfile(HOMEDATA,'Descriptors',sprintf('SP_Desc_k%d%s',K,segSuffix));
if(~isempty(index))
    range = unique(index.image);
else
    range = 1:length(fileList);
end

if(~exist('segSuffix','var'))
    segSuffix = '';
end

segDescs = [];
unLabeledSPDesc = [];
for k = 1:length(descFuns)
    segDescs.(descFuns{k}) = [];
    unLabeledSPDesc.(descFuns{k}) = [];
end

minPBar = 400;

if(length(range)>minPBar)
    pfig = ProgressBar('Loading Segment Descriptors');
end
current = 1;
currentUn = 1;
for i = range(:)'
    [dirN base] = fileparts(fileList{i});
    baseFName = fullfile(dirN,base);
    
    if(~isempty(index))
        numSP = sum(index.image==i);
        load(fullfile(HOMEDATA,'super_pixels',dirN,[base '.mat']));
        uSP = unique(superPixels);
        spMap = zeros(max(uSP),1);
        spMap(uSP) = 1:length(uSP);
    end
    allDescFile = fullfile(HOMEDATA,'AllDesc',sprintf('%s.mat',baseFName));
    allDesc = [];
    if(exist(allDescFile,'file'))
        try
            load(allDescFile);
        catch
            allDesc = [];
        end
    end
    covered = zeros(size(descFuns))==1;
    for k = 1:length(descFuns)
        descFun = descFuns{k};
        outFileName = fullfile(HOMEDATA,descFun,sprintf('%s.mat',baseFName));
        if(~isfield(allDesc,descFun))
            load(outFileName);
            allDesc.(descFun) = desc;
        else
            desc = allDesc.(descFun);
        	covered(k) = true;
        end
        
        if(~isempty(index))
            if(isempty(segDescs.(descFun)))
                segDescs.(descFun) = zeros(length(index.image),size(desc,1));
            end
            %display(outFileName);
            %display(i);
            segDescs.(descFun)(current:current+numSP-1,:) = desc(:,spMap(index.sp(index.image==i)))';
            if(nargout>2)
                unMask = ones(size(desc,2),1)==1;
                unMask(index.sp(index.image==i)) = false;
                unLabeledSPDesc.(descFun)(currentUn:currentUn+sum(unMask)-1,:) = desc(:,unMask)';
            end
        else
            segDescs.(descFun) = [segDescs.(descFun); desc'];
        end
    end
    if(any(~covered))
        make_dir(allDescFile);
        for r = 1:100
            try
                save(allDescFile,'allDesc');
            catch
                pause(1);
                continue;
            end
            break;
        end
                
    end
    if(~isempty(index))
        current = current +numSP;
        if(nargout>2)
            currentUn = currentUn +sum(unMask);
        end
    end
    
    if(length(range)>minPBar && mod(find(range==i),10)==0)
        ProgressBar(pfig,find(range==i),length(range));
    end
end
if(length(range)>minPBar)
    close(pfig);drawnow();
end

if(nargout>1)
    outSPName = fullfile(HOMEDATA,'super_pixels',sprintf('%s.mat',baseFName));
    load(outSPName);
end
