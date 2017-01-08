function [probPerLabel,totalTime,probPerDescPerLabel] = GetAllProbPerLabel(HOMEDATA,baseFName,suffix,index,rawNNs,filteredLabels,filtLabCounts,type,smoothingConst,canSkip)

if(~exist('canSkip','var'))
    canSkip = 1;
end
if(type==0)
    type='sum';
elseif(type==1)
    type='ratio';
end

totalTime = 0;

curIndexLength = length(index.image);

if(strcmp(type,'ratio'))
    outfilename = fullfile(HOMEDATA,['probPerLabel' suffix],[baseFName '.mat']);
else
    outfilename = fullfile(HOMEDATA,['probPerLabel' suffix type],[baseFName '.mat']);
end
if(exist(outfilename,'file')&&canSkip)
    load(outfilename);
    if(exist('indexLength','var') && indexLength == curIndexLength) %#ok<NODEF>
        if(exist('probPerDescPerLabel','var'))
            if(size(probPerLabel,2)==length(filteredLabels))
                return;
            end
        end
    end
end

if(isempty(rawNNs))
    probPerLabel = [];
    probPerDescPerLabel=[];
    return;
end

startTime = clock;
probPerLabel = zeros(length(rawNNs),length(filteredLabels));
probPerDescPerLabel = zeros(length(rawNNs),length(filteredLabels),length(fieldnames(rawNNs(1))));

    
for i = 1:length(rawNNs)
	[probPerLabel(i,:) probPerDescPerLabel(i,:,:)] = GetProbPerLabel(index,rawNNs(i),filteredLabels,filtLabCounts,type,smoothingConst);
end
totalTime = totalTime+etime(clock,startTime);
indexLength = curIndexLength;
if(canSkip)
    make_dir(outfilename);
    save(outfilename,'probPerLabel','probPerDescPerLabel','indexLength');
end
end