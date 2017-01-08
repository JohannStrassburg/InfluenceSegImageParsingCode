function [rawNNs totalTime totalMatches] = DoRNNSearch(imDesc,descTrain,HOMEDATA,baseFName,suffix,testParams,Kndx,canSkip)

if(~exist('canSkip','var'))
    canSkip = 1;
end

Rs = testParams.Rs{Kndx};
targetNN = testParams.targetNN;
descFuns = testParams.segmentDescriptors;

totalTime = 0;

outfilename = fullfile(HOMEDATA,['rNNSearch' suffix],[baseFName '.mat']);

%{-
if(exist(outfilename,'file')&&canSkip)
    load(outfilename);
else
%}
    if(isempty(descTrain))
        rawNNs=[]; totalTime=[]; totalMatches=[];
        return;
    end
    totalMatches = 0;
    startTime = clock;
    for i = 1:length(descFuns)
        searchDesc = descTrain.(descFuns{i});
        queryDesc = imDesc.(descFuns{i});
        R = Rs.(descFuns{i}).Rs(Rs.(descFuns{i}).numNNs==targetNN);
        %R = max(Rs.(descFuns{i}).Rs);
        numQueries = size(queryDesc,1);
        distAll = zeros(numQueries,size(searchDesc,1));
        for j = 1:ceil(numQueries/100)
            finx = 1+(j-1)*100;
            sinx = min(numQueries,j*100);
            distAll(finx:sinx,:) = sqrt(dist2(double(queryDesc(finx:sinx,:)),double(searchDesc)));
        end
        for j = 1:numQueries
            %distind = lpnorm(queryDesc(:,j),searchDesc,2);
            nns = find(distAll(j,:)<=R);%;distind%
            dist = distAll(j,nns);%distind(nns);%
            [rawNNs(j).(descFuns{i}).dists ind] = sort(dist);
            rawNNs(j).(descFuns{i}).nns = nns(ind);
            totalMatches = totalMatches + length(rawNNs(j).(descFuns{i}).nns);
        end
    end
    totalTime = totalTime+etime(clock,startTime);
    if(canSkip)
        make_dir(outfilename);save(outfilename,'rawNNs');
    end
end


end
