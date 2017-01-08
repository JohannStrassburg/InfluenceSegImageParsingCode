function [retInds rank ranksort] = FindRetrievalSet(trainGlobalDesc,testGlobalDesc,DataDir,fileName,testParams,suffix)
if(~exist('suffix','var'));suffix = '';end
outfile = fullfile(DataDir,'RetrievalSet',suffix,[fileName '.mat']);
if(exist(outfile,'file'))
    %retInds = []; rank = []; ranksort = [];
    %return;
    load(outfile);
else
    fields = fieldnames(trainGlobalDesc);
    rank = [];
    for i = 1:length(fields)
        if(~iscell(trainGlobalDesc.(fields{i})))
            testData = testGlobalDesc.(fields{i});
            nPts = size(testData,1);
            dists = zeros(size(trainGlobalDesc.(fields{i}),1),nPts);
            for d = 1:100:nPts
                dists(:,d:min(d+99,nPts)) = dist2(trainGlobalDesc.(fields{i}),testData(d:min(d+99,nPts),:));
            end
            [dists ndx] = sort(dists);
            rank.(fields{i}) = ones(size(ndx));
            rank.(fields{i})(ndx) = 1:size(ndx);
        end
    end
    make_dir(outfile);save(outfile,'rank');
end

fields = testParams.globalDescriptors;
rankfinal = ones(length(fields),size(rank.(fields{1}),1));
for i = 1:length(fields)
    rankfinal(i,:) = rank.(fields{i});
end

rank2 = min(rankfinal,[],1);
[ranksort retInds] = sort(rank2);