function [labels votes] = GetLabelVotes(D,index,nns,filteredLabels,listFun)
% segIndex:
%  segIndex.DBImageIndex: Image in the db corresponding to the Segment
%  segIndex.DBSegmentIndex: Segment Index of image
%  the segment
% spIndex:
%  spIndex.DBImageIndex: Image in the db corresponding to the SP
%  spIndex.DBSegmentIndexList: Cell of segments with OS > .5 with SP
%  spIndex.ImageSPIndex: SP index in the image
%  spIndex.SegmentCenterList: Cell of vetors from center of SP to center of
%  the segment
labels  = [];
if(~exist('listFun','var'))
    listFun = @GetLabelListNum;
end

for i = 1:length(nns)
    if(iscell(filteredLabels))
        if(isfield(index,'DBSegmentIndexList'))
            ll = GetLabelList(D, index.DBImageIndex(nns(i)), index.DBSegmentIndexList{nns(i)});
        elseif(isfield(index,'DBSegmentIndex'))
            ll = GetLabelList(D, index.DBImageIndex(nns(i)), index.DBSegmentIndex(nns(i)));
        end
        for j = 1:length(ll)
            mask = strcmp(ll{j},filteredLabels);
            if(sum(mask)>0)
                labels(length(labels)+1) = find(mask,1);
            end
        end
    else
        if(isfield(index,'DBSegmentIndexList'))
            ll = listFun(D, index.DBImageIndex(nns(i)), index.DBSegmentIndexList{nns(i)});
        elseif(isfield(index,'DBSegmentIndex'))
            ll = listFun(D, index.DBImageIndex(nns(i)), index.DBSegmentIndex(nns(i)));
        end
        [foo labelInd] = intersect(filteredLabels,ll);
        labels = [labels labelInd];
    end
end
[labels, votes] = UniqueAndCounts(labels);
end
