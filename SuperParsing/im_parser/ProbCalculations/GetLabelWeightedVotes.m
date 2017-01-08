function [labels votes] = GetLabelWeightedVotes(D,index,nns,weights,filteredLabels)
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
nWeights = [];
for i = 1:length(nns)
    if(isfield(index,'DBSegmentIndexList'))
        ll = GetLabelList(D, index.DBImageIndex(nns(i)), index.DBSegmentIndexList{nns(i)});
    elseif(isfield(index,'DBSegmentIndex'))
        ll = GetLabelList(D, index.DBImageIndex(nns(i)), index.DBSegmentIndex(nns(i)));
    end        
    for j = 1:length(ll)
        mask = strcmp(ll{j},filteredLabels);
        if(sum(mask)>0)
            labels(length(labels)+1) = find(mask,1);
            nWeights(length(nWeights)+1) = weights(i);
        end
    end
end
[labels, i, descriptionndx] = unique(labels);
votes = zeros(length(labels),1);
for i = 1:length(labels)
    votes(i) = sum(nWeights(descriptionndx==i));
end
end
