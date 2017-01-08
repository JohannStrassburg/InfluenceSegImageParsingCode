function [descProb, descVotes] = GetProbPerLabelPerDescSimp(spIndex,nns,filteredLabels,filtLabCounts,type,smoothingConst)
    
    sc = smoothingConst; %smoothing constant
    
    descVotes = zeros(length(filtLabCounts),1);

    [labelNum votes] = UniqueAndCounts(spIndex.label(nns));
    [labelNum ind filtind] = intersect(labelNum,filteredLabels);
    votes = votes(ind);
    descVotes(filtind) = votes;
    %for ratio
    if(strfind(type,'ratio')==1)
        tv = sum(votes);
        tc = sum(filtLabCounts);
        descProb = ((sc+descVotes)./(sc+tv-descVotes)) .*((tc./filtLabCounts(:)));
        if(isempty(strfind(type,'ratio-raw')==1))
            descProb(descVotes==0) = sc/(tv+sc);
        end
    else
    	descProb = (votes+sc)/filtLabCounts(:);
    end
    
%end


