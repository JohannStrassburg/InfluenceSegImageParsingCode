function [clusters, clusters2, clusters3, clusterkmeans, clusterStats, extendedClust] = ClusterRetrievalSet(labelHist,clusterSize,clusterCutoff)%DStrain,imageNNs,clusterSize,shortLabNum,linkageType,type,fullDesc)

%DS = DStrain(imageNNs);

if(~exist('linkageType','var'))
    linkageType = 'average';
end
if(~exist('clusterSize','var'))
    clusterSize = 5;
end
if(~exist('clusterCutoff','var'))
    clusterCutoff = .7;
end
Y = pdist(labelHist,'cityblock');
Z = linkage(Y,linkageType);
%figure(2);dendrogram(Z,200,'colorthreshold',clusterCutoff*(max(Z(:,3))));
clusters = cluster(Z,'maxclust',clusterSize);
clusters = resortCluster(clusters);
%figure(3);hist(clusters,clusterSize);
%figure(4);scatter3(labelHist(:,1),labelHist(:,2),labelHist(:,3),3,clusters);
clusters2 = cluster(Z,'cutoff',clusterCutoff*(max(Z(:,3))),'criterion','distance'); 
clusters2 = resortCluster(clusters2);
%figure(5);hist(clusters2,length(unique(clusters2))); 
%figure(6);scatter3(labelHist(:,1),labelHist(:,2),labelHist(:,3),3,clusters2);
clusters3 = cluster(Z,'cutoff',clusterCutoff,'criterion','distance'); 
clusters3 = resortCluster(clusters3);
%figure(7);hist(clusters3,length(unique(clusters2))); 
%2);


if(nargout>3)
    cns = unique(clusters);
    dist = squareform(Y);
    for i = cns(:)'
        dt = dist(clusters==i,:);
        dt = dt(:,clusters==i);
        dt = dt(:);
        clusterStats.maxDist(i) = max(dt);
        clusterStats.meanDist(i) = mean(dt);
        clusterStats.medianDist(i) = median(dt);
    end
end

if(exist('fullDesc','var'))
    cns = unique(clusters);
    for i = cns(:)'
        cDesc = labelHist(:,clusters==i);
        meanDesc = mean(cDesc,2);
        descTemp = fullDesc;
        descTemp(:,imageNNs) = [];
        clustersDists = dist2(meanDesc',descTemp');
        [clustersDists ndx] = sort(clustersDists,'ascend');
        map = 1:length(DStrain);
        map(imageNNs) = [];
        extendedClust{i} = map(ndx);
    end
end
end

function [clusterout] = resortCluster(cluster)
    [inds counts] = UniqueAndCounts(cluster);
    clusterout = cluster;
    [counts inds] = sort(counts,'descend');
    for i = 1:length(inds)
        clusterout(cluster==inds(i)) = i;
    end
end
    

function [clusters] = CompressClusters(clusters)
    clusters(max(clusters,[],2)==0,:)=[];
end

function [clusters] = MoveCluster(clusters,startClust,endClust)
    if(startClust~=endClust)
        rowS = find(clusters(startClust,:)~=0);
        rowE = find(clusters(endClust,:)==0,length(rowS));
        clusters(endClust,rowE) = clusters(startClust,rowS);
        clusters(startClust,rowS) = 0;
    end
end
function [ind] = FindCluster(clusters,ind)
    [ind, foo] = find(clusters==ind);
end