x2thresh = 0;% 15;
clusterSizeThresh = 10;
descList = testParams.SVMDescs;
tprThresh = 0;
%ittResult = cell(max(UseGlobalSVM),1);
for ls = UseGlobalSVM
    [fo labelTypeName] = fileparts(HOMELABELSETS{ls});
    
    numLabels = length(Labels{ls});
    numTrainIms = length(trainFileList);
    numTestIms = length(testFileList);
    numValIms = length(valFileList);
    
    trainLabels = -1*ones(numTrainIms,numLabels);
    for l = 1:numLabels; trainLabels(trainIndex{ls}.image(trainIndex{ls}.label==l),l) = 1; end
    testLabels = -1*ones(numTestIms,numLabels);
    for l = 1:numLabels; testLabels(testIndex{ls}.image(testIndex{ls}.label==l),l) = 1; end
    valLabels = -1*ones(numValIms,numLabels);
    for l = 1:numLabels; valLabels(valIndex{ls}.image(valIndex{ls}.label==l),l) = 1; end
        
    tl = double(trainLabels==1);ntl = double(trainLabels==-1);
    a = tl'*tl;
    b = tl'*ntl;
    c = ntl'*tl;
    d = ntl'*ntl;
    x2 = b./(b+a) + c./(c+a);%(((b.*c-a.*d))./(a+b+c+d));%./((a+b).*(c+d).*(b+d).*(a+c));
    maxx2 = max(x2(:));
    x2 = maxx2-x2;
    x2 = x2 - diag(diag(x2));
    %x2s = sort(x2(:),'descend');
    
    D = squareform(x2);
    Z = linkage(D,'complete');
    figure(1);dendrogram(Z,'colorthreshold',maxx2);
    
    pfig = ProgressBar('Cluster Itt');
    ittResult{ls} = [];
    curCluster = zeros(numLabels);curCluster(1:end,1) = 1:numLabels;
    ittResult{ls}(1).cluster = curCluster;
    [ittResult{ls}(1).svm  ittResult{ls}(1).tpr ittResult{ls}(1).fpr] = TrainGlobalSVMBasic(trainGlobalDesc, trainLabels, valGlobalDesc, valLabels, testGlobalDesc, testLabels,  descList, tprThresh);
    ittResult{ls}(1).combPattern = cell(size(curCluster,1),1);

    totalCluster = curCluster;
    for i = 2:18%numLabels-1
        ProgressBar(pfig,i-1,numLabels);
        %{
        if(x2s(1)<x2thresh) break; end
        [r c] = find(x2==x2s(1));x2s(1:2) = [];
        [r1 c1] = find(curCluster==r(2));
        [r2 c2] = find(curCluster==c(2));
        if(r1==r2) ittResult{ls}(i)=ittResult{ls}(i-1);continue; end
        [c1] = find(curCluster(r1,:) == 0,1);
        [c2] = find(curCluster(r2,:) == 0,1);
        curCluster(r1,c1:c1+c2-2) = curCluster(r2,1:c2-1);
        curCluster(r2,:) = [];
        [r c] = find(curCluster>0);
        if(max(c)>clusterSizeThresh) break; end
        %}
         
        if(Z(i-1,3)>maxx2) break; end
        [c1] = find(totalCluster(Z(i-1,1),:) == 0,1);
        [c2] = find(totalCluster(Z(i-1,2),:) == 0,1);
        totalCluster(numLabels+i-1,:) = totalCluster(Z(i-1,1),:);
        totalCluster(numLabels+i-1,c1:c1+c2-2) = totalCluster(Z(i-1,2),1:c2-1);
        totalCluster(Z(i-1,1),:) = 0;totalCluster(Z(i-1,2),:) = 0;
        curCluster = totalCluster;
        curCluster(sum(curCluster,2)==0,:) = [];
        
        numClusters = size(curCluster,1);
        ittResult{ls}(i).cluster = curCluster;
        ittResult{ls}(i).tpr = zeros(numClusters,3);
        ittResult{ls}(i).fpr = zeros(numClusters,3);
        ittResult{ls}(i).svm = cell(numClusters,1);
        ittResult{ls}(i).combPattern = cell(numClusters,1);
        for c = 1:numClusters
            prevC =find(sum(ittResult{ls}(i-1).cluster==repmat(curCluster(c,:),[size(ittResult{ls}(i-1).cluster,1) 1]),2)==size(curCluster,2));
            if(isempty(prevC))
                csize = sum(curCluster(c,:)>0);
                clind = curCluster(c,:); clind(clind==0) = [];
                combPattern = unique(combnk(repmat([-1 1],1,csize),csize),'rows');
                newtrainLabels = -1*ones(size(trainLabels,1),size(combPattern,1));
                newvalLabels = -1*ones(size(valLabels,1),size(combPattern,1));
                newtestLabels = -1*ones(size(testLabels,1),size(combPattern,1));
                noTraining = [];
                for newl = 1:size(combPattern,1)
                    newtrainLabels(sum(trainLabels(:,clind)==repmat(combPattern(newl,:),[size(newtrainLabels,1) 1]),2)==csize,newl) = 1;
                    if(all(newtrainLabels(:,newl) == newtrainLabels(1,newl)))
                        noTraining = [noTraining newl];
                    end
                    newvalLabels(sum(valLabels(:,clind)==repmat(combPattern(newl,:),[size(newvalLabels,1) 1]),2)==csize,newl) = 1;
                    newtestLabels(sum(testLabels(:,clind)==repmat(combPattern(newl,:),[size(newtestLabels,1) 1]),2)==csize,newl) = 1;
                end
                newtrainLabels(:,noTraining) = [];newvalLabels(:,noTraining) = [];newtestLabels(:,noTraining) = [];
                combPattern(noTraining,:) = [];
                %newvalLabels = [];
                for d = 1:length(descList)
                    [ittResult{ls}(i).svm{c}{d} tprt fprt] = TrainGlobalSVMBasic(trainGlobalDesc, newtrainLabels, valGlobalDesc, newvalLabels, testGlobalDesc, newtestLabels,  descList(d), tprThresh);
                end
                [a b] = find(isnan(tprt)); a = unique(a);
                tprt(a,:) = []; fprt(a,:) = [];
                ittResult{ls}(i).tpr(c,:) = mean(tprt);
                ittResult{ls}(i).fpr(c,:) = mean(fprt);
                ittResult{ls}(i).combPattern{c} = combPattern;
            else
                ittResult{ls}(i).svm{c} = ittResult{ls}(i-1).svm{prevC};
                ittResult{ls}(i).combPattern{c} = ittResult{ls}(i-1).combPattern{prevC};
                ittResult{ls}(i).tpr(c,:) = ittResult{ls}(i-1).tpr(prevC,:);
                ittResult{ls}(i).fpr(c,:) = ittResult{ls}(i-1).fpr(prevC,:);
            end
        end
    end
    close(pfig);
end
