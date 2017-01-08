function [results bestR svmR svmPick bestPick resultsExp bestRExp expRExp bestPickExp expPickExp] = ParseTestImagesRetCluster(HOMEDATA,HOMETESTDATA,HOMEIMAGES,HOMELABELSETS,testFileList,trainFileList,trainGlobalDesc,testGlobalDesc,trainIndex,testIndex,trainCounts,labelPenality,Labels,testParams,fullSPDesc)

try for i = 3:100;fclose(i);fprintf('closed %d\n',i);end;catch ERR;end
%close all;
pfig = ProgressBar('Parsing Images');
pfig2=ProgressBar('Classifing Expanded Clusters');

DataDir = fullfile(HOMETESTDATA,testParams.TestString);
if(~exist('fullSPDesc','var'))
    fullSPDesc = cell(length(HOMELABELSETS),length(testParams.K));
end

doWeb = 1;
%WEB SETUP
if(doWeb)
    if(~exist('labelColors','var'))
        labelColors = cell(size(Labels));
        for k = 1:length(Labels)
            temp = jet(length(Labels{k}));
            labelColors{k} = [temp(length(Labels{k}):-1:1,:); [0 0 0]];
        end
    end
    HOMEWEB = fullfile(DataDir,'ClusterWebsite');
    webIndexFile = fullfile(HOMEWEB,'index.htm');
    make_dir(webIndexFile);
    indexFID = fopen(webIndexFile,'w');
    fprintf(indexFID,'<HTML>\n<HEAD>\n<TITLE>Retrieval Set Clusters</TITLE>\n</HEAD>\n<BODY>');
    fprintf(indexFID,'\n<center>\n');
    fprintf(indexFID,'\n<table border="0">\n');
    fprintf(indexFID,'\t<tr>\n');
    numCols = 6;
    numColsClust = 20;
    colCount = 0;
    maxDim = 400;
end

range = 202:300;
for i = range
    [folder file ext] = fileparts(testFileList{i});
    baseFName = fullfile(folder,file);
    
    %Get Retrieval Set
    [retIndsAll rankAll] = FindRetrievalSet(trainGlobalDesc,SelectDesc(testGlobalDesc,i,1),DataDir,baseFName,testParams);
    
    if(doWeb)
        backOutString = '../';
        if(isempty(folder))
            folder = '.';
            backOutString = '';
        end
        localImageFile = fullfile(HOMEWEB,'Images',folder,[file ext]);make_dir(localImageFile);
        if(~exist(localImageFile,'file'))
            copyfile(fullfile(HOMEIMAGES,testFileList{i}),localImageFile);
        end
        fprintf(indexFID,'\t\t<td><center> <a href="%s">',['ImageWeb/' folder '/' file '.htm']);
        fprintf(indexFID,'<img  width="200" src="%s"></a> ',['Images/' folder '/' file ext]);
        fprintf(indexFID,'</center> </td>\n');
        colCount = colCount +1;
        if(colCount == numCols)
            colCount = 0;
            fprintf(indexFID,'\t</tr><tr>\n');
        end
        %{-
        imageWebPage = fullfile(HOMEWEB,'ImageWeb',folder,[file '.htm']);make_dir(imageWebPage);
        imFID = fopen(imageWebPage,'w');
        fprintf(imFID,'<HTML>\n<HEAD>\n<TITLE>%s %s</TITLE>\n</HEAD>\n<BODY>',folder,file);
        fprintf(imFID,'<img width="%d" src="%s"> ',maxDim,[backOutString '../Images/' folder '/' file ext]);% width="400"
        %}
    end
    
    %{-
    for retSetSize = testParams.retSetSize
        Kstr = '';
        clear imSP testImSPDesc;
        retInds = retIndsAll(1:min(end,retSetSize));
        if(doWeb && retSetSize == testParams.retSetSize(1))
            for imInd = retInds(:)'
                [retFolder retBase retExt] = fileparts(trainFileList{imInd});
                localImageFile = fullfile(HOMEWEB,'Images',retFolder,[retBase retExt]);make_dir(localImageFile);
                if(~exist(localImageFile,'file'))
                    copyfile(fullfile(HOMEIMAGES,trainFileList{imInd}),localImageFile);
                end
            end
        end
        for Kndx=1:length(testParams.K)
            [testImSPDesc imSP{Kndx}] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K(Kndx));
            if(length(testParams.K)==1)
                %imSP = imSP{Kndx};
                adjFile = fullfile(HOMETESTDATA,'Descriptors',sprintf('SP_Desc_k%d',testParams.K(Kndx)),'sp_adjacency',[baseFName '.mat']);
                load(adjFile);
            else
                %implement multi segmentation code
                adjPairs = FindSPAdjacnecy(imSP);
            end
            for labelType=2:length(HOMELABELSETS)
                [foo labelSet] = fileparts(HOMELABELSETS{labelType});
                suffix = sprintf('R%dK%d',retSetSize,testParams.K(Kndx));
                labelNums = 1:length(trainCounts{labelType});
                [retSetIndex descMask] = PruneIndex(trainIndex{labelType,Kndx},retInds);
                
                
                %find the clusters
                retlabhist = double(trainGlobalDesc.labelHist{labelType}(retInds,:));
                [fixed8  var06 thresh06] = ClusterRetrievalSet(retlabhist,8,.6);
                [fixed8  var07 thresh07] = ClusterRetrievalSet(retlabhist,8,.7);
                clusters = {var07,thresh06};%retSetClustersVarVerysmal,retSetClustersVarsmal,
                clusterNames = {'Var.7','Thresh.6'};
                if(doWeb && retSetSize == testParams.retSetSize(1))
                    for j = 1:length(clusters)
                        fprintf(imFID,'<p><a href="#%d">Cluster %s: %d clusters</a></p>',j,clusterNames{j},length(unique(clusters{j})));
                    end
                    %{
                    for j = 1:length(clusters)
                        cluster= clusters{j};
                        clusterInds = unique(cluster);
                        fprintf(imFID,'<a name="%d"></a><h2>Cluster %s: %d clusters</h2>',j,clusterNames{j},length(clusterInds));
                        for cNdx = clusterInds(:)'
                            fprintf(imFID,'<h3>Cluster %d: %d images</h3>',cNdx,sum(cluster==cNdx));
                            DisplayCluster(imFID,cluster,cNdx,[backOutString '../Images/'],trainFileList(retInds),numColsClust);
                        end
                    end
                    %}
                end
                
                if(~exist('results','var'))
                    results = cell([length(range) length(clusters)+1]);
                    bestR = zeros([length(range) length(clusters) 2]);
                    svmR = zeros([length(range) length(clusters) 2]);
                    bestPick = zeros([length(range) length(clusters)]);
                    svmPick = zeros([length(range) length(clusters)]);
                    resultsExp = cell([length(range) length(clusters)+1]);
                    bestRExp = zeros([length(range) length(clusters) 2]);
                    expRExp = zeros([length(range) length(clusters) 2]);
                    bestPickExp = zeros([length(range) length(clusters)]);
                    expPickExp = zeros([length(range) length(clusters)]);
                end
                
                %Find the labeling for just using the ret set
                %{-
                gtLabels = testIndex{labelType}.label(testIndex{labelType}.image==i)';
                spSizes = testIndex{labelType}.spSize(testIndex{labelType}.image==i)';
                probPerLabelBase = GetAllProbPerLabel(fullfile(DataDir,labelSet,'probPerLabel'),baseFName,suffix,retSetIndex,[],labelNums,trainCounts{labelType,Kndx},'ratio',1); %#ok<AGROW>
                if(isempty(probPerLabelBase))
                    if(isempty(fullSPDesc{labelType,Kndx}))
                        retSetSPDesc = LoadSegmentDesc(trainFileList,retSetIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(1));
                    else
                        retSetSPDesc = [];
                        for dNdx = 1:length(testParams.segmentDescriptors)
                            retSetSPDesc.(testParams.segmentDescriptors{dNdx}) = fullSPDesc{labelType,Kndx}.(testParams.segmentDescriptors{dNdx})(descMask,:);
                        end
                    end
                    rawNNs = DoRNNSearch(testImSPDesc,retSetSPDesc,fullfile(DataDir,labelSet,'RNN'),baseFName,suffix,testParams,Kndx,1);
                    probPerLabelBase = GetAllProbPerLabel(fullfile(DataDir,labelSet,'probPerLabel'),baseFName,suffix,retSetIndex,rawNNs,labelNums,trainCounts{labelType,Kndx},'ratio',1); %#ok<AGROW>
                end
                [foo lmax] = max(probPerLabelBase,[],2);
                L = lmax(imSP{Kndx});
                labelList = Labels{labelType};
                make_dir(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) 'NoClust'],baseFName));
                save(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) 'NoClust'],baseFName),'L','labelList');
                
                %{
                dataCost{1} = -(probPerLabelBase-max(probPerLabelBase(:))-1);
                for labelSmoothing = testParams.LabelSmoothing
                    testName = sprintf('%s S%d IS%d P%s IP%s',[myN2S(retSetSize) 'NoClust'],labelSmoothing,testParams.InterLabelSmoothing,...
                                testParams.LabelPenality{1}(1:3),testParams.InterLabelPenality{1}(1:3));
                    smoothingMatrix = BuildSmoothingMatrix(labelPenality(2,2),labelSmoothing,testParams.InterLabelSmoothing,testParams.LabelPenality{1},testParams.InterLabelPenality{1});
                    [L Lsp] = MultiLevelSegMRF(DataDir,HOMELABELSETS(2),testName,baseFName,Labels(2),imSP{Kndx},adjPairs,dataCost,smoothingMatrix);
                end
                %}
                    
                mask = zeros(size(lmax))==1;mask(testIndex{labelType}.sp(testIndex{labelType}.image==i))=1;
                correct = lmax(mask)==gtLabels;
                results{i,end} = [sum(correct.*spSizes) sum(spSizes)];
                if(doWeb && retSetSize == testParams.retSetSize(1))
                    fprintf(imFID,'<h3>Base Classification (No Clustering)</h3>');
                    gtLabelsFixed = zeros(size(lmax));
                    gtLabelsFixed(mask) = gtLabels;
                    gsL = gtLabelsFixed(imSP{Kndx});
                    gsL = gsL+1;
                    labelImOut = fullfile(HOMEWEB,'Cluster','GT',folder,[file '.png']);make_dir(labelImOut);
                    DrawImLabels(gsL,gsL,[0 0 0; labelColors{labelType}],{'Unlabeld' Labels{labelType}{:}},labelImOut,128,0,labelType,maxDim);
                    labelImOut = fullfile(HOMEWEB,'Cluster',[myN2S(retSetSize) 'NoCluster'],folder,[file '.png']);make_dir(labelImOut);
                    DrawImLabels(L,L,labelColors{labelType},Labels{labelType},labelImOut,128,0,labelType,maxDim);
                    fprintf(imFID,'\n<table border="0">\n');
                    fprintf(imFID,'\t<tr><td><center>Ground Truth<br>\n');
                    fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' 'GT/' folder '/' file '.png']);
                    fprintf(imFID,'\t%.2f%%\n',100);
                    fprintf(imFID,'\t</center></td><td><center>ML<br>\n');
                    fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' myN2S(retSetSize) 'NoCluster/' folder '/' file '.png']);
                    fprintf(imFID,'\t%.2f%%\n',100*results{i,end}(1)/results{i,end}(2));
                    fprintf(imFID,'\t</center></td></tr></table>');
                end
                
                %find the labeling for each cluster type
                for j = 1:length(clusters)
                    cluster= clusters{j};
                    clusterName = clusterNames{j};
                    clusterInds = unique(cluster);
                    %Do SVM Scheem
                    %{
                    results{i,j} = zeros(length(clusterInds),2);
                    probPerLabel = cell(length(clusterInds),1);
                    for cNdx = clusterInds(:)'
                        suffix = [myN2S(retSetSize) clusterName myN2S(cNdx)];
                        [clusterIndex descMask] = PruneIndex(retSetIndex,retInds(cluster==cNdx));
                        clusterSPDesc = [];
                        %{
                        for dNdx = 1:length(testParams.segmentDescriptors)
                            clusterSPDesc.(testParams.segmentDescriptors{dNdx}) = retSetSPDesc.(testParams.segmentDescriptors{dNdx})(descMask,:);
                        end
                        probPerLabel{cNdx} = GetAllProbPerLabel(fullfile(DataDir,labelSet,'probPerLabel'),baseFName,suffix,clusterIndex,[],labelNums,trainCounts{labelType,Kndx},'ratio',1); %#ok<AGROW>
                        if(isempty(probPerLabel{cNdx}))
                            rawNNs = DoRNNSearch(testImSPDesc,clusterSPDesc,fullfile(DataDir,labelSet,'RNN'),baseFName,suffix,testParams,Kndx,1);
                            probPerLabel{cNdx} = GetAllProbPerLabel(fullfile(DataDir,labelSet,'probPerLabel'),baseFName,suffix,clusterIndex,rawNNs,labelNums,trainCounts{labelType,Kndx},'ratio',1); %#ok<AGROW>
                        end
                        %}
                        probPerLabel{cNdx} = probPerLabelBase;
                        clusterLs = unique(clusterIndex.label);
                        mask = zeros(size(Labels{labelType}))==1;mask(clusterLs) = 1;
                        probPerLabel{cNdx}(:,mask') = min(probPerLabelBase(:))-1;
                        [foo lmax] = max(probPerLabel{cNdx},[],2);
                        mask = zeros(size(lmax))==1;mask(testIndex{labelType}.sp(testIndex{labelType}.image==i))=1;
                        correct = lmax(mask)==gtLabels;
                        results{i,j}(cNdx,:) = [sum(correct.*spSizes) sum(spSizes)];
                    end
                    bestR(i,j,:) = [max(results{i,j}(:,1)); results{i,j}(1,2)];
                    svmclusterNdx = SVMPickCluster(cluster,SelectDesc(trainGlobalDesc,retInds,1),SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,fullfile(DataDir,labelSet),baseFName,[myN2S(retSetSize) clusterName]);
                    [foo bestclNdx] = max(results{i,j}(:,1));
                    bestPick(i,j) = bestclNdx;
                    svmPick(i,j) = svmclusterNdx;
                    svmR(i,j,:) = [results{i,j}(svmclusterNdx,1); results{i,j}(1,2)];
                    [foo lmax] = max(probPerLabel{bestclNdx},[],2);
                    L = lmax(imSP{Kndx});
                    make_dir(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) clusterNames{j} ' SL Best'],baseFName));
                    save(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) clusterNames{j} ' SL Best'],baseFName),'L','labelList');
                    [foo lmax] = max(probPerLabel{svmclusterNdx},[],2);
                    L = lmax(imSP{Kndx});
                    make_dir(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) clusterNames{j} ' SL SVM'],baseFName));
                    save(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) clusterNames{j} ' SL SVM'],baseFName),'L','labelList');
                    
                    dataCost{1} = -(probPerLabel{svmclusterNdx}-max(probPerLabel{svmclusterNdx}(:))-1);
                    for labelSmoothing = testParams.LabelSmoothing
                        testName = sprintf('%s S%d IS%d P%s IP%s',[myN2S(retSetSize) clusterNames{j} ' SL SVM'],labelSmoothing,testParams.InterLabelSmoothing,...
                                    testParams.LabelPenality{1}(1:3),testParams.InterLabelPenality{1}(1:3));
                        smoothingMatrix = BuildSmoothingMatrix(labelPenality(2,2),labelSmoothing,testParams.InterLabelSmoothing,testParams.LabelPenality{1},testParams.InterLabelPenality{1});
                        [L Lsp] = MultiLevelSegMRF(DataDir,HOMELABELSETS(2),testName,baseFName,Labels(2),imSP{Kndx},adjPairs,dataCost,smoothingMatrix);
                    end
                    dataCost{1} = -(probPerLabel{bestclNdx}-max(probPerLabel{bestclNdx}(:))-1);
                    for labelSmoothing = testParams.LabelSmoothing
                        testName = sprintf('%s S%d IS%d P%s IP%s',[myN2S(retSetSize) clusterNames{j} ' SL Best'],labelSmoothing,testParams.InterLabelSmoothing,...
                                    testParams.LabelPenality{1}(1:3),testParams.InterLabelPenality{1}(1:3));
                        smoothingMatrix = BuildSmoothingMatrix(labelPenality(labelType,labelType),labelSmoothing,testParams.InterLabelSmoothing,testParams.LabelPenality{1},testParams.InterLabelPenality{1});
                        [L Lsp] = MultiLevelSegMRF(DataDir,HOMELABELSETS(labelType),testName,baseFName,Labels(labelType),imSP{Kndx},adjPairs,dataCost,smoothingMatrix);
                    end
                        
                        
                    if(doWeb && retSetSize == testParams.retSetSize(1))
                        fprintf(imFID,'<a name="%d"></a><h2>Cluster %s: %d clusters</h2>',j,clusterNames{j},length(unique(clusters{j})));
                        fprintf(imFID,'\n<table border="0">\n');
                        fprintf(imFID,'\t<tr>\n');
                        %best cluster 
                        fprintf(imFID,'\t\t<td><center> Best Cluster: %d<br>',bestclNdx);
                        fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' clusterName '/' folder '/' file myN2S(bestclNdx) '.png']);
                        fprintf(imFID,'\t%.2f%%\n ',100*results{i,j}(bestclNdx,1)/results{i,j}(bestclNdx,2));
                        fprintf(imFID,'</center> </td>\n');
                         %svm Cluster
                        fprintf(imFID,'\t\t<td><center> SVM Selected Cluster: %d<br>',svmclusterNdx);
                        fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' clusterName '/' folder '/' file myN2S(svmclusterNdx) '.png']);
                        fprintf(imFID,'\t%.2f%%\n ',100*results{i,j}(svmclusterNdx,1)/results{i,j}(svmclusterNdx,2));
                        fprintf(imFID,'</center> </td>\n');
                        fprintf(imFID,'\t</tr>\n</table>');
                        for cNdx = clusterInds(:)'
                            labelImOut = fullfile(HOMEWEB,'Cluster',clusterName,folder,[file myN2S(cNdx) '.png']);make_dir(labelImOut);
                            [foo lmax] = max(probPerLabel{cNdx},[],2);
                            L = lmax(imSP{Kndx});
                            if(~exist(labelImOut,'file'))
                                DrawImLabels(L,L,labelColors{labelType},Labels{labelType},labelImOut,128,0,labelType,maxDim);
                            end
                            fprintf(imFID,'<h3>Cluster %d: %d images</h3>',cNdx,sum(cluster==cNdx));
                            fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' clusterName '/' folder '/' file myN2S(cNdx) '.png']);
                            fprintf(imFID,'\t%.2f%%\n ',100*results{i,j}(cNdx,1)/results{i,j}(cNdx,2));
                            DisplayCluster(imFID,cluster,cNdx,[backOutString '../Images/'],trainFileList(retInds),numColsClust);
                        end
                    end
                    %}
                    
                    %Do expand and pick scheme
                    %{-
                    resultsExp{i,j} = zeros(length(clusterInds),2);
                    probPerLabelExp = cell(length(clusterInds),1);
                    clusterRank = zeros(length(clusterInds),1);
                    expandedInds = cell(size(clusterInds));
                    clusterCutOff = 0;
                    %Do SVM Scheem
                    ProgressBar(pfig2,0,length(clusterInds));
                    for cNdx = clusterInds(:)'
                        if(sum(cluster==cNdx)>=testParams.MinClusterSize)
                            clusterCutOff = cNdx;
                        end
                        suffix = ['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize) myN2S(cNdx)];
                        expandedInds{cNdx} = ExpandCluster(trainGlobalDesc.labelHist{labelType},retInds(cluster==cNdx),testParams.ExpansionSize);
                        clusterRank(cNdx) = sum(rankAll(expandedInds{cNdx}));
                        [clusterIndex descMask] = PruneIndex(trainIndex{labelType,Kndx},expandedInds{cNdx});
                        
                        probPerLabelExp{cNdx} = GetAllProbPerLabel(fullfile(DataDir,labelSet,'probPerLabel'),baseFName,suffix,clusterIndex,[],labelNums,trainCounts{labelType,Kndx},'ratio',1); %#ok<AGROW>
                        if(isempty(probPerLabelExp{cNdx}))
                            if(isempty(fullSPDesc{labelType,Kndx}))
                                clusterSPDesc = LoadSegmentDesc(trainFileList,clusterIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(1));
                            else
                                clusterSPDesc = [];
                                for dNdx = 1:length(testParams.segmentDescriptors)
                                    clusterSPDesc.(testParams.segmentDescriptors{dNdx}) = fullSPDesc{labelType,Kndx}.(testParams.segmentDescriptors{dNdx})(descMask,:);
                                end
                            end
                            rawNNs = DoRNNSearch(testImSPDesc,clusterSPDesc,fullfile(DataDir,labelSet,'RNN'),baseFName,suffix,testParams,Kndx,1);
                            probPerLabelExp{cNdx} = GetAllProbPerLabel(fullfile(DataDir,labelSet,'probPerLabel'),baseFName,suffix,clusterIndex,rawNNs,labelNums,trainCounts{labelType,Kndx},'ratio',1); %#ok<AGROW>
                        end
                        %force a shortlist for when there are no matches
                        slinds = unique(clusterIndex.label);
                        slmask = zeros(size(probPerLabelExp{cNdx},2),1)==1;slmask(slinds) = 1;
                        probPerLabelExp{cNdx}(:,~slmask) = min(probPerLabelExp{cNdx}(:))-1;
                        [foo lmax] = max(probPerLabelExp{cNdx},[],2);
                        mask = zeros(size(lmax))==1;mask(testIndex{labelType}.sp(testIndex{labelType}.image==i))=1;
                        correct = lmax(mask)==gtLabels;
                        resultsExp{i,j}(cNdx,:) = [sum(correct.*spSizes) sum(spSizes)];
                        fprintf('%.2f ',sum(correct.*spSizes)./sum(spSizes));
                        ProgressBar(pfig2,cNdx,length(clusterInds));
                    end
                    [svmclusterNdx clusterSVMScore] = SVMPickCluster(expandedInds,trainGlobalDesc,SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,fullfile(DataDir,labelSet),baseFName,['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize)],100);
                    svmPick(i,j) = svmclusterNdx;
                    svmR(i,j,:) = [resultsExp{i,j}(svmclusterNdx,1); resultsExp{i,j}(1,2)];
                    [foo lmax] = max(probPerLabelExp{svmclusterNdx},[],2);
                    L = lmax(imSP{Kndx});
                    saveFile = fullfile(DataDir,labelSet,'ClusterLabeling',['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize) ' SVM'],baseFName);make_dir(saveFile);
                    save(saveFile,'L','labelList');
                    
                    bestRExp(i,j,:) = [max(resultsExp{i,j}(:,1)); resultsExp{i,j}(1,2)];
                    [foo bestclNdx] = max(resultsExp{i,j}(:,1));
                    bestPickExp(i,j) = bestclNdx;
                    [foo expClusterPick] = min(clusterRank);
                    expPickExp(i,j) = expClusterPick;
                    expRExp(i,j,:) = [resultsExp{i,j}(expClusterPick,1); resultsExp{i,j}(1,2)];
                    [foo lmax] = max(probPerLabelExp{bestclNdx},[],2);
                    L = lmax(imSP{Kndx});
                    saveFile = fullfile(DataDir,labelSet,'ClusterLabeling',['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize) ' Best'],baseFName);make_dir(saveFile);
                    save(saveFile,'L','labelList');
                    [foo lmax] = max(probPerLabelExp{expClusterPick},[],2);
                    L = lmax(imSP{Kndx});
                    saveFile = fullfile(DataDir,labelSet,'ClusterLabeling',['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize) ' Rank'],baseFName);make_dir(saveFile);
                    save(saveFile,'L','labelList');
                    
                    [~, svmclusterNdxCutOff] = max(clusterSVMScore(1:clusterCutOff));
                    [~, lmax] = max(probPerLabelExp{svmclusterNdxCutOff},[],2);
                    L = lmax(imSP{Kndx});
                    saveFile = fullfile(DataDir,labelSet,'ClusterLabeling',['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize) myN2S(testParams.MinClusterSize) 'Cut SVM'],baseFName);make_dir(saveFile);
                    save(saveFile,'L','labelList');
                    [~, bestclNdxCutOff] = max(resultsExp{i,j}(1:clusterCutOff,1));
                    [~, lmax] = max(probPerLabelExp{bestclNdxCutOff},[],2);
                    L = lmax(imSP{Kndx});
                    saveFile = fullfile(DataDir,labelSet,'ClusterLabeling',['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize) myN2S(testParams.MinClusterSize) 'Cut Best'],baseFName);make_dir(saveFile);
                    save(saveFile,'L','labelList');
                    [~, expClusterPickCutOff] = min(clusterRank(1:clusterCutOff));
                    [~, lmax] = max(probPerLabelExp{bestclNdxCutOff},[],2);
                    L = lmax(imSP{Kndx});
                    saveFile = fullfile(DataDir,labelSet,'ClusterLabeling',['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize) myN2S(testParams.MinClusterSize) 'Cut Rank'],baseFName);make_dir(saveFile);
                    save(saveFile,'L','labelList');
                    
                    
                    
                    fprintf('%s %d SVMpick\n',clusterName,svmclusterNdx);
                    
                    if(doWeb && retSetSize == testParams.retSetSize(1))
                        fprintf(imFID,'<a name="%d"></a><h2>Cluster %s: %d clusters</h2>',j,clusterName,length(unique(clusters{j})));
                        fprintf(imFID,'\n<table border="0">\n');
                        fprintf(imFID,'\t<tr>\n');
                        %best cluster 
                        fprintf(imFID,'\t\t<td><center> Best Cluster: %d<br>',bestclNdx);
                        fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' ['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize)] '/' folder '/' file myN2S(bestclNdx) '.png']);
                        fprintf(imFID,'\t%.2f%%\n ',100*resultsExp{i,j}(bestclNdx,1)/resultsExp{i,j}(bestclNdx,2));
                        fprintf(imFID,'</center> </td>\n');
                         %svm Cluster
                        fprintf(imFID,'\t\t<td><center>SVM Selected Cluster: %d<br>',svmclusterNdx);
                        fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' ['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize)] '/' folder '/' file myN2S(svmclusterNdx) '.png']);
                        fprintf(imFID,'\t%.2f%%\n ',100*resultsExp{i,j}(svmclusterNdx,1)/resultsExp{i,j}(svmclusterNdx,2));
                        fprintf(imFID,'</center> </td>\n');
                         %svm Cluster
                        fprintf(imFID,'\t\t<td><center>Best Cluster Cutoff %d: %d<br>',testParams.MinClusterSize,bestclNdxCutOff);
                        fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' ['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize)] '/' folder '/' file myN2S(bestclNdxCutOff) '.png']);
                        fprintf(imFID,'\t%.2f%%\n ',100*resultsExp{i,j}(bestclNdxCutOff,1)/resultsExp{i,j}(bestclNdxCutOff,2));
                        fprintf(imFID,'</center> </td>\n');
                         %svm Cluster
                        fprintf(imFID,'\t\t<td><center>SVM Selected Cluster Cutoff %d: %d<br>',testParams.MinClusterSize,svmclusterNdxCutOff);
                        fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' ['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize)] '/' folder '/' file myN2S(svmclusterNdxCutOff) '.png']);
                        fprintf(imFID,'\t%.2f%%\n ',100*resultsExp{i,j}(svmclusterNdxCutOff,1)/resultsExp{i,j}(svmclusterNdxCutOff,2));
                        fprintf(imFID,'</center> </td>\n');
                        fprintf(imFID,'\t</tr>\n</table>');
                        for cNdx = clusterInds(:)'
                            labelImOut = fullfile(HOMEWEB,'Cluster',['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize)],folder,[file myN2S(cNdx) '.png']);make_dir(labelImOut);
                            [foo lmax] = max(probPerLabelExp{cNdx},[],2);
                            L = lmax(imSP{Kndx});
                            %if(~exist(labelImOut,'file'))
                                DrawImLabels(L,L,labelColors{labelType},Labels{labelType},labelImOut,128,0,labelType,maxDim);
                            %end
                            fprintf(imFID,'<h3>Cluster %d: %d images</h3>',cNdx,sum(cluster==cNdx));
                            fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' ['Exp' myN2S(retSetSize) clusterName myN2S(testParams.ExpansionSize)] '/' folder '/' file myN2S(cNdx) '.png']);
                            fprintf(imFID,'\t%.2f%%\n ',100*resultsExp{i,j}(cNdx,1)/resultsExp{i,j}(cNdx,2));
                            fprintf(imFID,'<br><h3>Expanded Cluster</h3>\n');
                            DisplayCluster(imFID,ones(size(expandedInds{cNdx}))',1,[backOutString '../Images/'],trainFileList(expandedInds{cNdx}),numColsClust);
                            fprintf(imFID,'<br><h3>Original Cluster</h3>\n');
                            DisplayCluster(imFID,cluster,cNdx,[backOutString '../Images/'],trainFileList(retInds),numColsClust);
                    	end
                    end
                    %}
                end
                %}
            end
            Kstr = [Kstr num2str(testParams.K(Kndx))];
        end
    end
    
    if(doWeb)
        fprintf(imFID,'</BODY>\n</HTML>');
        fclose(imFID);
    end
    %}
    ProgressBar(pfig,find(i==range),length(range));
end
close(pfig);
close(pfig2);
if(doWeb)
    fprintf(indexFID,'\t</tr>\n</table></center>');
    fprintf(indexFID,'</BODY>\n</HTML>');
    fclose(indexFID);
end

end

function [clusterNdx clusterScore] = SVMPickCluster(cluster,trainGlobalDesc,testGlobalDesc,descList,HOMEDATA,baseFName,clusterName,maxClusterSize)
    savefile = fullfile(HOMEDATA,'SVMCluster',[baseFName clusterName '.mat']);
    if(exist(savefile,'file'))
        load(savefile);
        return;
    end
    if(iscell(cluster))
        if(length(cluster) > maxClusterSize)
            cluster = cluster(1:maxClusterSize);
        end
        allinds = cell2mat(cluster);
        uniqeuCellClusterInds = unique(allinds);
        trainGlobalDesc = SelectDesc(trainGlobalDesc,uniqeuCellClusterInds,1);
        for i = 1:length(cluster)
            [a b] = intersect(uniqeuCellClusterInds,cluster{i});
            cluster{i} = b;
        end
    end
    
    kernelList = cell(0);%{'gaussian';'intersection';'intersection'};
    trainData = cell(0);%{trainGlobalDesc.colorGist;trainGlobalDesc.coHist;trainGlobalDesc.spatialPry};
    testData = cell(0);%{testGlobalDesc.colorGist;testGlobalDesc.coHist;testGlobalDesc.spatialPry};
    descStr = '';
    for i = 1:length(descList)
        trainData{i,1} = trainGlobalDesc.(descList{i});
        testData{i,1} = testGlobalDesc.(descList{i});
        descStr = [descStr descList{i}(1:3) descList{i}(end-2:end)];
        if(strcmp(descList{i},'colorGist'))
            kernelList{i,1} = 'gaussian';
        else
            kernelList{i,1} = 'intersection';
        end
    end
    kerneloptionList=cell(length(trainData),1);
    for i=1:length(trainData)
        % estimate the bandwidth for the gaussian kernel
        if strcmp(kernelList{i,1},'gaussian')
            x=trainData{i};
            index=randperm(size(x,1));
            xDis = dist2(x(index(1:min(end,100)),:), x);
            xSort=sort(xDis,2);
            xSort=xSort(:,10);
            kerneloptionList{i}=mean(xSort);
        else
            kerneloptionList{i}=1;
        end
    end
    ps  =  zeros(size(trainData{1},1));	
    for i=1:length(trainData)
        pstemp=svmkernel(trainData{i},kernelList{i},kerneloptionList{i});
        ps=ps+pstemp;
    end
    
    if(iscell(cluster))
        clusterNdx = 1:length(cluster);
    else
        clusterNdx = unique(cluster);
    end
    c = 2;
    epsilon = .000001;
    verbose = 0;
    clusterScore = zeros(length(clusterNdx),1);
    for i = clusterNdx(:)'
        if(iscell(cluster))
            labeling = -ones(length(uniqeuCellClusterInds),1);
            labeling(cluster{i}) = 1;
        else
            labeling = -ones(length(cluster),1);
            labeling(cluster==i) = 1;
        end
        [xsupList,w,b]=svmclassMK(trainData,labeling,c,epsilon,kernelList,kerneloptionList,verbose,[],[],ps);
        clusterScore(i) = svmvalMK(testData,xsupList,w,b,kernelList,kerneloptionList);
    end
    [foo clusterNdx] = max(clusterScore);
    make_dir(savefile);save(savefile,'clusterNdx','clusterScore');
end

function [str] = myN2S(num,prec)
    if(~exist('prec','var'))
        prec = 2;
    end
    if(num<1)
        str = sprintf('%%.%df',prec);
        str = sprintf(str,num);
    else
        str = sprintf('%%0%dd',prec);
        str = sprintf(str,num);
    end
end

function DisplayCluster(fid,clusters,clNdx,imageFolder,fileList,numCols)
    mask = clusters==clNdx;
    colCount = 0;
    fprintf(fid,'\n<table border="0">\n');
    fprintf(fid,'\t<tr>\n');
    for i = find(mask)'
        [folder file ext] = fileparts(fileList{i});
        fprintf(fid,'\t\t<td><center> <img  width="128" src="%s"></a> ',[imageFolder folder '/' file ext]);
        fprintf(fid,'</center> </td>\n');
        colCount = colCount +1;
        if(colCount == numCols)
            colCount = 0;
            fprintf(fid,'\t</tr><tr>\n');
        end
    end
    fprintf(fid,'\t</tr>\n</table>');
end
