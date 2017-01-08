function ParseTestImagesRetClusterPerfSL(HOMEDATA,HOMETESTDATA,HOMEIMAGES,HOMELABELSETS,testFileList,trainFileList,trainGlobalDesc,testGlobalDesc,trainIndex,testIndex,trainCounts,labelPenality,Labels,testParams,fullSPDesc)

try for i = 3:100;fclose(i);fprintf('closed %d\n',i);end;catch ERR;end
%close all;
pfig = ProgressBar('Parsing Images');

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
    HOMEWEB = fullfile(DataDir,'PSLWebsite');
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

range = 1:length(testFileList);
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
                %{
                retSetSPDesc = [];
                if(isempty(fullSPDesc{labelType,Kndx}))
                    retSetSPDesc = LoadSegmentDesc(trainFileList,retSetIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(1));
                else
                    for dNdx = 1:length(testParams.segmentDescriptors)
                        retSetSPDesc.(testParams.segmentDescriptors{dNdx}) = fullSPDesc{labelType,Kndx}.(testParams.segmentDescriptors{dNdx})(descMask,:);
                    end
                end
                %}
                %Find the labeling for just using the ret set
                %{-
                gtLabels = testIndex{labelType}.label(testIndex{labelType}.image==i)';
                spSizes = testIndex{labelType}.spSize(testIndex{labelType}.image==i)';
                probPerLabelBase = GetAllProbPerLabel(fullfile(DataDir,labelSet,'probPerLabel'),baseFName,suffix,retSetIndex,[],labelNums,trainCounts{labelType,Kndx},'ratio',1); %#ok<AGROW>
                if(isempty(probPerLabelBase))
                    retSetSPDesc = LoadSegmentDesc(trainFileList,retSetIndex,HOMEDATA,testParams.segmentDescriptors,testParams.K(1));
                    rawNNs = DoRNNSearch(testImSPDesc,retSetSPDesc,fullfile(DataDir,labelSet,'RNN'),baseFName,suffix,testParams,Kndx,1);
                    probPerLabelBase = GetAllProbPerLabel(fullfile(DataDir,labelSet,'probPerLabel'),baseFName,suffix,retSetIndex,rawNNs,labelNums,trainCounts{labelType,Kndx},'ratio',1); %#ok<AGROW>
                end
                slinds = unique(testIndex{labelType}.label(testIndex{labelType}.image==i));
                slmask = zeros(size(probPerLabelBase,2),1)==1;slmask(slinds) = 1;
                [~, lmaxnsl] = max(probPerLabelBase,[],2);
                Lnsl = lmaxnsl(imSP{Kndx});
                probPerLabelBase(:,~slmask) = min(probPerLabelBase(:))-1;
                [~, lmax] = max(probPerLabelBase,[],2);
                L = lmax(imSP{Kndx});
                labelList = Labels{labelType};
                make_dir(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) 'PerfSL'],baseFName));
                save(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) 'PerfSL'],baseFName),'L','labelList');
                
                dataCost{1} = -(probPerLabelBase-max(probPerLabelBase(:))-1);
                for labelSmoothing = testParams.LabelSmoothing
                    testName = sprintf('%s S%d IS%d P%s IP%s',[myN2S(retSetSize) 'PerfSL'],labelSmoothing,testParams.InterLabelSmoothing,...
                                testParams.LabelPenality{1}(1:3),testParams.InterLabelPenality{1}(1:3));
                    smoothingMatrix = BuildSmoothingMatrix(labelPenality(2,2),labelSmoothing,testParams.InterLabelSmoothing,testParams.LabelPenality{1},testParams.InterLabelPenality{1});
                    [L Lsp] = MultiLevelSegMRF(DataDir,HOMELABELSETS(2),testName,baseFName,Labels(2),imSP{Kndx},adjPairs,dataCost,smoothingMatrix);
                end
                
                [~, lmax] = max(probPerLabelBase,[],2);
                L = lmax(imSP{Kndx});
                labelList = Labels{labelType};
                make_dir(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) 'PerfSL'],baseFName));
                save(fullfile(DataDir,labelSet,'ClusterLabeling',[myN2S(retSetSize) 'PerfSL'],baseFName),'L','labelList');
                
                mask = zeros(size(lmax))==1;mask(testIndex{labelType}.sp(testIndex{labelType}.image==i))=1;
                correct = lmax(mask)==gtLabels;
                slr = sum(correct.*spSizes)/sum(spSizes);
                mask = zeros(size(lmaxnsl))==1;mask(testIndex{labelType}.sp(testIndex{labelType}.image==i))=1;
                correct = lmaxnsl(mask)==gtLabels;
                nslr = sum(correct.*spSizes)/sum(spSizes);
                if(doWeb && retSetSize == testParams.retSetSize(1))
                    gtLabelsFixed = zeros(size(lmax));
                    gtLabelsFixed(mask) = gtLabels;
                    gsL = gtLabelsFixed(imSP{Kndx});
                    gsL = gsL+1;
                    labelImOut = fullfile(HOMEWEB,'Cluster','GT',folder,[file '.png']);make_dir(labelImOut);
                    %if(~exist(labelImOut,'file'))
                        DrawImLabels(gsL,gsL,[0 0 0; labelColors{labelType}],{'Unlabeld' Labels{labelType}{:}},labelImOut,128,0,labelType,maxDim);
                    %end
                    labelImOut = fullfile(HOMEWEB,'Cluster',[myN2S(retSetSize) 'NoCluster'],folder,[file '.png']);make_dir(labelImOut);
                    %if(~exist(labelImOut,'file'))
                        DrawImLabels(Lnsl,Lnsl,labelColors{labelType},Labels{labelType},labelImOut,128,0,labelType,maxDim);
                    %end
                    labelImOut = fullfile(HOMEWEB,'Cluster',[myN2S(retSetSize) 'PerfSL'],folder,[file '.png']);make_dir(labelImOut);
                    %if(~exist(labelImOut,'file'))
                        DrawImLabels(L,L,labelColors{labelType},Labels{labelType},labelImOut,128,0,labelType,maxDim);
                    %end
                    fprintf(imFID,'<h3>Perfect Shortlist</h3>');
                    fprintf(imFID,'\n<table border="0">\n');
                    fprintf(imFID,'\t<tr><td><center>Ground Truth<br>\n');
                    fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' 'GT/' folder '/' file '.png']);
                    fprintf(imFID,'\t%.2f%%\n',100);
                    fprintf(imFID,'\t</center></td><td><center>Base<br>\n');
                    fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' myN2S(retSetSize) 'NoCluster/' folder '/' file '.png']);
                    fprintf(imFID,'\t%.2f%%\n',100*nslr);
                    fprintf(imFID,'\t</center></td><td><center>Perfect Short List<br>\n');
                    fprintf(imFID,'\t<img  src="%s"><br>\n ',[backOutString '../Cluster/' myN2S(retSetSize) 'PerfSL/' folder '/' file '.png']);
                    fprintf(imFID,'\t%.2f%%\n',100*slr);
                    fprintf(imFID,'\t</center></td></tr></table><br>\nShort List:<br>');
                    fprintf(imFID,'%s,',Labels{labelType}{slinds});
                end
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
if(doWeb)
    fprintf(indexFID,'\t</tr>\n</table></center>');
    fprintf(indexFID,'</BODY>\n</HTML>');
    fclose(indexFID);
end

end

function clusterNdx = SVMPickCluster(cluster,trainGlobalDesc,testGlobalDesc,descList,HOMEDATA,baseFName,clusterName,maxClusterSize)
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
    make_dir(savefile);save(savefile,'clusterNdx');
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
