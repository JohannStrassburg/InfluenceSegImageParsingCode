
WEBLABELSETS = HOMELABELSETS(UseLabelSet);

labelColors = GetColors(HOME, HOMECODE, WEBLABELSETS, Labels(UseLabelSet));

mrfFold = 'MRF';
if(exist('testParams','var') && isfield(testParams,'MRFFold'))
    mrfFold = testParams.MRFFold;
end

try for i = 3:100;fclose(i);fprintf('closed %d\n',i);end;catch ERR;end
try close(pfig);catch ERR; end
%SIFT Flow
%{
WebTestList = {
    'R200K200TNN80-SPscGistCoHist-sc01ratio C10 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1',
    'R200K200TNN80-SPscGistCoHist-sc01ratio C10 B.5.1 S100 IS0.000 Pcon IPpot Seg WbS1',
    'R200K200TNN80-SPscGistCoHist-sc01ratio C10 B.5.1 S100 IS064 Pcon IPpot Seg WbS1'};
WebTestName = {'Base',
    'Smooth',
    'Joint'};
%}
%LMSun
%{
WebTestList = {'R200K200TNN80-SPscGistCoHist-sc01ratio C10 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1',
    %'R200K200TNN80-SPscGistCoHist-sc01ratio C10 B.5.1 S256 IS0.000 Pcon IPpot Seg WbS1',
    %'R200K200TNN80-SPscGistCoHist-sc01ratio C10 B.5.1 S256 IS256 Pcon IPpot Seg WbS1',
    'R200K200TNN80-SPscGistCoHist-sc01ratio C10 B.5.1 S256 IS512 Pcon IPpot Seg WbS1',
    'R200K200TNN80-SPscGistCoHist-sc01ratio C10 B.5.1 S0.000 IS0.000 Pcon IPpot RetSVMSeg WbS1',
    'R200K200TNN80-SPscGistCoHist-sc01ratio C10 B.5.1 S0.000 IS0.000 Pcon IPpot RetGTSeg WbS1'};
WebTestName = {'Base',
    %'Smooth',
    %'Joint 256',
    'Joint 512',
    'Base InOut SVM',
    'Base InOut GT'};
%}
%LMSun Detector Tests
%{-
WebTestList = {'MaxW0.00','MaxW1.00','MaxWinf'};
WebTestName = {'SuperPixels','Combo 1','Detector Max'};
%}
%Video
%{
WebTestList = {'K200 C11 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1',
    'Base-11-cFreq',
    'CombMaxDataSPSize-11-cFreq S016 IS064 Pcon IPcon wbs wbs'
    };
WebTestName = {'Base w/ ImSeg',
    'Base w/ VidSeg',
    'Joint 16'};
%}
if(exist('HOMETEST','var'))
    HOMETESTSET = fullfile(HOMETESTDATA,testParams.TestString);
    HOMEWEBTESTIMAGES = HOMETESTIMAGES;
else
    HOMETESTSET = fullfile(HOMEDATA,testParams.TestString);
    HOMEWEBTESTIMAGES = HOMEIMAGES;
end
HOMEWEB = fullfile(HOMETESTSET,'Website');

webIndexFile = fullfile(HOMEWEB,'index2.htm');
webSelectedFile = fullfile(HOMEWEB,'selected.htm');
if(exist('rangeN','var'))
    webIndexFile = fullfile(HOMEWEB,sprintf('index%d%d.htm',rangeN(1),rangeN(2)));
    webSelectedFile = fullfile(HOMEWEB,sprintf('selected%d%d.htm',rangeN(1),rangeN(2)));
end
make_dir(webIndexFile);
indexFID = fopen(webIndexFile,'w');
selectedFID = fopen(webSelectedFile,'w');
PrimaryLabelSet = 1;
PrimaryTestSet = 2;
SecondaryTestSet = 1;
numCols = 6;
colCount = 0;
colCountSelected = 0;
genIm = false;
sortFiles = true;

onlyLabeledArea = 1;
onlyCorrect = 1;
maxDim = 1000;


resultsStats = zeros([length(WebTestList) length(HOMELABELSETS) length(testFileList)]);

fprintf(indexFID,'\n<center>\n');
fprintf(indexFID,'\n<table border="0">\n');
fprintf(indexFID,'\t<tr>\n');
fprintf(selectedFID,'\n<center>\n');
fprintf(selectedFID,'\n<table border="0">\n');
fprintf(selectedFID,'\t<tr>\n');

pfig = ProgressBar('Generating Web');
range = 1:length(testFileList);
if(exist('rangeN','var'))
    range = SetupRange(rangeN(1),rangeN(2),length(testFileList));
end
selectedNdx = [];
%LM+Sun
%range = [26 130 137 152 157 192 194 221 292 387 463 411];
%CamVid
%range = [14 15 16 82:86];
%SiftFlow
%range = [ 77 4 20 55 56 72 126 156 195 170 116];
if(sortFiles)
    accGain = zeros(max(range),1);
    for i = range
        [folder base ext] = fileparts(testFileList{i});
        secondaryRestults = 0;
        for j = 1:length(WebTestList)
            for k = 1:length(WEBLABELSETS)
                [foo setBase] = fileparts(WEBLABELSETS{k});
                resultFile = fullfile(HOMETESTSET,mrfFold,setBase,WebTestList{j},folder,[base '.mat']);
                resultCache = [resultFile '.cache'];
                if(exist(resultCache,'file'))
                    load(resultCache,'-mat'); %metaData perLabelStat(#labelsx2) perPixelStat([# pix correct, # pix total]);
                end
                if(exist('perPixelStat','var'))
                    if(k==PrimaryLabelSet&&j==SecondaryTestSet)
                        secondaryRestults = 100*perPixelStat(1)/perPixelStat(2);
                    end
                    if(k==PrimaryLabelSet&&j==PrimaryTestSet)
                        if(exist('secondaryRestults','var'))
                            accGain(i) = (100*perPixelStat(1)/perPixelStat(2))-secondaryRestults;
                        end
                    end
                end
            end
        end
    end
    [accGain rndx] = sort(accGain(range),'descend');
    range = range(rndx);
end
    

for i = range
    im = imread(fullfile(HOMEWEBTESTIMAGES,testFileList{i}));
    [ro co ch] = size(im);
    [folder base ext] = fileparts(testFileList{i});
    backOutString = '../';
    if(isempty(folder))
        folder = '.';
        backOutString = '';
    else
        [f b] = fileparts(folder);
        if(~isempty(f))
            backOutString = '../../';
        end
    end
    clear secondaryRestults;
    
    %{
    retWebPage = fullfile(HOMEWEB,'RetrievalWeb',folder,[base '.htm']);make_dir(retWebPage);
    retFID = fopen(retWebPage,'w');
    fprintf(retFID,'<HTML>\n<HEAD>\n<TITLE>%s %s Retrieval Set</TITLE>\n</HEAD>\n<BODY>',folder,base);
    fprintf(retFID,'\n<center>\n');
    retInds = FindRetrievalSet(trainGlobalDesc,SelectDesc(testGlobalDesc,i,1),HOMETESTSET,fullfile(folder,base),testParams);
    fprintf(retFID,'<img width="%d" src="%s"> ',min(co,maxDim),[backOutString '../Images/' folder '/' base ext]);% width="400"
    for retSetSize = testParams.retSetSize
        retSetSize = min(retSetSize, length(retInds));
        imInds = retInds(1:retSetSize);
        fprintf(retFID,'<H1>%d image retrieval set</H1>',retSetSize);
        fprintf(retFID,'\n<table border="0">\n');
        count = 0;
        for imInd = imInds(:)'
            [retFolder retBase retExt] = fileparts(trainFileList{imInd});
            localImageFile = fullfile(HOMEWEB,'Images',retFolder,[retBase retExt]);make_dir(localImageFile);
            if(~exist(localImageFile,'file'))
                copyfile(fullfile(HOMEIMAGES,trainFileList{imInd}),localImageFile);
            end
            fprintf(retFID,'\t\t<td><center><a href="%s">',[backOutString '../Images/' retFolder '/' retBase retExt]);
            fprintf(retFID,'<img width="%d" src="%s"> ',200,[backOutString '../Images/' retFolder '/' retBase retExt]);% width="400"
            fprintf(retFID,'</a></center> </td>\n');
            count = count +1;
            if(count == numCols)
                count = 0;
                fprintf(retFID,'\t</tr><tr>\n');
            end
        end
        fprintf(retFID,'\t<tr>\n');
        fprintf(retFID,'\t</tr>\n</table></center>');
    end
    fprintf(retFID,'</BODY>\n</HTML>');
    fclose(retFID);
    
    
    if(isfield(testParams,'RetrievalMetaMatch'))
        metaFields = fieldnames(testParams.TrainMetadata);
        totalMask = ones(size(retInds))==1;
        for f = 1:length(metaFields)
            mask = strcmp(testParams.TrainMetadata.(metaFields{f})(retInds),testParams.TestMetadata.(metaFields{f}){i});
            totalMask = mask&totalMask;
        end
        retInds = retInds(totalMask);
        
        retWebPage = fullfile(HOMEWEB,'RetrievalWebInOut',folder,[base '.htm']);make_dir(retWebPage);
        retFID = fopen(retWebPage,'w');
        fprintf(retFID,'<HTML>\n<HEAD>\n<TITLE>%s %s Retrieval Set</TITLE>\n</HEAD>\n<BODY>',folder,base);
        fprintf(retFID,'\n<center>\n');
        fprintf(retFID,'<img width="%d" src="%s"> ',min(co,maxDim),[backOutString '../Images/' folder '/' base ext]);% width="400"
        for retSetSize = testParams.retSetSize
            retSetSize = min(retSetSize, length(retInds));
            imInds = retInds(1:retSetSize);
            fprintf(retFID,'<H1>%d image retrieval set</H1>',retSetSize);
            fprintf(retFID,'\n<table border="0">\n');
            count = 0;
            for imInd = imInds(:)'
                [retFolder retBase retExt] = fileparts(trainFileList{imInd});
                localImageFile = fullfile(HOMEWEB,'Images',retFolder,[retBase retExt]);make_dir(localImageFile);
                if(~exist(localImageFile,'file'))
                    copyfile(fullfile(HOMEIMAGES,trainFileList{imInd}),localImageFile);
                end
                fprintf(retFID,'\t\t<td><center><a href="%s">',[backOutString '../Images/' retFolder '/' retBase retExt]);
                fprintf(retFID,'<img width="%d" src="%s"> ',200,[backOutString '../Images/' retFolder '/' retBase retExt]);% width="400"
                fprintf(retFID,'</a></center> </td>\n');
                count = count +1;
                if(count == numCols)
                    count = 0;
                    fprintf(retFID,'\t</tr><tr>\n');
                end
            end
            fprintf(retFID,'\t<tr>\n');
            fprintf(retFID,'\t</tr>\n</table></center>');
        end
        fprintf(retFID,'</BODY>\n</HTML>');
        fclose(retFID);
    end
    %}
    imageWebPage = fullfile(HOMEWEB,'ImageWeb',folder,[base '.htm']);make_dir(imageWebPage);
    imFID = fopen(imageWebPage,'w');
    fprintf(imFID,'<HTML>\n<HEAD>\n<TITLE>%s %s</TITLE>\n</HEAD>\n<BODY>',folder,base);
    fprintf(imFID,'<SCRIPT LANGUAGE="JAVASCRIPT" TYPE="TEXT/JAVASCRIPT">\n<!--\nvar lAreas = false;\nvar lLegend = false;\nvar files = new Array();\n');
    fileBase = cell(0);
    webFileBase = cell(0);
    for j = 1:length(WebTestList)
        for k = 1:length(WEBLABELSETS)
            [foo setBase] = fileparts(WEBLABELSETS{k});
            fileBase{end+1} = fullfile(HOMEWEB,setBase,WebTestList{j},folder,base);make_dir(fileBase{end});
            webFileBase{end+1} = [backOutString '../' setBase '/' WebTestList{j} '/' folder '/' base];
        end
    end
    for j = 1:length(fileBase)
        fprintf(imFID,'files[%d] = "%s";\n',j,webFileBase{j});
    end
    fprintf(imFID,'function roll_over(img_name, i, over)\n{\n\tif(over){\n\t\tdocument[img_name].src = files[i]+"Correct.png";}\n');
    fprintf(imFID,'\telse{\n\t\tif(lAreas){\n\t\t\tdocument[img_name].src = files[i]+"LArea.png";\n\t\t}\n');
    fprintf(imFID,'\t\telse if(lLegend){\n\t\t\tdocument[img_name].src = files[i]+"NoLegend.png";\n\t\t}\n');
    fprintf(imFID,'\t\telse{\n\t\t\tdocument[img_name].src = files[i]+".png";}\n\t}\n}\n');
    fprintf(imFID,'function switch_larea(){\n\tlAreas=~lAreas;\n\tlLegend=false;\n\tfor(var i = 1; i < files.length; i++)\n\t{\n\t\timg_name = "im_"+i.toString();\n');
    fprintf(imFID,'\t\tif(lAreas){\n\t\t\tdocument[img_name].src = files[i]+"LArea.png";\n\t\t}\n');
    fprintf(imFID,'\t\telse if(lLegend){\n\t\t\tdocument[img_name].src = files[i]+"NoLegend.png";\n\t\t}\n\t\telse{\n\t\t\tdocument[img_name].src = files[i]+".png";\n\t\t}\n\t}\n}\n');
    fprintf(imFID,'function switch_legend(){\n\tlLegend=~lLegend;\n\tlAreas=false;\n\tfor(var i = 1; i < files.length; i++)\n\t{\n\t\timg_name = "im_"+i.toString();\n');
    fprintf(imFID,'\t\tif(lAreas){\n\t\t\tdocument[img_name].src = files[i]+"LArea.png";\n\t\t}\n');
    fprintf(imFID,'\t\telse if(lLegend){\n\t\t\tdocument[img_name].src = files[i]+"NoLegend.png";\n\t\t}\n\t\telse{\n\t\t\tdocument[img_name].src = files[i]+".png";\n\t\t}\n\t}\n}\n');
    fprintf(imFID,'//-->\n</SCRIPT>\n');
    fprintf(imFID,'<A HREF="javascript:void(0)" onclick="switch_larea()">Toggle Labeled Area</a><br>');
    fprintf(imFID,'<A HREF="javascript:void(0)" onclick="switch_legend()">Toggle Legend</a>');
    fprintf(imFID,'<br>imageNum %d',i);
    fprintf(imFID,'\n<center>\n');
    
    fprintf(imFID,'\n<table border="0">\n');
    fprintf(imFID,'\t<tr>\n');
    fprintf(imFID,'<td><img height="%d" src="%s"> </td>',min(ro,maxDim),[backOutString '../Images/' folder '/' base ext]);% width="400"
    
    %edge images:
    %{
    edgeFile = fullfile(HOMEDATA,'BSEdata','Images',folder,[base '.JPEG.pbs']);
    edgeSuffix = '';
    if(EdgeParam == 0 && exist(edgeFile,'file'))
        edgeData = read_array(edgeFile);
        minVal = .5; div = (1-minVal)/.95;
        vC = 1-edgeData{1};       vC = vC(2:end-1,:);        vC = vC.^4;
        hC = 1-edgeData{2};       hC = hC(:,2:end-1);        hC = hC.^4;
        edgeSuffix = 'BSE';
    else
        [hCS vCS] = SpatialCues(im2double(im),EdgeParam);
        hCS = hCS(:,1:end-1); vCS = vCS(1:end-1,:);
        vC = exp(-vCS*5);
        hC = exp(-hCS*5);
        edgeSuffix = 'Norm';
    end
    vC = (vC-min(vC(:)))*(100/(max(vC(:))-min(vC(:))));
    hC = (hC-min(hC(:)))*(100/(max(hC(:))-min(hC(:))));
    
    edgeImOut = fullfile(HOMEWEB,'Edges',folder,[base '-' edgeSuffix]);make_dir(edgeImOut);
    if(~exist([edgeImOut '-v.png'],'file'))
        show(vC,1);
        set(gcf,'PaperPositionMode','auto');
        hold off;
        print([edgeImOut '-v.png'],'-dpng','-r96');%saveas(gcf,outfname);
    end
    fprintf(imFID,'<td><img height="%d" src="%s">  </td>',min(ro,maxDim),[backOutString '../Edges/' folder '/' base '-' edgeSuffix '-v.png']);% width="400"
    if(~exist([edgeImOut '-h.png'],'file'))
        show(hC,1);
        set(gcf,'PaperPositionMode','auto');
        hold off;
        print([edgeImOut '-h.png'],'-dpng','-r96');%saveas(gcf,outfname);
    end
    fprintf(imFID,'<td><img height="%d" src="%s">  </td>',min(ro,maxDim),[backOutString '../Edges/' folder '/' base '-' edgeSuffix '-h.png']);% width="400"
    %}
    fprintf(imFID,'\t</tr>\n</table><br>');
    
    fprintf(imFID,'\n<table border="0">\n');
    fprintf(imFID,'\t<tr>\n');
    localImageFile = fullfile(HOMEWEB,'Images',folder,[base ext]);make_dir(localImageFile);
    copyfile(fullfile(HOMEWEBTESTIMAGES,testFileList{i}),localImageFile,'f');
    fprintf(imFID,'\t\t<td><center>%s</center></td>\n','Ground Truth');
    groundTruth = cell(size(WEBLABELSETS));
    for k = 1:length(WEBLABELSETS)
        [foo setBase] = fileparts(WEBLABELSETS{k});
        groundTruthFile = fullfile(WEBLABELSETS{k},folder,[base '.mat']);
        if(~exist(groundTruthFile,'file'))
            fprintf(imFID,'\t\t<td><center>%s</center></td>\n','No Ground Truth');
            groundTruth{k} = [];
            continue;
        end
        load(groundTruthFile); %S metaData names
        groundTruth{k} = S;
        labelImOut = fullfile(HOMEWEB,'GroundTruth',setBase,folder,[base '.png']);make_dir(labelImOut);
        if(~exist(labelImOut,'file')||genIm)
            STemp = S+1;
            STemp(STemp<1) = 1;
            [imLabeled] = DrawImLabels(im,STemp,[0 0 0; labelColors{k}],{'unlabeled' names{:}},labelImOut,128,0,k,maxDim);
        end
        labelImOutNoLegend = fullfile(HOMEWEB,'GroundTruth',setBase,folder,[base 'NoLegend.png']);make_dir(labelImOut);
        if(~exist(labelImOutNoLegend,'file')||genIm)
            STemp = S+1;
            STemp(STemp<1) = 1;
            [imLabeledNL] = DrawImLabels(im,STemp,[0 0 0; labelColors{k}],{'unlabeled' names{:}},labelImOutNoLegend,0,0,k,maxDim);
        end
        
        if(k>1 && exist('slInd','var'))
            fprintf(imFID,'\t\t<td><center>');
            fprintf(imFID,'</center> </td>\n');
            fprintf(imFID,'\t\t<td><center>');
            fprintf(imFID,'</center> </td>\n');
        end
        fprintf(imFID,'\t\t<td><center>');
        fprintf(imFID,'<img  src="%s"> ',[backOutString '../GroundTruth/' setBase '/' folder '/' base '.png']);%width="800"
        fprintf(imFID,'</center> </td>\n');
        
    end
    fprintf(indexFID,'\t\t<td><center> <a href="%s">',['ImageWeb/' folder '/' base '.htm']);
    fprintf(indexFID,'<img  width="200" src="%s"></a> ',['Images/' folder '/' base ext]);
    if(any(i==selectedNdx))
        fprintf(selectedFID,'\t\t<td><center> <a href="%s">',['ImageWeb/' folder '/' base '.htm']);
        fprintf(selectedFID,'<img  width="200" src="%s"></a> ',['Images/' folder '/' base ext]);
    end
    fprintf(imFID,'\t</tr><tr>\n');
    count = 1;
    for j = 1:length(WebTestList)
        fprintf(imFID,'\t\t<td><center>%s</center></td>\n',WebTestName{j});
        for k = 1:length(WEBLABELSETS)
            [foo setBase] = fileparts(WEBLABELSETS{k});
            resultFile = fullfile(HOMETESTSET,mrfFold,setBase,WebTestList{j},folder,[base '.mat']);
            if(~exist(resultFile,'file'))
                continue;
            end
            load(resultFile); %L Lsp labelList
            resultCache = [resultFile '.cache'];
            if(exist(resultCache,'file'))
                load(resultCache,'-mat'); %metaData perLabelStat(#labelsx2) perPixelStat([# pix correct, # pix total]);
            end
            
                
            if(max(unique(L))== length(labelList)+1)
                labelList = [labelList(:)' {'unlabeled'}];
            end
            
            labelImOutCorrect = [fileBase{count} 'Correct.png'];
            labelImOutLArea = [fileBase{count} 'LArea.png'];
            labelImOutNoLegend = [fileBase{count} 'NoLegend.png'];
            labelImOut = [fileBase{count} '.png'];
            mask = ones(size(L))==1;
            if(~isempty(groundTruth{k}))
                mask = groundTruth{k}>0;
                if(all(mask>0))
                    mask = groundTruth{k}>1;
                end
            end
            if(~exist(labelImOutLArea,'file')||genIm)
                if(isempty(groundTruth{k}))
                    [imLabeled] = DrawImLabels(im,L,labelColors{k},labelList,labelImOutLArea,128,0,k,maxDim);
                else
                    temp = L;
                    temp(groundTruth{k}<1) = 0;
                    [imLabeled] = DrawImLabels(im,temp+1,[0 0 0; labelColors{k}],{'unlabeled' labelList{:}},labelImOutLArea,128,0,k,maxDim);
                end
            end
            if(~exist(labelImOutCorrect,'file')||genIm)
                if(isempty(groundTruth{k}))
                    [imLabeled] = DrawImLabels(im,L,labelColors{k},labelList,labelImOutCorrect,128,0,k,maxDim);
                else
                    temp = L;
                    temp(groundTruth{k}~=L) = -1;
                    temp(groundTruth{k}<1) = 0;
                    [imLabeled] = DrawImLabels(im,temp+2,[.5 0 0; 0 0 0; labelColors{k}],{'wrong' 'unlabeled' labelList{:}},labelImOutCorrect,128,0,k,maxDim);
                end
            end
            if(~exist(labelImOutNoLegend,'file')||genIm)
                [imLabeled] = DrawImLabels(im,L,labelColors{k},labelList,labelImOutNoLegend,0,0,k,maxDim);
            end
            if(~exist(labelImOut,'file')||genIm)
                [imLabeled] = DrawImLabels(im,L,labelColors{k},labelList,labelImOut,128,0,k,maxDim);
            end
            if(j==1 && k>1 && exist('slInd','var'))
                fprintf(imFID,'\t\t<td><center>');
                fprintf(imFID,'</center> </td>\n');
                fprintf(imFID,'\t\t<td><center>');
                fprintf(imFID,'</center> </td>\n');
            end
            if(j>1 && k>1 && exist('slInd','var'))
                shortList = slInd{i,j-1}{k};
                classOut = slFs{i,j-1}{k};
                ls = Labels{k};
                fprintf(imFID,'\t\t<td>\n');
                fprintf(imFID,'\t\t\t%s<br>\n','-------------------- ');
                [foo ind] = sort(classOut,'descend');
                for l = ind(:)'
                    fprintf(imFID,'\t\t\t%.2f: %s<br>\n',classOut(l),ls{l});
                end
                fprintf(imFID,'</td>\n');
                fprintf(imFID,'\t\t<td>\n');
                fprintf(imFID,'\t\t\t%s<br>\n','Short_List:');
                for l = find(shortList)
                    fprintf(imFID,'\t\t\t%s<br>\n',ls{l});
                end
                fprintf(imFID,'</td>\n');
            end
            fprintf(imFID,'\t\t<td><center>');
            imName = sprintf('im_%d',count);
            fprintf(imFID,'<a herf="javascript:void(0)" onmouseover="roll_over(''%s'',%d,1)" onmouseout="roll_over(''%s'',%d,0)">',imName,count,imName,count);
            fprintf(imFID,'<img  src="%s" name="%s"> </a>',[webFileBase{count} '.png'],imName);
            if(exist('perPixelStat','var'))
                fprintf(imFID,'<br>%.1f%',100*perPixelStat(1)/perPixelStat(2));
                resultsStats(j,k,i) = perPixelStat(1)/perPixelStat(2);
            end
            fprintf(imFID,'</center> </td>\n');
            
            count= count+1;
            if(exist('perPixelStat','var'))
                if(k==PrimaryLabelSet&&j==SecondaryTestSet)
                    secondaryRestults = 100*perPixelStat(1)/perPixelStat(2);
                end
                if(k==PrimaryLabelSet&&j==PrimaryTestSet)
                    fprintf(indexFID,'<br>%d: %.1f%',i,100*perPixelStat(1)/perPixelStat(2));
                    if(any(i==selectedNdx)); fprintf(selectedFID,'<br>%d: %.1f%',i,100*perPixelStat(1)/perPixelStat(2));end
                    if(exist('secondaryRestults','var'))
                        fprintf(indexFID,' %.1f%',(100*perPixelStat(1)/perPixelStat(2))-secondaryRestults);
                        accGain(i) = (100*perPixelStat(1)/perPixelStat(2))-secondaryRestults;
                        if(any(i==selectedNdx)); fprintf(selectedFID,' %.1f%',(100*perPixelStat(1)/perPixelStat(2))-secondaryRestults);end
                    end
                end
            end
        end
        fprintf(imFID,'\t</tr><tr>\n');
    end
    fprintf(indexFID,'<br><a href="%s">Retrieval Set</a>\n',['RetrievalWeb/' folder '/' base '.htm']);
    fprintf(indexFID,'<br><a href="%s">Retrieval InOut</a>\n',['RetrievalWebInOut/' folder '/' base '.htm']);
    fprintf(indexFID,'</center> </td>\n');
    colCount = colCount +1;
    if(colCount == numCols)
        colCount = 0;
        fprintf(indexFID,'\t</tr><tr>\n');
    end
    if(any(i==selectedNdx))
        fprintf(selectedFID,'<br><a href="%s">Retrieval Set</a>\n',['RetrievalWeb/' folder '/' base '.htm']);
        fprintf(selectedFID,'</center> </td>\n');
        colCountSelected = colCountSelected +1;
        if(colCountSelected == numCols)
            colCountSelected = 0;
            fprintf(selectedFID,'\t</tr><tr>\n');
        end
    end
    fprintf(imFID,'\t</tr>\n</table></center>');
    fprintf(imFID,'</BODY>\n</HTML>');
    fclose(imFID);
    ProgressBar(pfig,find(range==i),length(range));
end
close(pfig);
fprintf(indexFID,'\t</tr>\n</table></center>');
fclose(indexFID);
fprintf(selectedFID,'\t</tr>\n</table></center>');
fclose(selectedFID);
