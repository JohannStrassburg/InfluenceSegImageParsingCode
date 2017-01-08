
if(~exist('labelColors','var'))
    labelColors = cell(size(Labels));
    for k = 1:length(HOMELABELSETS)
        [foo setname] = fileparts(HOMELABELSETS{k});
        saveFile = fullfile(HOME,[setname 'colors.mat']);
        if(~exist(saveFile,'file'))
            labelColor = [rand([length(Labels{k}) 3]); [0 0 0]];
        else
            load(saveFile);
        end
        labelColors{k} = labelColor;
    end
end

for k = 1:length(HOMELABELSETS)
    [foo setname] = fileparts(HOMELABELSETS{k});
    saveFile = fullfile(HOME,[setname 'colors.mat']);
    labelColor = labelColors{k};
    save(saveFile,'labelColor');
end


try for i = 3:100;fclose(i);fprintf('closed %d\n',i);end;catch ERR;end
try close(pfig);catch ERR; end

%if(~exist('WebTestList','var'))
    WebTestList = {'K200 BConst.5 W0101010101 S04 IS0.00 Pcon IPcon',
                    'K200 BConst.5 W0101010101 S04 IS04 Pcon IPcon'};
    WebTestName = {'Smoothed',
                   'Joint'};
%end
if(exist('HOMETEST','var'))
    HOMETESTSET = fullfile(HOMETESTDATA,testParams.TestString);
    HOMEWEBTESTIMAGES = HOMETESTIMAGES;
else
    HOMETESTSET = fullfile(HOMEDATA,testParams.TestString);
    HOMEWEBTESTIMAGES = HOMEIMAGES;
end
HOMEWEB = fullfile(HOMETESTSET,'Website');
webIndexFile = fullfile(HOMEWEB,'index.htm');
make_dir(webIndexFile);
PrimaryLabelSet = 2;
PrimaryTestSet = 2;
genIm = false;
numCols = 6;
colCount = 0;

onlyLabeledArea = 1;
onlyCorrect = 1;
maxDim = 400;

indexFID = fopen(webIndexFile,'w');
fprintf(indexFID,'\n<center>\n');
fprintf(indexFID,'\n<table border="0">\n');
fprintf(indexFID,'\t<tr>\n');

pfig = ProgressBar('Generating Web');
improvement = zeros(length(testFileList),length(HOMELABELSETS));
labeledPixels = zeros(length(testFileList),length(HOMELABELSETS));
for i = 1:length(testFileList)
    im = imread(fullfile(HOMEWEBTESTIMAGES,testFileList{i}));
    [folder base ext] = fileparts(testFileList{i});
    backOutString = '../';
    if(isempty(folder))
        folder = '.';
        backOutString = '';
    end
        
    imageWebPage = fullfile(HOMEWEB,'ImageWeb',folder,[base '.htm']);make_dir(imageWebPage);
    imFID = fopen(imageWebPage,'w');
    fprintf(imFID,'<HTML>\n<HEAD>\n<TITLE>%s %s</TITLE>\n</HEAD>\n<BODY>',folder,base);
    fprintf(imFID,'<SCRIPT LANGUAGE="JAVASCRIPT" TYPE="TEXT/JAVASCRIPT">\n<!--\nvar lAreas = false;\nvar lLegend = false;\nvar files = new Array();\n');
    fileBase = cell(0);
    webFileBase = cell(0);
    for j = 1:length(WebTestList)
        for k = 1:length(HOMELABELSETS)
            [foo setBase] = fileparts(HOMELABELSETS{k});
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
    fprintf(imFID,'\n<center>\n');
    fprintf(imFID,'<img width="%d" src="%s"> ',maxDim,[backOutString '../Images/' folder '/' base ext]);% width="400"
    fprintf(imFID,'\n<table border="0">\n');
    fprintf(imFID,'\t<tr>\n');
    localImageFile = fullfile(HOMEWEB,'Images',folder,[base ext]);make_dir(localImageFile);
    copyfile(fullfile(HOMEWEBTESTIMAGES,testFileList{i}),localImageFile,'f');
    fprintf(imFID,'\t\t<td><center>%s</center></td>\n','Ground Truth');
    groundTruth = cell(size(HOMELABELSETS));
    for k = 1:length(HOMELABELSETS)
        [foo setBase] = fileparts(HOMELABELSETS{k});
        groundTruthFile = fullfile(HOMELABELSETS{k},folder,[base '.mat']);
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
            [imLabeled] = DrawImLabels(im,STemp,[0 0 0; labelColors{k}],{'Unlabeld' names{:}},labelImOut,128,0,k,maxDim);
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
    fprintf(imFID,'\t</tr><tr>\n');
    count = 1;
    for j = 1:length(WebTestList)
        fprintf(imFID,'\t\t<td><center>%s</center></td>\n',WebTestName{j});
        for k = 1:length(HOMELABELSETS)
            [foo setBase] = fileparts(HOMELABELSETS{k});
            resultFile = fullfile(HOMETESTSET,'MRF',setBase,WebTestList{j},folder,[base '.mat']);
            load(resultFile); %L Lsp labelList
            resultCache = [resultFile '.cache'];
            if(exist(resultCache,'file'))
                load(resultCache,'-mat'); %metaData perLabelStat(#labelsx2) perPixelStat([# pix correct, # pix total]);
            end
            
            labelImOutCorrect = [fileBase{count} 'Correct.png'];
            labelImOutLArea = [fileBase{count} 'LArea.png'];
            labelImOutNoLegend = [fileBase{count} 'NoLegend.png'];
            labelImOut = [fileBase{count} '.png'];
            if(~exist(labelImOutLArea,'file')||genIm)
                if(isempty(groundTruth{k}))
                    [imLabeled] = DrawImLabels(im,L,labelColors{k},labelList,labelImOutLArea,128,0,k,maxDim);
                else
                    temp = L;
                    temp(groundTruth{k}<1) = 0;
                    [imLabeled] = DrawImLabels(im,temp+1,[0 0 0; labelColors{k}],{'Unlabeld' labelList{:}},labelImOutLArea,128,0,k,maxDim);
                end
            end
            if(~exist(labelImOutCorrect,'file')||genIm)
                if(isempty(groundTruth{k}))
                    [imLabeled] = DrawImLabels(im,L,labelColors{k},labelList,labelImOutCorrect,128,0,k,maxDim);
                else
                    temp = L;
                    temp(groundTruth{k}~=L) = -1;
                    temp(groundTruth{k}<1) = 0;
                    [imLabeled] = DrawImLabels(im,temp+2,[1 0 0; 0 0 0; labelColors{k}],{'Wrong' 'Unlabeled' labelList{:}},labelImOutCorrect,128,0,k,maxDim);
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
                if(j==1)
                    labeledPixels(i,k) = perPixelStat(2);
                    if(perPixelStat(2)==0)
                        labeledPixels(i,k) = 1;
                    end
                    improvement(i,k) = perPixelStat(1);
                    fprintf(imFID,'<br>%.1f%%',100*perPixelStat(1)/labeledPixels(i,k));
                elseif(j==2)
                    if(perPixelStat(2)>0)
                        improvement(i,k) = perPixelStat(1) - improvement(i,k);
                    end
                    fprintf(imFID,'<br>Improvement: %.1f%%',100*improvement(i,k)/labeledPixels(i,k));
                end
               
            end
            fprintf(imFID,'</center> </td>\n');
            

            count= count+1;
            if(k==PrimaryLabelSet&&j==PrimaryTestSet)
                fprintf(indexFID,'\t\t<td><center> <a href="%s">',['ImageWeb/' folder '/' base '.htm']);
                fprintf(indexFID,'<img  width="200" src="%s"></a> ',['Images/' folder '/' base ext]);
                if(exist('perPixelStat','var'))
                    fprintf(indexFID,'<br>%d: %.1f%',i,100*perPixelStat(1)/perPixelStat(2));
                end
                %fprintf(indexFID,'<br><a href="%s">Retrieval Set</a>\n',['RetrievalWeb/' folder '/' base '.htm']);
                fprintf(indexFID,'</center> </td>\n');
                colCount = colCount +1;
                if(colCount == numCols)
                    colCount = 0;
                    fprintf(indexFID,'\t</tr><tr>\n');
                end
            end
        end
        fprintf(imFID,'\t</tr><tr>\n');
    end
    fprintf(imFID,'\t</tr>\n</table></center>');
    fprintf(imFID,'</BODY>\n</HTML>');
    fclose(imFID);
    ProgressBar(pfig,i,length(testFileList));
end
close(pfig);
fprintf(indexFID,'\t</tr>\n</table></center>');
fclose(indexFID);


indexFID = fopen(webIndexFile,'w');
fprintf(indexFID,'\n<center>\n');
fprintf(indexFID,'\n<table border="0">\n');
fprintf(indexFID,'\t<tr>\n');

colCount = 0;
[foo ndxs] = sort(sum(improvement,2),'descend');
for i = ndxs(:)'
    [folder base ext] = fileparts(testFileList{i});
    
    fprintf(indexFID,'\t\t<td><center> <a href="%s">',['ImageWeb/' folder '/' base '.htm']);
    fprintf(indexFID,'<img  width="200" src="%s"></a> ',['Images/' folder '/' base ext]);
    fprintf(indexFID,'<br>%d:',i);
    for k = 1:size(improvement,2)
        fprintf(indexFID,' %.1f%%',100*improvement(i,k)./labeledPixels(i,k));
    end
    fprintf(indexFID,'</center> </td>\n');
    colCount = colCount +1;
    if(colCount == numCols)
        colCount = 0;
        fprintf(indexFID,'\t</tr><tr>\n');
    end
end




