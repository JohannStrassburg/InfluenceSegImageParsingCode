DataDir = fullfile(HOMEDATA,testParams.TestString);

try for i = 3:100;fclose(i);fprintf('closed %d\n',i);end;catch ERR;end
try close(pfig);catch ERR; end

HOMEWEB = fullfile(DataDir,'Website');
webIndexFile = fullfile(HOMEWEB,'index.htm');
make_dir(webIndexFile);
indexFID = fopen(webIndexFile,'w');
fprintf(indexFID,'\n<center>\n');
fprintf(indexFID,'\n<table border="0">\n');
fprintf(indexFID,'\t<tr>\n');

numCols = 6;
colCount = 0;
maxDim = 400;

pfig = ProgressBar('Generating Web');
range = 1:length(testFileList);
for i = range
    [folder base ext] = fileparts(testFileList{i});
    backOutString = '../';
    if(isempty(folder))
        folder = '.';
        backOutString = '';
    end
    baseFName = fullfile(folder,base);
    
    %Index Page Stuff
    localImageFile = fullfile(HOMEWEB,'Images',folder,[base ext]);make_dir(localImageFile);
    copyfile(fullfile(HOMEIMAGES,testFileList{i}),localImageFile,'f');
    fprintf(indexFID,'\t\t<td><center> <a href="%s">',['ImageWeb/' folder '/' base '.htm']);
    fprintf(indexFID,'<img  width="200" src="%s"></a> ',['Images/' folder '/' base ext]);
    if(exist('perPixelStat','var'))
        fprintf(indexFID,'<br>%.1f%',100*perPixelStat(1)/perPixelStat(2));
    end
    fprintf(indexFID,'</center> </td>\n');
    colCount = colCount +1;
    if(colCount == numCols)
        colCount = 0;
        fprintf(indexFID,'\t</tr><tr>\n');
    end
    
    
    imageWebPage = fullfile(HOMEWEB,'ImageWeb',folder,[base '.htm']);make_dir(imageWebPage);
    imFID = fopen(imageWebPage,'w');
    fprintf(imFID,'<HTML>\n<HEAD>\n<TITLE>%s %s</TITLE>\n</HEAD>\n<BODY>',folder,base);
    fprintf(imFID,'\n<center>\n');
    fprintf(imFID,'<img width="%d" src="%s"> ',maxDim,[backOutString '../Images/' folder '/' base ext]);% width="400"
    fprintf(imFID,'\n<table border="0">\n');
    fprintf(imFID,'\t<tr>\n');
    tmax = -1000;tmin = 1000;
    p = cell(length(testParams.K),length(HOMELABELSETS));
    for Kndx=1:length(testParams.K)
        [testImSPDesc imSP{Kndx}] = LoadSegmentDesc(testFileList(i),[],HOMEDATA,testParams.segmentDescriptors,testParams.K(Kndx));
        for labelType=1:length(HOMELABELSETS)
            features = GetFeaturesForClassifier(testImSPDesc);
            p{Kndx,labelType} = test_boosted_dt_mc(classifiers{Kndx,labelType}, features);
            tmax = max(tmax,max(p{Kndx,labelType}(:)));
            tmin = min(tmin,min(max(p{Kndx,labelType},[],2)));
        end
    end
    
    for Kndx=1:length(testParams.K)
        [testImSPDesc imSP{Kndx}] = LoadSegmentDesc(testFileList(i),[],HOMEDATA,testParams.segmentDescriptors,testParams.K(Kndx));
        for labelType=1:length(HOMELABELSETS)
            [foo setName] = fileparts(HOMELABELSETS{labelType});
            pmax = max(p{Kndx,labelType},[],2);
            dsp = pmax(imSP{Kndx});
            dsp(1,1) = tmax;
            dsp(1,2) = tmin;
            fileName = fullfile(HOMEWEB,'Images',setName,folder,[base '.png']);make_dir(fileName);
            DrawImLabels(dsp,[],[],[],fileName,0,404,1,maxDim);
            DrawImLabels(dsp,[],[],[],fileName,0,404,1,maxDim);
            fprintf(imFID,'\t\t<td><center>%s</center></td>\n ',setName);
            fprintf(imFID,'\t\t<td><center><img  src="%s"></a> <br>%s %.1f</center></td>\n ',[backOutString '../Images/' setName '/' folder '/' base '.png'],'Max Classifer Output', max(pmax));
            if(size(p{Kndx,labelType},2)<3)
                for j = 1:size(p{Kndx,labelType},2)
                    pt = p{Kndx,labelType}(:,j);
                    fileName = fullfile(HOMEWEB,'Images',setName,Labels{labelType}{j},folder,[base '.png']);make_dir(fileName);
                    DrawImLabels(pt(imSP{Kndx}),[],[],[],fileName,0,404,1,maxDim);
                    fprintf(imFID,'\t\t<td><center><img  src="%s"></a> <br>%s</center> </td>\n ',[backOutString '../Images/' setName '/' Labels{labelType}{j} '/' folder '/' base '.png'],Labels{labelType}{j});
                end
            end
            fprintf(imFID,'\t</tr><tr>');
        end
    end
    fprintf(imFID,'\t</tr>\n</table></center>');
    fprintf(imFID,'</BODY>\n</HTML>');
    fclose(imFID);
    
    ProgressBar(pfig,find(i==range),length(range));
end
close(pfig);