VIDLABELSETS = HOMELABELSETS(UseLabelSet);
videoTestFileList = fullTestFileList;%testFileList;%

%{-
labelColors = cell(size(Labels));
for k = 1:length(VIDLABELSETS)
    [foo setname] = fileparts(VIDLABELSETS{k});
    saveFile = fullfile(HOME,[setname 'colors.mat']);
    if(~exist(saveFile,'file'))
        numL = length(Labels{k});
        labelColor = MakeLabelColors(numL);
    else
        load(saveFile);
    end
    labelColors{k} = labelColor;
end
%}

for k = 1:length(VIDLABELSETS)
    [foo setname] = fileparts(VIDLABELSETS{k});
    saveFile = fullfile(HOME,[setname 'colors.mat']);
    labelColor = labelColors{k};
    if(~exist(saveFile,'file'))
        save(saveFile,'labelColor');
    end
end

mrfFold = 'MRF';
if(exist('testParams','var') && isfield(testParams,'MRFFold'))
    mrfFold = testParams.MRFFold;
end

WebTestList = {'Base-1-cFreq','CombMaxSPSize-1-cFreq'};

WebTestName = {'Base','Max (CL X SP Size)'};

if(exist('HOMETEST','var'))
    HOMETESTSET = fullfile(HOMETESTDATA,testParams.TestString);
    HOMEVIDTESTIMAGES = HOMETESTIMAGES;
else
    HOMETESTSET = fullfile(HOMEDATA,testParams.TestString);
    HOMEVIDTESTIMAGES = HOMEIMAGESALL;
end
HOMEVID = fullfile(HOMETESTSET,'Video');

maxDim = 1000;
genIm = 0;

pfig = ProgressBar('Generating Video');
range = 1:length(videoTestFileList);
if(exist('rangeN','var'))
    range = SetupRange(rangeN(1),rangeN(2),length(videoTestFileList));
end
frameCount = 0;
for k = 1:length(VIDLABELSETS)
    gtIm = [];rateString = cell(size(WebTestList));
    for i = range
        im = imread(fullfile(HOMEVIDTESTIMAGES,videoTestFileList{i}));
        [ro co ch] = size(im);
        [folder base ext] = fileparts(videoTestFileList{i});
        backOutString = '../';
        if(isempty(folder))
            folder = '.';
            backOutString = '';
        end
        clear secondaryRestults;

        fileBase = cell(0);
        webFileBase = cell(0);
        for j = 1:length(WebTestList)
            [foo setBase] = fileparts(VIDLABELSETS{k});
            fileBase{end+1} = fullfile(HOMEVID,setBase,WebTestList{j},folder,base);make_dir(fileBase{end});
            webFileBase{end+1} = [backOutString '../' setBase '/' WebTestList{j} '/' folder '/' base];
        end
        [foo setBase] = fileparts(VIDLABELSETS{k});
        groundTruthFile = fullfile(VIDLABELSETS{k},folder,[base '.mat']);
        if(exist(groundTruthFile,'file'))
            load(groundTruthFile); %S metaData names
            groundTruth{k} = S;
            gtImOut = fullfile(HOMEVID,'GroundTruth',setBase,folder,[base '.png']);make_dir(gtImOut);
            if(~exist(gtImOut,'file')||genIm)
                STemp = S;
                STemp(STemp<1) = length(names)+1;
                [imLabeled] = DrawImLabelsVideo(im,STemp,[ labelColors{k}; 0 0 0],{names{:} 'unlabeled' },gtImOut,128,0,k,maxDim);
            end
            gtIm = imread(gtImOut);
        end
        resultIm = cell(size(WebTestList));
        for j = 1:length(WebTestList)
            [foo setBase] = fileparts(VIDLABELSETS{k});
            resultFile = fullfile(HOMETESTSET,mrfFold,setBase,WebTestList{j},folder,[base '.mat']);
            if(~exist(resultFile,'file'));  continue; end
            load(resultFile); %L Lsp labelList
            resultCache = [resultFile '.cache'];
            if(exist(resultCache,'file'))
                load(resultCache,'-mat'); %metaData perLabelStat(#labelsx2) perPixelStat([# pix correct, # pix total]);
                rateString{j} = sprintf('\n%.1f%%',100*perPixelStat(1)/perPixelStat(2));
            end
            labelImOut = [fileBase{j} '.png'];
            if(~exist(labelImOut,'file')||genIm)
                [imLabeled] = DrawImLabelsVideo(im,L,labelColors{k},labelList,labelImOut,128,0,k,maxDim,[],[WebTestName{j} rateString{j}]);
            end
            resultIm{j} = imread(labelImOut);
        end
        [fold] = fileparts(fileBase{j});fold = fileparts(fold);fold = fileparts(fold);
        frameImOut = fullfile(fold,'videoFrames',sprintf('frame%06d.png',frameCount));make_dir(frameImOut);
        if(~exist(frameImOut,'file')||genIm)
            frameIm = [resultIm{1} resultIm{2}];
            frameIm2 = [gtIm im];
            frameIm = [frameIm; frameIm2 zeros([ro size(frameIm,2)-size(frameIm2,2) ch])];
            imwrite(imresize(frameIm,.5),frameImOut);
        end
        frameCount = frameCount +1;
        ProgressBar(pfig,find(i==range),length(range));
    end
end
close(pfig);
%{-
for j = 1:length(WebTestList)
    for k = 1:length(VIDLABELSETS)
        [foo setBase] = fileparts(VIDLABELSETS{k});
        frameImOut = fullfile(HOMEVID,setBase,WebTestList{j},'videoFrames','frame%06d.png');
        VideoOut = fullfile(HOMEVID,setBase,WebTestList{j},'videoFrames','video.mp4');
        if(~exist(VideoOut,'file')||genIm||1)
            if(exist(VideoOut,'file'))
                delete(VideoOut);
            end
            cmd = ['ffmpeg -r 30 -b 40000k -i "' frameImOut '" "' VideoOut '"'];
            system(cmd);
        end
    end
end
%}