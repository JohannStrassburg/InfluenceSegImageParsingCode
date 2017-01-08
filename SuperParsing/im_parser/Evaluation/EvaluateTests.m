function [conMats perPixelRates] =  EvaluateTests(HOMEDATA,HOMELABELSETS,testStrings,suffix,matchMetaData,conMatTest,outputFolder)

labels = cell(length(HOMELABELSETS),length(testStrings));
tests = cell(length(HOMELABELSETS),length(testStrings));
perPixelRates = cell(length(HOMELABELSETS),length(testStrings));
perLabelStats = cell(length(HOMELABELSETS),length(testStrings));
conMats = cell(length(HOMELABELSETS),length(testStrings));
if(~exist('matchMetaData','var'));matchMetaData = [];end
if(~exist('conMatTest','var'));conMatTest = [];end
if(~exist('suffix','var'));suffix = '';end
if(~exist('outputFolder','var'));outputFolder = [];end
if(isempty(outputFolder));outputFolder = 'MRF';end%'ClusterLabeling';%
for testSet = 1:length(testStrings)
    TestDir = fullfile(HOMEDATA,testStrings{testSet});
    for i = 1:length(HOMELABELSETS)
        [foo labelSet] = fileparts(HOMELABELSETS{i});
        [labels{i,testSet} tests{i,testSet} perPixelRates{i,testSet} perLabelStats{i,testSet} conMats{i,testSet}] = EvaluateLabelSetTests(HOMELABELSETS{i},fullfile(TestDir,outputFolder,labelSet),matchMetaData,conMatTest,1);
    end
    if(~isempty(HOMELABELSETS))
        if(isempty(conMatTest))
            savefile = fullfile(TestDir,sprintf('Results%s.txt',suffix));
            WriteToFile(savefile,labels(:,testSet), tests(:,testSet), perPixelRates(:,testSet), perLabelStats(:,testSet));
        else
            savefile = fullfile(TestDir,sprintf('ConMat%s.txt',suffix));
            WriteConMat(savefile,labels(:,testSet), conMats(:,testSet));
        end
    end
end


function [labels tests perPixelRates perLabelStats conMat] = EvaluateLabelSetTests(HOMELABEL,HOMETESTS,matchMetaData,conMatTest,skipNonCon)
dirR = dir(fullfile(HOMETESTS,'*'));
tests = cell(0);
for i = 3:length(dirR)
    if(dirR(i).isdir)
        tests{end+1} = dirR(i).name;
    end
end
if(isempty(conMatTest))
    skipNonCon = 0;
end
conMat = [];
perPixelRates = zeros(1,length(tests));
[foo labelset] = fileparts(HOMELABEL);
pfig = ProgressBar(sprintf('Evaluating Tests for: %s',labelset));
for i = 1:length(tests)
    if(strcmp(conMatTest,tests{i}))
        [labels perPixelRates(i) perLabelStat conMat] = EvaluateLabelSetTest(HOMELABEL,fullfile(HOMETESTS,tests{i}),matchMetaData);
    elseif(~skipNonCon)
        [labels perPixelRates(i) perLabelStat] = EvaluateLabelSetTest(HOMELABEL,fullfile(HOMETESTS,tests{i}),matchMetaData);
    end
    if(~exist('perLabelStats','var'))
        if(~exist('perLabelStat','var'))
            [labels perPixelRates(i) perLabelStat] = EvaluateLabelSetTest(HOMELABEL,fullfile(HOMETESTS,tests{i}),matchMetaData);
        end
        perLabelStats=zeros([size(perLabelStat) length(tests)]);
    end
    perLabelStats(:,:,i) = perLabelStat;
    ProgressBar(pfig,i,length(tests));
end
close(pfig);


function [labels perPixelRates perLabelStats conMat] = EvaluateLabelSetTest(HOMELABEL,HOMETEST,matchMetaData)
files = dir_recurse(fullfile(HOMETEST,'*.mat'),0);
perPixelStats = zeros(2,length(files));
fields = [];
perLabelStats = [];
if(isstruct(matchMetaData))
    fields = fieldnames(matchMetaData);
end
conMat = [];
for i = 1:length(files)
    [folder file] = fileparts(files{i});
    saveFile = fullfile(HOMETEST,[files{i} '.cache']);
    clear metaData conM;
    if(exist(saveFile,'file'))
        load(saveFile,'-mat');
        mustRedo = false;
        for j = 1:length(fields)
            if(~isfield(metaData,fields{j}))
                mustRedo = true; 
            end
        end
    end
    %{
    %code for finding specific files with classifications of sepcific labels
    if(any(perLabelStat([45 53],:)>0))
        fprintf('%d %d/%d %d/%d\n',i,perLabelStat(45,1),perLabelStat(45,2),perLabelStat(53,1),perLabelStat(53,2));
        fprintf('');
    end
    %}
    if(~exist(saveFile,'file') || (nargout>3&&~exist('conM','var')) || mustRedo)
        load(fullfile(HOMETEST,files{i})); %L Lsp labelList
        load(fullfile(HOMELABEL,files{i})); %S names
        metaFile = fullfile(HOMELABEL,'..','Metadata',files{i}); %metaData
        if(exist(metaFile,'file'))
            load(metaFile); % metadata
        end
        if(nargout>3)
            [perPixelStat perLabelStat conM] = EvalPixelLabeling(L,labelList,S,names);
        else
            [perPixelStat perLabelStat] = EvalPixelLabeling(L,labelList,S,names);
        end
        if(~exist('metaData','var'))
            metaData=[];
        end
        if(nargout>3)
            save(saveFile,'perPixelStat','perLabelStat','metaData','conM','-mat');
        else
            save(saveFile,'perPixelStat','perLabelStat','metaData','-mat');
        end
    end
    bad = 0;
    for j = 1:length(fields)
        if(~isfield(metaData,fields{j}) || ~strcmp(metaData.(fields{j}),matchMetaData.(fields{j})))
            bad = 1;
        end
    end
    if(bad); continue; end;
    perPixelStats(:,i) = perPixelStat;
    if(isempty(perLabelStats))
        perLabelStats = zeros(size(perLabelStat));
    end
    perLabelStats = perLabelStats+perLabelStat;
    if(nargout>3)
        if(isempty(conMat))
            conMat = conM;
        else
            conMat = conMat+conM;
        end
    end
end
if(~exist('names','var'))
    load(fullfile(HOMELABEL,files{1})); %S names metadata
    load(fullfile(HOMETEST,files{i})); %L Lsp labelList
end
labels = labelList;
perPixelRates = sum(perPixelStats(1,:))/sum(perPixelStats(2,:));


function WriteToFile(outFile,labels, tests, perPixelRates, perLabelStats)
allTests = tests{1};
for i = 2:length(tests)
    allTests = unique([allTests tests{i}]);
end
fid = fopen(outFile,'w');
fprintf(fid,'Class\t# of Testing Examples\t');
for i = 1:length(allTests)
    fprintf(fid,'%s\t',allTests{i});
end
fprintf(fid,'\n');
for i = 1:length(labels)
    fprintf(fid,'%s\t','PerPixel');
    if(isempty(perLabelStats{i}))
        continue;
    end
    labelCounts = perLabelStats{i}(:,2,1);
    fprintf(fid,'%d\t',fix(sum(labelCounts)));
    testInds = zeros(size(allTests));
    for j = 1:length(tests{i})
        testInds(strcmp(tests{i}{j},allTests))=j;
    end
    for j = testInds(:)'
        if(j==0); fprintf(fid,'0.00%%\t'); continue;end
        fprintf(fid,'%.2f%%\t',100.*perPixelRates{i}(j));
    end
    fprintf(fid,'\n');
    
    fprintf(fid,'%s\t','Mean Class');
    fprintf(fid,'%d\t',fix(sum(labelCounts)));
    [foo labelOrder] = sort(labelCounts,'descend');
    for j = testInds(:)'
        if(j==0); fprintf(fid,'0.00%%\t'); continue;end
        fprintf(fid,'%.2f%%\t',100.*mean((perLabelStats{i}(labelCounts~=0,1,j)./perLabelStats{i}(labelCounts~=0,2,j))));
    end
    fprintf(fid,'\n');
    for l = labelOrder(:)'
        if(labelCounts(l)==0);continue;end
        fprintf(fid,'%s\t',labels{i}{l});
        fprintf(fid,'%d\t',fix(labelCounts(l)));
        for j = testInds(:)'
            if(j==0); fprintf(fid,'0.00%%\t'); continue;end
            fprintf(fid,'%.2f%%\t',100.*perLabelStats{i}(l,1,j)./perLabelStats{i}(l,2,j));
        end
        fprintf(fid,'\n');
    end
    fprintf(fid,'\n');
end
fclose(fid);


function WriteConMat(savefile,labels,conMats)
fid = fopen(savefile,'w');
for i = 1:length(labels)
    conMat = conMats{i};
    label = labels{i};
    labelCounts = sum(conMat,2);
    label(labelCounts==0) = [];
    conMat(labelCounts==0,:) = [];
    conMat(:,labelCounts==0) = [];
    labelCounts(labelCounts==0) = [];
    [labelCounts sinds] = sort(labelCounts,'descend');
    conMat = conMat(sinds,:);
    conMat = conMat(:,sinds);
    label = label(sinds);
    conMat = conMat./repmat(labelCounts,[1 size(conMat,2)]);
    
    fprintf(fid,'Class\t# of Testing Examples\t');
    for j = 1:length(label)
        fprintf(fid,'%s\t',label{j});
    end
    fprintf(fid,'\n');
    for j = 1:length(label)
        fprintf(fid,'%s\t%d\t',label{j},labelCounts(j));
        for k = 1:length(label)
            fprintf(fid,'%.2f\t',conMat(j,k));
        end
        fprintf(fid,'\n');
    end
end
fclose(fid);











