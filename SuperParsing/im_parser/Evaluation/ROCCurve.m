%{
DataDir = fullfile(HOMEDATA,testParams.TestString);

try for i = 3:100;fclose(i);fprintf('closed %d\n',i);end;catch ERR;end
try close(pfig);catch ERR; end


fgindex = testIndex{1};
fgindex.pbk = zeros(size(fgindex.image));
fgindex.pfg = zeros(size(fgindex.image));

numCols = 6;
colCount = 0;
maxDim = 400;

pfig = ProgressBar('Generating ROC');
range = 1:length(testFileList);
for i = range
    [folder base ext] = fileparts(testFileList{i});
    baseFName = fullfile(folder,base);
    
    Kndx=1;labelType=1;
    [testImSPDesc imSP{Kndx}] = LoadSegmentDesc(testFileList(i),[],HOMEDATA,testParams.segmentDescriptors,testParams.K(Kndx));
    features = GetFeaturesForClassifier(testImSPDesc);
    p = test_boosted_dt_mc(classifiers{Kndx,labelType}, features);
    spsInIndex = fgindex.sp(fgindex.image==i);
    spNdx = intersect(spsInIndex,1:size(p,1));
    fgindex.pbk(fgindex.image==i) = p(spNdx,1);
    fgindex.pfg(fgindex.image==i) = p(spNdx,2);
    ProgressBar(pfig,find(i==range),length(range));
end
try close(pfig);catch ERR; end
%}
close all;

numFg = sum(fgindex.label==2);
numBk = sum(fgindex.label==1);

%fg threshold
minp = min(fgindex.pfg);
maxp = max(fgindex.pfg);

i = 1;
range = maxp:-.1:minp;
recal = zeros(size(range));
prec = zeros(size(range));
trueNeg = zeros(size(range));
for thresh = range
    vote = fgindex.pfg>thresh; vote = vote'+1;
    correct = vote==fgindex.label;
    recal(i) = sum(correct(vote==2))/numFg;
    prec(i) = sum(correct(vote==2))/sum(vote==2);
    trueNeg(i) = sum(correct(vote==1))/numBk;
    i = i +1;
end
figure,plot(trueNeg,recal);set(gca, 'xdir','reverse');


%bg threshold
minp = min(fgindex.pbk);
maxp = max(fgindex.pbk);

i = 1;
range = maxp:-.1:minp;
recal = zeros(size(range));
prec = zeros(size(range));
trueNeg = zeros(size(range));
for thresh = range
    vote = fgindex.pbk>thresh; vote = 2-vote';
    correct = vote==fgindex.label;
    recal(i) = sum(correct(vote==2))/numFg;
    prec(i) = sum(correct(vote==2))/sum(vote==2);
    trueNeg(i) = sum(correct(vote==1))/numBk;
    i = i +1;
end
figure,plot(trueNeg,recal);set(gca, 'xdir','reverse');



%fg threshold
minp = min(fgindex.pbk-fgindex.pfg);
maxp = max(fgindex.pbk-fgindex.pfg);

i = 1;
range = minp:.1:maxp;
recal = zeros(size(range));
prec = zeros(size(range));
trueNeg = zeros(size(range));
for thresh = range
    vote = fgindex.pfg+thresh>fgindex.pbk; vote = vote'+1;
    correct = vote==fgindex.label;
    recal(i) = sum(correct(vote==2))/numFg;
    prec(i) = sum(correct(vote==2))/sum(vote==2);
    trueNeg(i) = sum(correct(vote==1))/numBk;
    i = i +1;
end
figure,plot(trueNeg,recal);set(gca, 'xdir','reverse');
figure,plot(recal,prec);


