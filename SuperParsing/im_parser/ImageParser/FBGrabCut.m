function timing = FBGrabCut(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testGlobalDesc,labelPenality,Labels,classifiers,globalSVM,testParams)

DataDir = fullfile(HOMETESTDATA,testParams.TestString);
%close all;
pfig = ProgressBar('Parsing Images');
range = 1:length(testFileList);
timing = zeros(length(testFileList),4);
for i = range
    im = imread(fullfile(HOMEDATA,'..','Images',testFileList{i}));
    [folder file] = fileparts(testFileList{i});
    baseFName = fullfile(folder,file);
    
    FGSet = 1;
    classifierStr = repmat('0',[1 length(HOMELABELSETS)]);
    Kstr = num2str(testParams.K);
    clear imSP testImSPDesc;
    [testImSPDesc imSP] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K);
    probPerLabel = cell(size(HOMELABELSETS));
    for labelType=1
        [foo labelSet] = fileparts(HOMELABELSETS{labelType});
        if(~isempty(classifiers{labelType}))
            probCacheFile = fullfile(DataDir,'ClassifierOutput',labelSet,[baseFName '.mat']);
            classifierStr(labelType) = '1';
            if(~exist(probCacheFile,'file'))
                features = GetFeaturesForClassifier(testImSPDesc);
                prob = test_boosted_dt_mc(classifiers{labelType}, features);
                make_dir(probCacheFile);save(probCacheFile,'prob');
            else
                clear prob;
                load(probCacheFile);
            end
            probPerLabel{labelType} = prob;
            if(size(prob,2) == 1 && length(Labels{labelType}) ==2)
                probPerLabel{labelType}(:,2) = -prob;
            end
            if(labelType == FGSet)
                probPerLabel{labelType}(:,2) = probPerLabel{labelType}(:,2) + testParams.FGShift;
            end
        end
    end
    %Add Forground Background Classes
    
    
    
    [foo spL] = max(probPerLabel{1},[],2);
    L = spL(imSP);
    inputMask = L==2;
    [a b] = find(L==2);
    bb = [max(10,min(b)) max(10,min(a)) min(max(b),size(L,2)-10) min(max(a),size(L,1)-10)];
    labelList = Labels{1};
    Lsp = spL;
    outFileName = fullfile(DataDir,'MRF',labelSet,sprintf('base-fs%.1f',testParams.FGShift),sprintf('%s.mat',baseFName));%
    make_dir(outFileName);
    save(outFileName,'L','labelList','Lsp');
    for l1 = .5:.4:1.3
        for l2 = 1:4:9
            testName = sprintf('Mask fs%.1f l1-%.2f l2-%.1f',l1,l2,testParams.FGShift);%
            outFileName = fullfile(DataDir,'MRF',labelSet,testName,sprintf('%s.mat',baseFName));make_dir(outFileName);
            if(exist(outFileName,'file'))
                continue;
            end
            L = gcmask(im,inputMask,l1,l2);L = L+1;
            %L = gc(im,bb,l1,l2);L = L+1;
            Lsp = spL;
            for j = 1:length(Lsp)
                l = L(imSP==j);
                if(sum(l==1)>sum(l==2))
                    Lsp(j) = 1;
                else
                    Lsp(j) = 2;
                end
            end
            show(L,1);
            show(Lsp(imSP),2);
            save(outFileName,'L','labelList','Lsp');
        end
    end
    
    
    ProgressBar(pfig,find(i==range),length(range));
end
close(pfig);

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

