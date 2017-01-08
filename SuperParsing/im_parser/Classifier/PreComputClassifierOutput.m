function PreComputClassifierOutput(HOMETESTDATA,HOMELABELSETS,testFileList,classifiers,testParams)

DataDir = fullfile(HOMETESTDATA,testParams.TestString);
segmentDescriptors = testParams.segmentDescriptors;
K = testParams.K;
imBlock = 100;
close all;
for labelType=1:length(HOMELABELSETS)
	[foo labelSet] = fileparts(HOMELABELSETS{labelType});
    if(isempty(classifiers{labelType}))
        continue;
    end
    classifier = classifiers{labelType};
    numIm = length(testFileList);
    features = cell(numIm,1);
    mask = zeros(numIm,1)==1;
    for i = 1:numIm
        [folder file] = fileparts(testFileList{i});
        baseFName = fullfile(folder,file);
        probCacheFile = fullfile(DataDir,'ClassifierOutput',labelSet,[baseFName '.mat']);
        if(~exist(probCacheFile,'file'))
            mask(i) = true;
        end
    end
    probs = cell(numIm,1);
    tic
    if(matlabpool('size')==0); matlabpool(7); end
    
    pfig = ProgressBar(labelSet);
    for block = 1:ceil(numIm/imBlock)
        minR = 1+(block-1)*imBlock;
        maxR = min(numIm,block*imBlock);
        parfor i = minR:maxR
            if(mask(i))
                [testImSPDesc] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,segmentDescriptors,K);
                features = GetFeaturesForClassifier(testImSPDesc);
                probs{i} = test_boosted_dt_mc(classifier, features);
                %fprintf('%d\n',i);
            end
            %
        end
        for i = minR:maxR
            [folder file] = fileparts(testFileList{i});
            baseFName = fullfile(folder,file);
            prob = probs{i};
            probCacheFile = fullfile(DataDir,'ClassifierOutput',labelSet,[baseFName '.mat']);
            if(~isempty(prob))
                make_dir(probCacheFile);save(probCacheFile,'prob');
            end
        end
        ProgressBar(pfig,block,ceil(numIm/imBlock));
    end
    close(pfig);
end
matlabpool close;

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

