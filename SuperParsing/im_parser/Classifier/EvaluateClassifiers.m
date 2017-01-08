function EvaluateClassifiers(HOMETESTDATA,HOMELABELSETS,testFileList,testIndex,Labels,classifiers,testParams)

DataDir = fullfile(HOMETESTDATA,testParams.TestString);

close all;
fid = fopen(fullfile(DataDir,'Web','index.htm'),'w');fclose(fid);
for labelType=[1 2 4 5];%1:length(HOMELABELSETS)
	[foo labelSet] = fileparts(HOMELABELSETS{labelType});
    index = testIndex{labelType};
    numLabels = length(Labels{labelType});
    numSP = length(index.sp);
    probPerLabel = zeros(numSP,numLabels);
    pfig = ProgressBar('Evaluating Classifiers');
    range = unique(index.image);
    for i = range(:)'
    
        [folder file] = fileparts(testFileList{i});
        baseFName = fullfile(folder,file);
        clear imSP testImSPDesc;
        [testImSPDesc imSP] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K);
            
                
        if(isempty(classifiers{labelType}))
            fprintf('No Classifiers for class: %s\n',labelSet);
        else
            probCacheFile = fullfile(DataDir,'ClassifierOutput',labelSet,[baseFName '.mat']);
            if(~exist(probCacheFile,'file'))
                features = GetFeaturesForClassifier(testImSPDesc);
                prob = test_boosted_dt_mc(classifiers{labelType}, features);
                make_dir(probCacheFile);save(probCacheFile,'prob');
            else
                clear prob;
                load(probCacheFile);
            end
            if(size(prob,2) == 1 && numLabels ==2)
                probPerLabel(index.image==i,1) = prob(index.sp(index.image==i),1);
                probPerLabel(index.image==i,2) = -prob(index.sp(index.image==i),1);
            else
                probPerLabel(index.image==i,:) = prob(index.sp(index.image==i),:);
            end
        end
        ProgressBar(pfig,find(i==range),length(range));
    end
    close(pfig);
    [AUC balancePt] = ROC(probPerLabel,index.label,Labels{labelType},fullfile(DataDir,'Web'),labelSet,100,10);
end

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

