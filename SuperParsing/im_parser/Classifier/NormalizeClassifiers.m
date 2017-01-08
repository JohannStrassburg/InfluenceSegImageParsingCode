function [B Bmulti] = NormalizeClassifiers(HOMETESTDATA,HOMELABELSETS,valFileList,valIndex,Labels,classifiers,valParams)

DataDir = fullfile(HOMETESTDATA,valParams.TestString);

close all;

B = cell(length(HOMELABELSETS),1);
Bmulti = cell(length(HOMELABELSETS),1);
for labelType=1:length(HOMELABELSETS)
	[foo labelSet] = fileparts(HOMELABELSETS{labelType});
    loadFile = fullfile(DataDir,'Normalizers',[labelSet 'Norm.mat']);
    if(exist(loadFile,'file'))
        load(loadFile);
    else
        index = valIndex{labelType};
        numLabels = length(Labels{labelType});
        numSP = length(index.sp);
        probPerLabel = zeros(numSP,numLabels);
        pfig = ProgressBar('Evaluating Classifiers');
        range = unique(index.image);
        for i = range(:)'

            [folder file] = fileparts(valFileList{i});
            baseFName = fullfile(folder,file);


            if(isempty(classifiers{labelType}))
                fprintf('No Classifiers for class: %s\n',labelSet);
            else
                probCacheFile = fullfile(DataDir,'ClassifierOutput',labelSet,[baseFName '.mat']);
                if(~exist(probCacheFile,'file'))
                    clear imSP valImSPDesc;
                    valImSPDesc = LoadSegmentDesc(valFileList(i),[],HOMETESTDATA,valParams.segmentDescriptors,valParams.K);
                    features = GetFeaturesForClassifier(valImSPDesc);
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
        Bls = zeros(2,numLabels);
        for i = 1:numLabels
            [clConfidenceMap Bls(:,i)] = MakeClassifierConfidenceMap(probPerLabel(:,i),index.label==i,.5,0);
        end
        if(nargout>1)
            Bmultils = mnrfit(probPerLabel,index.label);
        end
        save(loadFile,'Bls','Bmultils');
    end
    B{labelType} = Bls;
    Bmulti{labelType} = Bmultils;
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

