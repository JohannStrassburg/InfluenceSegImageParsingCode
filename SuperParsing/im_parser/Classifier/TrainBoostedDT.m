function [ classifier geoclassifier] = TrainBoostedDT( Dtrain, features, spIndexTrain, HOMEDATA, shortLabels, params, trainSemantic)
%TRAINBOOSTEDDT Summary of this function goes here
%   Detailed explanation goes here

if(~exist('trainSemantic','var'))
    trainSemantic = 1;
end
if(~exist('params','var'))
    params = [8 100 .001 1 1];
end

num_nodes = params(1);
num_iterations = params(2);
stopval = params(3);
weighted = params(4);
subSample = params(5);
testsetnum = params(6);

[outfile semTmp geoTmp] = GetClassifierFile(HOMEDATA,params);

if(exist(outfile,'file'))
    load(outfile);
    %return;
end

labelList = Dtrain(1).classNames;
numSP = 0; 
labels = cell(0);
init_weights = [];
for i = 1:length(spIndexTrain)
    numSP = numSP + length(spIndexTrain{i}.ImageSPIndex);
    labels = [labels; labelList(spIndexTrain{i}.SPLabelNum)'];
    init_weights = [init_weights; (1+spIndexTrain{i}.SPSize(:,1)).*(1+spIndexTrain{i}.SPSize(:,2))];
end

%features = GetFeaturesForClassifier(spDescTrain);

cat_features = [];
%classifier = [];
classifierpart = cell(length(labels),1);
if(trainSemantic)
    for i = 1:length(shortLabels)
        classifierpart{i} = train_boosted_dt_mc(features, cat_features, labels, num_iterations, num_nodes, stopval, init_weights, subSample, i, semTmp, shortLabels);
    end
    classifier = train_boosted_dt_mc(features, cat_features, labels, num_iterations, num_nodes, stopval, init_weights, subSample, [], semTmp, shortLabels);
end

%make_dir(outfile);save(outfile,'classifier');

labelList = Dtrain(1).geoNames(Dtrain(1).class2Geo);
geolabels = [];
for i = 1:length(spIndexTrain)
    geolabels = [geolabels; labelList(spIndexTrain{i}.SPLabelNum)'];
end
shortGeoLabels=Dtrain(1).geoNames(1:end-1);
for i = 1:length(shortGeoLabels)
    geoclassifierpart{i} = train_boosted_dt_mc(features, cat_features, geolabels, num_iterations, num_nodes, stopval, init_weights, subSample, i, geoTmp, shortGeoLabels);
end
geoclassifier = train_boosted_dt_mc(features, cat_features, geolabels, num_iterations, num_nodes, stopval, init_weights, subSample, [], geoTmp, shortGeoLabels);
make_dir(outfile);save(outfile,'classifier','geoclassifier');
end
