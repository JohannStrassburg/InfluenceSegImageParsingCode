HOME = 'D:\im_parser\LMSun';
HOME = '/lustre/scr/j/t/jtighe/LMSun';
HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'LabelsSemantic')};
HOMEDATA = fullfile(HOME,'Data');
HOMEDESCRIPTOR = fullfile(HOMEDATA,'Descriptors');
HOMECLASSIFIER = fullfile(HOMEDATA,'Classifier','Exemplar');
HOMEBUSY = fullfile(HOMECLASSIFIER,'Busy');

if(exist(HOMEBUSY,'file'))
    %rmdir(HOMEBUSY, 's');
end
trainFiles = fullfile(HOME,'trainSet1.mat');
load(trainFiles);
trainFileList = trainFileUnix;
trainFileListFull = cell(size(trainFileList));
for i = 1:length(trainFileList)
    trainFileListFull{i} = fullfile(HOMEIMAGES,trainFileList{i});
end
labelPresenceFile = fullfile(HOMECLASSIFIER,'labelPresence.mat');
load(labelPresenceFile);

testParams.globalDescriptors = {'spatialPryScaled','colorGist','coHist'};


dataset_params.datadir = HOMEDATA;
dataset_params.localdir = '';%fullfile(HOMEDATA,testParams.TestString);
dataset_params.display = 0;
detectorParams = esvm_get_default_params;
detectorParams.dataset_params = dataset_params;
detectorParams.preComputeHOG = false;

train_params = detectorParams;
train_params.detect_max_scale = 0.5;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 100;
train_params.ordering = 1:length(trainFileList);
%train_params.queue_mode  = 'cycle-violators';

val_params = detectorParams;
val_params.detect_exemplar_nms_os_threshold = 0.5;
val_params.SKIP_M = 1;
val_params.dataset_params.display = 0;
val_params.gt_function = @esvm_load_gt_function;

myRandomize;
maxModels = 400;
rp = randperm(length(trainFileList));
%rp = 1:length(trainFileList);
lastMaxMined = 0;
for max_mined = [20.*4.^(3:-1:0)]%0
train_params.train_max_mined_images = max_mined;
rpL = randperm(length(labels));
for l = rpL(:)'
    posOptions = find(labelPresenceMap(:,l));
    posSet = trainFileListFull(posOptions);
    posSetBase = trainFileList(posOptions);
    rpOps = posOptions(randperm(length(posOptions)));
    labelModels = [];
    labelFileNdx = [];
    modelNdxList = [];
    loadFile = fullfile(HOMEBUSY,[labels{l} '_loading']);
    while(exist(loadFile,'file'))
        pause(1+rand());
    end
    mkdir(loadFile);
    try
    for i = rpOps(:)'
        [fold base] = fileparts(trainFileList{i});
        caliFile = fullfile(HOMECLASSIFIER,'Calibration',['Model' num2str(max_mined)],labels{l},fold,[base '.mat']);make_dir(caliFile);
        if(exist(caliFile,'file'))
            continue;
        end
        busyFold = fullfile(HOMEBUSY,labels{l},fold,base);
        if(exist(busyFold,'file'))
            continue;
        end
        mkdir(busyFold);
        modelMMFile = fullfile(HOMECLASSIFIER,['Model' num2str(max_mined)],fold,[base '.mat']);make_dir(modelMMFile);
        if(exist(modelMMFile,'file'))
            load(modelMMFile);
        else
            continue;
        end
        models = AddPolyToModel(HOMEANNOTATIONS,models,modelMMFile);
        clsNums = cellfun(@(x)x.clsNum,models);
        labelModels = [labelModels; models(clsNums==l)];
        labelFileNdx = [labelFileNdx; i*ones(sum(clsNums==l),1)];
        modelNdxList = [modelNdxList; find(clsNums==l)];
        if(length(labelModels)>maxModels)
            break;
        end
    end
    catch
    end
    try
        rmdir(loadFile);
    catch
    end
    if(length(labelFileNdx)==0)
        continue;
    end
    try
    posSet = AddRecsToSet(HOMEANNOTATIONS,posSet,posSetBase,labels{l});
    pos_grid = esvm_detect_imageset(posSet, labelModels, val_params);
    M = esvm_perform_calibration(pos_grid, posSet, labelModels, val_params);
    for i = unique(labelFileNdx)'
        betas = M.betas(labelFileNdx==i,:);
        modelNdx = modelNdxList(labelFileNdx==i);
        [fold base] = fileparts(trainFileList{i});
        caliFile = fullfile(HOMECLASSIFIER,'Calibration',['Model' num2str(max_mined)],labels{l},fold,[base '.mat']);make_dir(caliFile);
        save(caliFile,'betas','modelNdx');
    end
    catch
    end
    for i = unique(labelFileNdx)'
    try
        rmdir(fullfile(HOMEBUSY,labels{l},fold,base));
    catch
    end
    end
end
end