HOME = 'D:\im_parser\LMSun';
%HOME = '/lustre/scr/j/t/jtighe/LMSun';
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
val_params.gt_function = @sp_load_gt_function;

myRandomize;
rp = randperm(length(trainFileList));
%rp = 1:length(trainFileList);
lastMaxMined = 0;
for max_mined = [20.*4.^(0:3)]%0
train_params.train_max_mined_images = max_mined;
for rpndx = 1:length(rp)
    i = rp(rpndx);
    [fold base] = fileparts(trainFileList{i});
    modelMMFile = fullfile(HOMECLASSIFIER,['Model' num2str(max_mined)],fold,[base '.mat']);make_dir(modelMMFile);
    if(exist(modelMMFile,'file'))
        continue;
    end
    busyFold = fullfile(HOMEBUSY,fold,base);
    if(exist(busyFold,'file'))
        continue;
    end
    mkdir(busyFold);
    try
    modelFile = fullfile(HOMECLASSIFIER,'FullModel',fold,[base '.mat']);make_dir(modelFile);
    if(exist(modelFile,'file'))
        load(modelFile);
    else
        initFile = fullfile(HOMECLASSIFIER,'InitModel',fold,[base '.mat']);make_dir(initFile);
        if(exist(initFile,'file'))
            load(initFile);
        else
            e_stream_set = GetDetectorStreamForSingleLM(HOMEIMAGES, HOMEDATA, fullfile(HOMEANNOTATIONS,fold,[base '.xml']), detectorParams);
            %e_stream_set = e_stream_set(1:2);
            models = esvm_initialize_exemplars(e_stream_set, detectorParams, '');
            save(initFile,'models');
        end
    end

    [retInds rank] = FindRetrievalSet([],[],fullfile(HOMECLASSIFIER),fullfile(fold,base),testParams,'');
    retInds(retInds==i)=[];

    timing = zeros(size(models));
    for mNdx = 1:length(models)
        m = models{mNdx};
        negOptions = retInds(~labelPresenceMap(retInds,m.clsNum));
        %posOptions = retInds(labelPresenceMap(retInds,m.clsNum));
        statMiningNdx = 1;
        if(isfield(m,'total_mines'))
            statMiningNdx = m.total_mines+1;
        end
        negSet = trainFileListFull(negOptions(statMiningNdx:max_mined));
        tic
        [models(mNdx),models_name] = esvm_train_exemplars({m}, negSet, train_params);
        timing(mNdx) = toc;
    end
    %fprintf('Timing:\n');
    %fprintf('%.2f\n',timing);
    
    save(modelFile,'models');
    models = esvm_strip_models(models);
    save(modelMMFile,'models');
    clear models;
    catch
    end
    try
    rmdir(busyFold);
    catch
    end
end
lastMaxMined = max_mined;
end



