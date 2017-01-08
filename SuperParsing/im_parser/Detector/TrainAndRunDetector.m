function [ output_args ] = TrainAndRunDetector(HOMEIMAGES,HOMELABELSET,HOMEDATA,testFile,trainFileList,trainIndex,Labels,testParams)
%TRAINANDRUNDETECTOR Summary of this function goes here
%   Detailed explanation goes here


HOMELABELSET = strrep(HOMELABELSET,'\','/');
HOMEIMAGES = strrep(HOMEIMAGES,'\','/');
HOMEDATA = strrep(HOMEDATA,'\','/');
testFile = strrep(testFile,'\','/');
trainFileList = strrep(trainFileList,'\','/');

useTrainInd = unique(trainIndex.image);

dataset_params.datadir = HOMEDATA;
dataset_params.localdir = '';%fullfile(HOMEDATA,testParams.TestString);
dataset_params.display = 1;

detectorParams = esvm_get_default_params;
detectorParams.dataset_params = dataset_params;

cls = 'boat';

stream_params.stream_set_name = 'trainval';
stream_params.stream_max_ex = 200;
stream_params.must_have_seg = 0;
stream_params.must_have_seg_string = '';
stream_params.model_type = 'exemplar'; %must be scene or exemplar;
stream_params.cls = cls;

e_stream_set = GetDetectorStream(HOMEIMAGES,HOMELABELSET,trainFileList(useTrainInd),stream_params);

lind = find(strcmp(Labels,stream_params.cls));
imind = unique(trainIndex.image(trainIndex.label==lind));
[a b] = intersect(useTrainInd,imind);
negInd = useTrainInd;
negInd(b) = [];
neg_files = trainFileList(negInd);
neg_set = LoadImAndPyr(HOMEIMAGES,HOMEDATA,neg_files,detectorParams);
pos_files = trainFileList(useTrainInd(b));
pos_set = LoadImAndPyr(HOMEIMAGES,HOMEDATA,pos_files,detectorParams);
for i = 1:length(pos_files); pos_files{i} = fullfile(HOMEIMAGES,pos_files{i}); end

models_name = ...
    [stream_params.cls '-' detectorParams.init_params.init_type ...
     '.' detectorParams.model_type];

initial_models = esvm_initialize_exemplars(e_stream_set, detectorParams, models_name);

% Perform Exemplar-SVM training
train_params = detectorParams;
train_params.detect_max_scale = 0.5;
train_params.train_max_mined_images = 200;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 200;
%train_params.queue_mode  = 'cycle-violators';

% Train the exemplars and get updated models name
[models,models_name] = esvm_train_exemplars(initial_models, ...
                                            neg_set, train_params);


val_params = detectorParams;
val_params.detect_exemplar_nms_os_threshold = 0.5;
val_params.gt_function = @sp_load_gt_function;

val_set = pos_files;

% Apply trained exemplars on validation set
val_grid = esvm_detect_imageset(pos_set, models, val_params);

% Perform Platt calibration and M-matrix estimation
M = esvm_perform_calibration(val_grid, pos_set, models, ...
                             val_params);

% Define test-set
test_params = detectorParams;
test_set = LoadImAndPyr(HOMEIMAGES,HOMEDATA,{testFile},detectorParams);

% Apply on test set
test_grid = esvm_detect_imageset(test_set, models, test_params);

% Apply calibration matrix to test-set results
test_struct = esvm_pool_exemplar_dets(test_grid, models, M, test_params);

% Show top detections
maxk = 20;
allbbs = esvm_show_top_dets(test_struct, test_grid, test_set, models, ...
                       detectorParams,  maxk);


end

