
ls = 2;
if(~exist('cls','var'))
    cls = 'boat';
end
try for i = 3:100;fclose(i);fprintf('closed %d\n',i);end;catch ERR;end

if(~exist('classNDX','var') || ~strcmp(Dclass(1).annotation.object(1).name,cls))
    [Dclass classNDX] =  LMquery(D,'object.namendx',num2str(find(strcmp(cls,Labels{ls}))),'exact');
    %[Dclass classNDX] = LMquery(D,'object.name',cls);
    DclassAll = D(classNDX);
end

bb = cell(length(Dclass),1);
imNum = cell(length(Dclass),1);
for i = 1:length(Dclass)
    bb{i} = LMobjectboundingbox(Dclass(i).annotation,cls);
    imNum{i} = ones(size(bb{i},1),1)*i;
end

HOMEWEB = fullfile(HOMEDATA,testParams.TestString,['Web_' cls]);
indexList = fullfile(HOMEWEB,'index.mat');make_dir(indexList);
if(exist(indexList,'file'))
    load(indexList);
else
    bb = cell2mat(bb);
    imNum = cell2mat(imNum);
    bbsize = (bb(:,3)-bb(:,1)).*(bb(:,4)-bb(:,2));
    [bbsize b] = sort(bbsize,'descend');
    b = b(100:end);
    bbsize = bbsize(100:end);
    rp =randperm(length(b));
    rp = [1:5 rp];
    bbsize = bbsize(rp);
    imNum = imNum(b(rp));
    save(indexList,'imNum','bbsize','rp');
end
    


testParams.retSetSize = 2000;
dataset_params.datadir = HOMEDATA;
dataset_params.localdir = '';%fullfile(HOMEDATA,testParams.TestString);
dataset_params.display = 0;
detectorParams = esvm_get_default_params;
detectorParams.dataset_params = dataset_params;

stream_params.stream_set_name = 'trainval';
stream_params.stream_max_ex = 200;
stream_params.must_have_seg = 0;
stream_params.must_have_seg_string = '';
stream_params.model_type = 'exemplar'; %must be scene or exemplar;
stream_params.cls = cls;

train_params = detectorParams;
train_params.detect_max_scale = 0.5;
train_params.train_max_mined_images = max_mined;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 100;
%train_params.queue_mode  = 'cycle-violators';

val_params = detectorParams;
val_params.detect_exemplar_nms_os_threshold = 0.5;
val_params.gt_function = @sp_load_gt_function;

HOMEWEB = fullfile(HOMEDATA,testParams.TestString,sprintf('Web_%s_%03d',cls,train_params.train_max_mined_images));
webIndex = fullfile(HOMEWEB,sprintf('index%03d.html',train_params.train_max_mined_images));make_dir(webIndex);
indexFID = fopen(webIndex,'w');
%fprintf(indexFID,'<HTML>\n<HEAD>\n<TITLE>Detector Results</TITLE>\n</HEAD>\n<BODY>');
fprintf(indexFID,'\n<center>\n');
fprintf(indexFID,'\n<table border="0">\n');
close all;
count = 0;maxIms = 10;
while(~isempty(imNum))
    i = imNum(1);
    imFName = LMfilename(DclassAll(i).annotation);
    %LMplot(DclassAll, i, HOMEIMAGES);
    queryHOG = LoadImAndPyr(HOMEIMAGES,HOMEDATA,fileList(classNDX(i)),detectorParams);
    im = queryHOG{1}.I;    
    
    [fold base ext] = fileparts(imFName);
    
    modelsFile = fullfile(HOMEWEB,'Model',fold,sprintf('%s-%02d.mat',base,train_params.train_max_mined_images));make_dir(modelsFile);
    if(exist(modelsFile,'file'))
        load(modelsFile);
    else
    tic;
        [retInds rank]= FindRetrievalSet(GlobalDesc,SelectDesc(GlobalDesc,classNDX(i),1),fullfile(HOMEDATA ,testParams.TestString),fullfile(fold,base),testParams,'');
        retInd = retInds(2:testParams.retSetSize+1);
        [Dret posNdx] = LMquery(D(retInd),'object.namendx',num2str(find(strcmp(cls,Labels{ls}))),'exact');
        e_stream_set = GetDetectorStreamLM(HOMEIMAGES, Dret, stream_params);
        
        pos_set = LoadImAndPyr(HOMEIMAGES,HOMEDATA,fileList(retInd(posNdx)),detectorParams);
        negNdx = 1:length(retInd);
        negNdx(posNdx) = [];
        negNdx = negNdx(1:train_params.train_max_mined_images);
        neg_set = LoadImAndPyr(HOMEIMAGES,HOMEDATA,fileList(retInd(negNdx)),detectorParams);

        models_name = [stream_params.cls '-' detectorParams.init_params.init_type '.' detectorParams.model_type];
        initial_models = esvm_initialize_exemplars(e_stream_set, detectorParams, models_name);

        [models,models_name] = esvm_train_exemplars(initial_models, neg_set, train_params);
        timeE = toc
        save(modelsFile,'models','timeE');
    end
    
    detectorParams.do_nms = 0;
    test_grid = esvm_detect_imageset(queryHOG, models, detectorParams);
    test_struct = esvm_pool_exemplar_dets(test_grid, models,[], detectorParams);
    dataTerm = ProjectDetectorResponses(im,test_struct.unclipped_boxes{1},models);
    
    fprintf(indexFID,'\t<tr>\n');
    imFile = fullfile(HOMEWEB,'Images',fold,[base '.jpg']);make_dir(imFile);
    clf;show(im,2);drawnow;
    LMplot(DclassAll, i, HOMEIMAGES);drawnow;
    SaveCurrentFigure(imFile,'-djpeg');
    fprintf(indexFID,'\t\t<td><a href="%s"><img src="%s"></a><br>',['Detections/' fold '/' base '.html'],['Images/' fold '/' base '.jpg']);
    fprintf(indexFID,'<center>%d models took %.2f seconds.</center></td>\n',length(models),timeE);
    show(dataTerm,2);drawnow;
    show(dataTerm,2);drawnow;
    outFile = fullfile(HOMEWEB,'Maps',fold,[base '.jpg']);make_dir(outFile);
    SaveCurrentFigure(outFile,'-djpeg');
    fprintf(indexFID,'\t\t<td><img src="%s"></td>\n',['Maps/' fold '/' base '.jpg']);
    fprintf(indexFID,'\t</tr>\n');
    
    detectionPath = fullfile(HOMEWEB,'Detections',fold,base);make_dir(detectionPath);
    maxk = min(200,size(test_struct.unclipped_boxes{1},1));
    figure(2);
    allbbs = esvm_show_top_dets(test_struct, test_grid, queryHOG, models, detectorParams,  maxk, '', detectionPath);
    
    imFile = fullfile(HOMEWEB,'Detections',fold,[base '.html']);make_dir(imFile);
    fid = fopen(imFile,'w');
    fprintf(fid,'<HTML>\n<HEAD>\n<TITLE>%s/%s</TITLE>\n</HEAD>\n<BODY>',fold,base);
    fprintf(fid,'\n<center>\n');
    for k = 1:maxk 
        fprintf(fid,'\t<img src="%s%02d.jpg"><br>',base, k );
    end
    fprintf(fid,'\n</center>\n');
    fclose(fid);

    
    bbsize(imNum==i) = [];
    imNum(imNum==i) = [];
    count = count+1;
    if(count>maxIms)
        break;
    end
end

fprintf(indexFID,'\n</table>\n');
fprintf(indexFID,'\n</center>\n');
fprintf(indexFID,'\n</body>\n');
fclose(indexFID);