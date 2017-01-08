
if(~exist('trainOnly','var'))
    trainOnly = false;
end

ls = 2;
[foo lsName] = fileparts(HOMELABELSETS{ls});
HOMEDATATEST =  fullfile(HOMEDATA,testParams.TestString);
labelColors = GetColors(HOME, HOMECODE, HOMELABELSETS, Labels);
try for i = 3:100;fclose(i);fprintf('closed %d\n',i);end;catch ERR;end

testParams.retSetSize = 500;
dataset_params.datadir = HOMEDATA;
dataset_params.localdir = '';%fullfile(HOMEDATA,testParams.TestString);
dataset_params.display = 0;
detectorParams = esvm_get_default_params;
detectorParams.dataset_params = dataset_params;

stream_params.stream_set_name = 'trainval';
stream_params.stream_max_ex = 100;
stream_params.must_have_seg = 0;
stream_params.must_have_seg_string = '';
stream_params.model_type = 'exemplar'; %must be scene or exemplar;
stream_params.cls = '';

train_params = detectorParams;
train_params.detect_max_scale = 0.5;
train_params.train_max_mined_images = max_mined;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 100;
%train_params.queue_mode  = 'cycle-violators';

val_params = detectorParams;
val_params.detect_exemplar_nms_os_threshold = 0.5;
val_params.gt_function = @sp_load_gt_function;


if(~trainOnly)
    HOMEWEB = fullfile(HOMEDATATEST,sprintf('WebDetector_%03d',train_params.train_max_mined_images));
    webIndex = fullfile(HOMEWEB,'index.html');make_dir(webIndex);
    indexFID = fopen(webIndex,'w');
    fprintf(indexFID,'<HTML>\n<HEAD>\n<TITLE>Detector Results</TITLE>\n</HEAD>\n<BODY>');
    fprintf(indexFID,'\n<center>\n');
    fprintf(indexFID,'\n<table border="0">\n');
end
    
rpIm = randperm(length(testFileList));
if(~trainOnly)
    rpIm = 1:length(testFileList);
end
for n = 1:length(testFileList)
    i = rpIm(n);
    imFName = testFileList{i};
    queryHOG = LoadImAndPyr(HOMEIMAGES,HOMEDATA,{imFName},detectorParams);
    im = queryHOG{1}.I;    
    [ro co ch] = size(im);
    
    [fold base ext] = fileparts(imFName);
    dataTerm = zeros([ro co length(Labels{ls})]);
    
    paramsstr = sprintf('MM%03d-RS%04d-ME%03d',train_params.train_max_mined_images,testParams.retSetSize,stream_params.stream_max_ex);
    dataTermFile = fullfile(HOMEDATATEST,'DataTerm',paramsstr,fold,sprintf('%s.mat',base));make_dir(dataTermFile);
    if(exist(dataTermFile,'file'))
        load(dataTermFile);
    else
        %continue;
        [retInds rank]= FindRetrievalSet(trainGlobalDesc,SelectDesc(testGlobalDesc,i,1),fullfile(HOMEDATA ,testParams.TestString),fullfile(fold,base),testParams,'');
        retInd = retInds(1:testParams.retSetSize);
        %allPyr = LoadImAndPyr(HOMEIMAGES,HOMEDATA,trainFileList(retInd),detectorParams);
        models = cell(0);
        pfig = ProgressBar('Training Detectors');
        myRandomize;
        rp = randperm(length(Labels{ls}));
        %rp = sort(rp);
        for rpndx = 1:length(Labels{ls})
            try
            l = rp(rpndx);
            stream_params.cls = Labels{ls}{l};
            oneModelFile = fullfile(HOMEDATATEST,'Model',paramsstr,fold,sprintf('%s-%s.mat',base,stream_params.cls));make_dir(oneModelFile);
            clear model;
            if(exist(oneModelFile,'file'))
                try
                    load(oneModelFile);
                catch
                    delete(oneModelFile);
                end
            end
            if(~exist('model','var'))
                oneModelBusyFile = fullfile(HOMEDATATEST,'Busy',paramsstr,fold,sprintf('%s-%s',base,stream_params.cls));
                if(trainOnly && exist(oneModelBusyFile,'file'))
                    continue;
                end
                mkdir(oneModelBusyFile);
                [Dret posNdx] = LMquery(Dtrain(retInd),'object.namendx',num2str(l),'exact');
                e_stream_set = GetDetectorStreamLM(HOMEIMAGES, Dret, stream_params);
                %pos_set = allPyr(posNdx);
                negNdx = 1:length(retInd); negNdx(posNdx) = [];
                negNdx = negNdx(1:train_params.train_max_mined_images);
                %neg_set = allPyr(negNdx);
                neg_set = LoadImAndPyr(HOMEIMAGES,HOMEDATA,trainFileList(retInd(negNdx)),detectorParams);
                models_name = [stream_params.cls '-' detectorParams.init_params.init_type '.' detectorParams.model_type '.' num2str(max_mined)];
                initial_models = esvm_initialize_exemplars(e_stream_set, detectorParams, models_name);
                [model,models_name] = esvm_train_exemplars(initial_models, neg_set, train_params);
                save(oneModelFile,'model');
                rmdir(oneModelBusyFile);
            end
            catch
                if(exist(oneModelBusyFile,'file'))
                    rmdir(oneModelBusyFile);
                end
            end
            
            if(~trainOnly)
                detectorParams.do_nms = 0;
                test_grid = esvm_detect_imageset(queryHOG, model, detectorParams);
                test_struct = esvm_pool_exemplar_dets(test_grid, model,[], detectorParams);
                for b = 1:length(test_struct.unclipped_boxes)
                    dataTerm(:,:,l) = dataTerm(:,:,l) + ProjectDetectorResponses(im,test_struct.final_boxes{b},model);
                end
            end
            ProgressBar(pfig,rpndx,length(Labels{ls}));
        end
        close(pfig);
        if(~trainOnly)
            save(dataTermFile,'dataTerm');
        end
    end
    
    if(~trainOnly)
        fprintf(indexFID,'<tr>\n');
        fprintf(indexFID,'\t\t<td><img src="%s"></td>\n',['Images/' fold '/' base '.jpg']);
        
        load(fullfile(HOMELABELSETS{ls},fold,[base '.mat']));
        dispFile = fullfile(HOMEWEB,'GroundTruth',fold,[base '.png']);make_dir(dispFile);
        [imLabeled] = DrawImLabels(im,S+1,[0 0 0; labelColors{ls}],{'unlabeled' names{:}},dispFile,128,0,1,800);
        fprintf(indexFID,'\t\t<td><img src="%s"></td>\n',['GroundTruth/' lsName '/' fold '/' base '.png']);
        
        [v L] = max(dataTerm,[],3);
        [perPixStats perLabelStats] = EvalPixelLabeling(L,Labels{ls},S,names);
        plrate = perLabelStats(:,1)./perLabelStats(:,2);plrate(isnan(plrate)) = [];
        dispFile = fullfile(HOMEWEB,'Detector',fold,[base '.png']);make_dir(dispFile);
        [imLabeled] = DrawImLabels(im,L,labelColors{ls},Labels{ls},dispFile,128,0,1,800);
        fprintf(indexFID,'\t\t<td><img src="%s"><br>Detector Results: %.2f (%.2f)</td>\n',['Detector/' fold '/' base '.png'],perPixStats(1)/perPixStats(2),mean(plrate));

        parsingRFile = fullfile(HOMEDATATEST,lsName,'probPerLabelR200K200TNN80-SPscGistCoHist-sc01ratio',fold,[base '.mat']);
        if(exist(parsingRFile,'file'))
            
        load(fullfile(HOMEDATATEST,lsName,'probPerLabelR200K200TNN80-SPscGistCoHist-sc01ratio',fold,[base '.mat']));
        load(fullfile(HOMEDATA,'Descriptors','SP_Desc_k200','super_pixels',fold,[base '.mat']));

        pdataTerm = reshape(probPerLabel(superPixels,:),[ro co size(probPerLabel,2)]);;
        [v L] = max(pdataTerm,[],3);
        [perPixStats perLabelStats] = EvalPixelLabeling(L,Labels{ls},S,names);
        plrate = perLabelStats(:,1)./perLabelStats(:,2);plrate(isnan(plrate)) = [];
        dispFile = fullfile(HOMEWEB,'Parser',fold,[base '.png']);make_dir(dispFile);
        [imLabeled] = DrawImLabels(im,L,labelColors{ls},Labels{ls},dispFile,128,0,1,800);
        fprintf(indexFID,'\t\t<td><img src="%s"><br>Parser Results: %.2f (%.2f)</td>\n',['Parser/' fold '/' base '.png'],perPixStats(1)/perPixStats(2),mean(plrate));

        for w = 2.^(-2:2)
            [v L] = max(pdataTerm./100+dataTerm.*w,[],3);
            [perPixStats perLabelStats] = EvalPixelLabeling(L,Labels{ls},S,names);
            plrate = perLabelStats(:,1)./perLabelStats(:,2);plrate(isnan(plrate)) = [];
            dfname = sprintf('%s%.3f.png',base,w);
            dispFile = fullfile(HOMEWEB,'Combo',fold,dfname);make_dir(dispFile);
            [imLabeled] = DrawImLabels(im,L,labelColors{ls},Labels{ls},dispFile,128,0,1,800);
            fprintf(indexFID,'\t\t<td><img src="%s"><br>Combo %.3f: %.2f (%.2f)</td>\n',['Combo/' fold '/' dfname],w,perPixStats(1)/perPixStats(2),mean(plrate));
        end
        end
        fprintf(indexFID,'<\tr>\n');
    end
end

if(~trainOnly)
    fprintf(indexFID,'\n</table>\n');
    fprintf(indexFID,'\n</center>\n');
    fprintf(indexFID,'\n</body>\n');
    fclose(indexFID);
end



















%{
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
%}