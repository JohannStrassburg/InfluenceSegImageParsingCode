function PreComputeHOG( HOMEIMAGES,HOMEDATA,fileNames )
detectorParams = esvm_get_default_params;
detectorParams.detect_add_flip = 0;
detectorParamsf = esvm_get_default_params;
detectorParams.detect_add_flip = 1;
block = 100;
pfig = ProgressBar('Pre-computing HOG');
for b = 1:block:length(fileNames)
    imSet = cell(size(fileNames));
    for i = b:min(b+block,length(fileNames))
        [fold, base] = fileparts(fileNames{i});
        saveFile = fullfile(HOMEDATA,'Descriptors','HOG',fold,[base '.mat']);make_dir(saveFile);
        if(exist(saveFile,'file'))
            continue;
        end
        imSet{i}.fName = fullfile(HOMEIMAGES,fileNames{i});
        imSet{i}.I = convert_to_I(imSet{i}.fName);
    end
    parfor i = b:min(b+block,length(fileNames))
        if(~isempty(imSet{i}))
            t = cell(2,1);
            t{1} = esvm_get_pyramid(imSet{i}.I, detectorParams);
            t{2} = esvm_get_pyramid(imSet{i}.I, detectorParamsf);
            imSet{i}.t = t;
        end
    end
    for i = b:min(b+block,length(fileNames))
        [fold, base] = fileparts(fileNames{i});
        saveFile = fullfile(HOMEDATA,'Descriptors','HOG',fold,[base '.mat']);make_dir(saveFile);
        if(exist(saveFile,'file'))
            continue;
        end
        t = imSet{i}.t;
        save(saveFile,'t');
    end
    ProgressBar(pfig,b,length(fileNames));
end

end

