function imSet = LoadImAndPyr(HOMEIMAGES,HOMEDATA,fileNames,detectorParams)
%LOADIMANDPYR Summary of this function goes here
%   Detailed explanation goes here

imSet = cell(size(fileNames));
for i = 1:length(fileNames)
    [fold, base] = fileparts(fileNames{i});
    imSet{i}.fName = fullfile(HOMEIMAGES,fileNames{i});
    imSet{i}.I = convert_to_I(imSet{i}.fName);
    saveFile = fullfile(HOMEDATA,'Descriptors','HOG',fold,[base '.mat']);make_dir(saveFile);
    if(exist(saveFile,'file'))
        load(saveFile);
    else
        t = cell(2,1);
        detectorParams.detect_add_flip = 0;
        t{1} = esvm_get_pyramid(imSet{i}.I, detectorParams);
        detectorParams.detect_add_flip = 1;
        t{2} = esvm_get_pyramid(imSet{i}.I, detectorParams);
        save(saveFile,'t');
    end
    imSet{i}.t = t;
end
end

