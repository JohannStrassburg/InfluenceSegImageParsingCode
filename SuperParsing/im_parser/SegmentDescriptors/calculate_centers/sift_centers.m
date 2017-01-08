function [ centers ] = sift_centers( fileList, HOMEIMAGES, dictionarySize)
%MAX_RESP_CENTERS Summary of this function goes here
%   Detailed explanation goes here
SIFTparam.grid_spacing = 1; 
SIFTparam.patch_size = 16;
%[uV sV] = memory;
%ndata_max = min(sV.PhysicalMemory.Available/512/10,100000); %use 10% avalible memory if its smaller than the default
ndata_max = 100000
numTextonImages = min(1000,length(fileList));
imIndexs = randperm(length(fileList));
fullresp = [];
for i = imIndexs(1:numTextonImages)
    filename = fullfile(HOMEIMAGES,fileList{i});
    im = imread(filename);
    data2add = sp_dense_sift(im,SIFTparam.grid_spacing,SIFTparam.patch_size);
    data2add = reshape(data2add,[size(data2add,1)*size(data2add,2) size(data2add,3)]);
    if(size(data2add,1)>ndata_max/numTextonImages )
        p = randperm(size(data2add,1));
        data2add = data2add(p(1:floor(ndata_max/numTextonImages)),:);
    end
    fullresp = [fullresp; data2add];
end

opts = statset('MaxIter',40,'Display','iter');
%% run kmeans
fprintf('\nRunning k-means\n');
[foo centers] = kmeans(fullresp, dictionarySize,'Options',opts);
