function [ output_args ] = MakeSegmentVideo( imageFold, flowFolder, outFolder, K, L )
%MAKESEGMENTVIDEO Summary of this function goes here
%   Detailed explanation goes here
close all;
pfig = ProgressBar('Segmenting');
imageFiles = dir_recurse(fullfile(imageFold,'*'),0);

if(exist('L','var'))
    cmap =jet(length(unique(L)));
    cmap = cmap(randperm(size(cmap,1)),:);
end

make_dir(fullfile(outFolder,'me.me'));
for i = 1:length(imageFiles)
    im = imread(fullfile(imageFold,imageFiles{i}));
    [fold base] = fileparts(imageFiles{i});
    if(~exist('L','var'))
        load(fullfile(flowFolder,fold,[base '.mat'])); % vx vy
        imSP = GenerateSuperPixels(im,K);
        [imSPFlow input]= GenerateSuperPixels(im,K,vx,vy);
        fprintf('%d %d - %d\n',max(imSP(:)),max(imSPFlow(:)),i);
        edIm = MakeEdgeImage(im,imSP);
        edImFlow  = MakeEdgeImage(im,imSPFlow);
        input(1,1,4:5) = max(max(max(input(:,:,4:5))));
        input(1,2,4:5) = min(min(min(input(:,:,4:5))));
        vx = im2uint8(grs2rgb(input(:,:,4),colormap('jet')));
        vy = im2uint8(grs2rgb(input(:,:,5),colormap('jet')));
        im = [edIm edImFlow;vx vy];
    else
        edIm = MakeEdgeImage(im,L(:,:,i));
        im = [edIm label2rgb(L(:,:,i),cmap)];
        imSPFlow = L(:,:,i);
    end
    show(im,1);
    imwrite(im,fullfile(outFolder,sprintf('frame%06d.png',i-1)));
    superPixels = imSPFlow;
    saveFile = fullfile(outFolder,fold,[base '.mat']); make_dir(saveFile);
    save(saveFile,'superPixels');
    ProgressBar(pfig,i,length(imageFiles));
end
close(pfig);
cmd = ['ffmpeg -r 10 -b 40000k -i "' fullfile(outFolder,'frame%06d.png') '" "' fullfile(outFolder,'segmentation.mp4') '"'];
system(cmd);
for i = 1:length(imageFiles)
    %delete(fullfile(outFolder,sprintf('frame%06d.png',i-1)));
end

function im = MakeEdgeImage(im,imSP)
    mask = ((imSP([1 1:end-1],:)-imSP(1:end,:))~=0)|((imSP(:,[1 1:end-1])-imSP(:,1:end))~=0);
    %mask = edge(imSP,'canny');
    c = 2;
    imG = im(:,:,c);
    imG(mask) = 255-imG(mask);
    im(:,:,c) = imG;
    c = 1;
    imG = im(:,:,c);
    imG(mask) = 255;
    im(:,:,c) = imG;
    