%function segments = SegmentVideo(imageFold,flowFolder)
imageFold = 'D:\im_parser\CamVid\ImagesAll\05';
flowFolder = 'D:\im_parser\CamVid\Data\Descriptors\OpticalFlow\05';

imageFiles = dir_recurse(fullfile(imageFold,'*'),0);
chunkSize = 30;
numFiles = length(imageFiles);
filerange = 1:numFiles; 
ranges = accumarray(floor((0:numel(filerange)-1)/chunkSize)'+1,filerange,[],@(x) {x});
%im1 = imread(fullfile(imageFold,imageFiles{1}));
%[ro co ch] = size(im1);
%blockList = cell(size(ranges));
%numBlocks = zeros(size(ranges));
%d = cell(size(ranges));
%{-
pfig2 = ProgressBar('Range Group');
for rndx = 120:length(ranges)
    range = ranges{rndx};
    cform = makecform('srgb2lab');
    pfig = ProgressBar('Loading Image... ');
    dataIm1 = zeros([ro co chunkSize 5 ],'double');
    for i = 1:length(range)
        im = imread(fullfile(imageFold,imageFiles{range(i)}));
        [fold base] = fileparts(imageFiles{range(i)});
        load(fullfile(flowFolder,fold,[base '.mat']));
        vx = vx-min(vx(:));%vx = vx/max(vx(:));
        vy = vy-min(vy(:));%vy = vy/max(vy(:));
        imHSV = applycform(im2double(im),cform);%im2double(im);%
        dataIm1(:,:,i,1:3) = imHSV./std(imHSV(:));
        dataIm1(:,:,i,4) = vx./std(vx(:));
        dataIm1(:,:,i,5) = vy./std(vy(:));
        ProgressBar(pfig,i,length(range));
    end
    close(pfig);drawnow;
    integralImSq = dataIm1.^2;
    integralIm =   padarray(cumsum(cumsum(cumsum(dataIm1,1),2),3),[1 1 1],'pre');
    integralImSq = padarray(cumsum(cumsum(cumsum(integralImSq,1),2),3),[1 1 1],'pre');

    bl = SplitVideoBlock(integralIm,integralImSq,dataIm1);
    blockList{rndx} = bl;
    numBlocks(rndx) = size(bl,1);
    
    if(rndx>1)
        dataIm = cat(3,dataIm2,dataIm1);
        integralImSq = dataIm.^2;
        integralIm =   padarray(cumsum(cumsum(cumsum(dataIm,1),2),3),[1 1 1],'pre');
        integralImSq = padarray(cumsum(cumsum(cumsum(integralImSq,1),2),3),[1 1 1],'pre');
        bl(:,[3 6]) = bl(:,[3 6])+chunkSize;
        bl = [blockList{rndx-1}; bl];
        d{rndx} = ClusterBlockList(bl,integralIm,integralImSq);
    end
    dataIm2 = dataIm1;
    ProgressBar(pfig2,rndx-109,length(ranges)-109);
end
%}
alld = sparse(sum(numBlocks),sum(numBlocks));
curndx = 1;
for i = 1:length(numBlocks)-1
    alld(curndx:curndx+numBlocks(i)+numBlocks(i+1)-1,curndx:curndx+numBlocks(i)+numBlocks(i+1)-1) = d{i+1};
    curndx = curndx + numBlocks(i);
end
%MakeSegmentVideo('D:\im_parser\CamVid\ImagesAll\05short','D:\im_parser\CamVid\Data\Descriptors\OpticalFlow\05','D:\im_parser\CamVid\Data\SegmentationTests\FlowSeg\05-block',200,L);