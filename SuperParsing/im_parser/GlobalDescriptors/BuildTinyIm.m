function [ desc_all ] = BuildTinyIm( imageFileList, imageBaseDir, dataBaseDir, canSkip )
%function [ gist_all ] = BuildGist( imageFileList, imageBaseDir,dataBaseDir, canSkip )
%BUILDGIST Summary of this function goes here
%   Detailed explanation goes here



if(nargin<4)
    canSkip = 0;
end
I = zeros(256,256,3);
tinyIm = imresize(I,[16 16]); tinyIm = tinyIm(:);
desc_all = zeros(length(imageFileList),length(tinyIm));

tic
pfig = ProgressBar('Building Tiny Image');
for f = 1:length(imageFileList)
    % load image
    imageFName = imageFileList{f};
    [dirN base] = fileparts(imageFName);
    baseFName = [dirN filesep base];
    outFName = fullfile(dataBaseDir,sprintf('%s_tinyIm.mat',  baseFName));
    imageFName = fullfile(imageBaseDir, sprintf('%s', imageFName));
    
    
    if(mod(f,100)==0)
        ProgressBar(pfig,f,length(imageFileList));
    end
    if(size(dir(outFName),1)~=0 && canSkip)
        %fprintf('Skipping file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
        load(outFName);
        desc_all(f,:) = tinyIm(:)';
        continue;
    end
    
    try
        I = imread(imageFName);
        if(size(I,3)==1)
        	I = repmat(I,[1 1 3]);
        end
        %fprintf('Working on file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
        %I = im2double(I);

        tinyIm = imresize(I,[16 16]); 
        
        make_dir(outFName);
        save(outFName, 'tinyIm');

        desc_all(f,:) = tinyIm(:)';
    catch ME
        ME
    end
end
close(pfig);
end
