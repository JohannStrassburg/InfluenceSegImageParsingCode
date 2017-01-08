function [ gist_all ] = BuildGist( imageFileList, imageBaseDir, dataBaseDir, canSkip, color )
%function [ gist_all ] = BuildGist( imageFileList, imageBaseDir,dataBaseDir, canSkip )
%BUILDGIST Summary of this function goes here
%   Detailed explanation goes here


%% Parameters:
imageSize = 256; 
orientationsPerScale = [8 8 4];
numberBlocks = 4;

if(~exist('color','var'))
    color=0;
end
if(~exist('canSkip','var'))
    canSkip = 0;
end

%% Precompute filter transfert functions (only need to do this one, unless image size is changes):
%createGabor(orientationsPerScale, imageSize); % this shows the filters
G = createGabor(orientationsPerScale, imageSize);

if(color)
    I = zeros([imageSize imageSize 3]);
else
    I = zeros(imageSize,imageSize);
end
output = prefilt(I,4);
gist = gistGabor(output,numberBlocks,G);
gist_all = zeros(length(imageFileList),size(gist,1));

tic
pfig = ProgressBar('Building Gist');
for f = 1:length(imageFileList)
    %% load image
    imageFName = imageFileList{f};
    [dirN base] = fileparts(imageFName);
    baseFName = [dirN filesep base];
    outFName = fullfile(dataBaseDir,sprintf('%s_gist.txt',  baseFName));
    imageFName = fullfile(imageBaseDir, sprintf('%s', imageFName));
    
    if(mod(f,100)==0)
        ProgressBar(pfig,f,length(imageFileList));
    end
    if(size(dir(outFName),1)~=0 && canSkip)
        %fprintf('Skipping file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
        gist = load(outFName, '-ascii');
        gist_all(f,:) = gist;
        continue;
    end
    
    %try
        I = load_image(imageFName,color);
        I = imresizecrop(I,[imageSize imageSize], 'bicubic');
        %fprintf('Working on file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
        %I = im2double(I);

        output = prefilt(I,4);
        gist = gistGabor(output,numberBlocks,G);

        make_dir(outFName);
        save(outFName, 'gist', '-ascii');

        gist_all(f,:) = gist;
    %catch ME
    %    ME
    %end
end
close(pfig);

end
