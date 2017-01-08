function [ desc_all ] = BuildLabelHist( imageFileList, labelBaseDirs, dataBaseDir, canSkip )

close all;
if(nargin<4)
    canSkip = 0;
end
desc_all = cell(length(labelBaseDirs),1);
for ls = 1:length(labelBaseDirs)
    labelBaseDir = labelBaseDirs{ls};
    [foo labelSet] = fileparts(labelBaseDir);
    tic
    pfig = ProgressBar('Building Label Histogram');
    for f = 1:length(imageFileList)
        % load image
        imageFName = imageFileList{f};
        [dirN base] = fileparts(imageFName);
        baseFName = [dirN filesep base];
        outFName = fullfile(dataBaseDir,labelSet,sprintf('%s.mat',  baseFName));
        labelFName = fullfile(labelBaseDir, sprintf('%s.mat', baseFName));
        if(mod(f,100)==0)
            ProgressBar(pfig,f,length(imageFileList));
        end
        if(exist(outFName,'file') && canSkip)
            %fprintf('Skipping file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
            load(outFName);
            if(isempty(desc_all{ls}))
                desc_all{ls} = zeros(length(imageFileList),length(labelHist));
            end
            desc_all{ls}(f,:) = labelHist;
            continue;
        end

        try
            load(labelFName);
            if(~exist('S','var')&&exist('L','var'))
                S = L;
                names = labelList;
            end
            [l counts] = UniqueAndCounts(S);
            %counts(l<1) = [];
            %l(l<1) = [];
            labelHist = zeros(1,length(names)+1);
            labelHist(l+1) = counts;
            %labelHist = labelHist./sum(labelHist);
            make_dir(outFName);
            save(outFName, 'labelHist');
            
            if(isempty(desc_all{ls}))
                desc_all{ls} = zeros(length(imageFileList),length(labelHist));
            end
            desc_all{ls}(f,:) = labelHist;
        catch ME
            ME
        end
    end
    close(pfig);
end
end
