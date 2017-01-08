
labelSetNums = [3 4 5];
fileList = testFileList;


%{-
labelColorsIm = cell(max(labelSetNums),1);
weightsIm = cell(max(labelSetNums),1);
for ls = labelSetNums(:)'
    labelColorsIm{ls} = zeros(length(Labels{ls}),3,length(fileList));
    weightsIm{ls} = zeros(length(Labels{ls}),length(fileList));
end
pfig = ProgressBar('Computing Colors');
for i = 1:length(fileList)
    im = imread(fullfile(HOME,'Images',fileList{i}));
    [ro co ch] = size(im);
    if(ch~=3); continue;end
    im = reshape(im,[ro*co ch]);
    [folder file ext] = fileparts(fileList{i});
    for ls = labelSetNums(:)'
        load(fullfile(HOMELABELSETS{ls},folder,[file '.mat']))
        [labels lscount] = UniqueAndCounts(S);
        lscount(labels==0) = [];
        labels(labels==0) = [];
        for l = labels(:)'
            mask = S==l;
            labelColorsIm{ls}(l,:,i) = mean(im(mask(:),:),1);
            weightsIm{ls}(l,i) = lscount(labels==l);
        end
    end
    if(mod(i,100)==0)
        ProgressBar(pfig,i,length(fileList));
    end
end
close(pfig);

for ls = labelSetNums(:)'
    wm = repmat(reshape(weightsIm{ls},[size(weightsIm{ls},1) 1 size(weightsIm{ls},2)]),[1 3 1]);
    labelColors{ls} = sum(labelColorsIm{ls}.*wm,3)./sum(wm,3);
    labelColors{ls}(isnan(labelColors{ls})) = 127;
    labelColors{ls} = labelColors{ls}./255;
    labelColors{ls} = rgb2hsv(labelColors{ls});
    labelColors{ls}(:,2) = 1;
    %labelColors{ls}(:,3) = 1;
    labelColors{ls} = hsv2rgb(labelColors{ls});
end

%}
copyFrom = 5;
copyTo = 5;

for l = 1:length(Labels{copyFrom})
    ndx = find(strcmp(Labels{copyFrom}{l},Labels{copyTo}));
    if(~isempty(ndx));labelColors{copyTo}(ndx,:) = labelColors{copyFrom}(l,:);end
end