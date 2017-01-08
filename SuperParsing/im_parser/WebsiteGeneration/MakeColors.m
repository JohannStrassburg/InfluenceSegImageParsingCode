%test lab
%{
step = 32;
range = [0:step:255 255];
np = length(range);
X = repmat(range',[1 np np]);
Y = repmat(range, [np 1 np]);
Z = repmat(reshape(range, [1 1 np]),[np np 1]);
allrgb = uint8([X(:) Y(:) Z(:)]);
cform = makecform('srgb2lab');
alllab = applycform(im2double(allrgb),cform);
cform = makecform('lab2srgb');
%allrgb = 
figure(1);scatter3(allrgb(:,1),allrgb(:,2),allrgb(:,3),10,1:size(allrgb,1),'filled');
colormap(im2double(allrgb));
figure(2);scatter3(alllab(:,1),alllab(:,2),alllab(:,3),10,1:size(allrgb,1),'filled');
colormap(im2double(allrgb));

step = 20;
range1 = 0:step:100;
range2 = -100:step*2:100;
np = length(range1);
X = repmat(range1',[1 np np]);
Y = repmat(range2, [np 1 np]);
Z = repmat(reshape(range2, [1 1 np]),[np np 1]);
alllab = [X(:) Y(:) Z(:)];
cform = makecform('lab2srgb');
allrgb = applycform(alllab,cform);
figure(1);scatter3(allrgb(:,1),allrgb(:,2),allrgb(:,3),10,1:size(allrgb,1),'filled');
colormap(allrgb);
figure(2);scatter3(alllab(:,1),alllab(:,2),alllab(:,3),10,1:size(allrgb,1),'filled');
colormap(allrgb);

allrgb = [];
alllab = [];
cform = makecform('lab2srgb');
while(length(allrgb)<232)
    samplePt = [rand([1000 1])*100 rand([1000 2])*200-100];
    samplergb = applycform(samplePt,cform);
    good = sum(samplergb>.01,2)==3 & sum(samplergb<.99,2)==3;
    allrgb = [allrgb; samplergb(good,:)];
    alllab = [alllab; samplePt(good,:)];
end
colormap(allrgb);
figure(1);scatter3(allrgb(:,1),allrgb(:,2),allrgb(:,3),10,1:size(allrgb,1),'filled');
colormap(allrgb);
figure(2);scatter3(alllab(:,1),alllab(:,2),alllab(:,3),10,1:size(allrgb,1),'filled');
%}

labelColors = cell(size(HOMELABELSETS));
for ls = 2:length(HOMELABELSETS)
    [foo setname] = fileparts(HOMELABELSETS{ls});
    saveFile = fullfile(HOME,[setname 'Colors_Gen.mat']);
    if(1)%~exist(saveFile,'file'))
        %{
        numLabels =length(Labels{ls});
        trainLabels = trainIndex{ls}.label;
        [lables, cts] = UniqueAndCounts(trainLabels);
        lcounts = zeros(numLabels,1);
        lcounts(lables) = cts;
        meanColors = LoadSegmentDesc(trainFileList,trainIndex{ls},HOMEDATA,{'mean_color'},testParams.K,testParams.segSuffix);
        meanColors = meanColors.mean_color;
        cform = makecform('srgb2lab');
        meanColorsLab =  applycform(im2double(uint8(meanColors)),cform);
        for l = 1:numLabels
            pts = meanColorsLab(trainLabels == l,:);
            if(size(pts,1)<4)
                if(isempty(pts))
                    pts = [50 0 0];
                end
                while(size(pts,1)<4)
                    pts = [pts; pts(1,:)+(rand([1 3])-.5)*10];
                end
            end
            m(l,:) = mean(pts);
            c = cov(pts);
            [a b] = chol(c);
            if(length(a)<length(c))
                fprintf('bad covariance\n');
                keyboard;
            end
            gm{l} = gmdistribution(m(l,:),c);
        end
        
        rgboptions = [];
        laboptions = [];
        cform = makecform('lab2srgb');
        while(length(laboptions)<2*numLabels)
            samplePt = [rand([1000 1])*100 rand([1000 2])*200-100];
            samplergb = applycform(samplePt,cform);
            good = sum(samplergb>.01,2)==3 & sum(samplergb<.99,2)==3;
            rgboptions = [rgboptions; samplergb(good,:)];
            laboptions = [laboptions; samplePt(good,:)];
        end
        %}
        
        laboptionst = laboptions;
        rgboptionst = rgboptions;
        [~, sortedls] = sort(lcounts,'descend');
        labelColor = zeros(numLabels,3);
        assigned = [];
        for l = sortedls(:)'
            d = dist2(m(l,:),laboptionst);
            d2 = dist2(laboptionst,labelColor(assigned,:));
            p = ones(size(d));
            if(size(d2,2)>0)
                d2 = mean(d2,2);
                p(:) = max(d2)-d2;
            end
            d2 = d./mean(d) + p./mean(p);
            [a1 b1] = min(d2);
            [a b] = min(d2);
            labelColor(l,:) = rgboptionst(b,:);
            laboptionst(b,:) = [];
            rgboptionst(b,:) = [];
            assigned = [assigned; l];
        end
        
        %{
        bestp = 0;
        bestlabeling = 1:numLabels;
        bestpw = 0;
        bestlabelingw = 1:numLabels;
        pfig = ProgressBar('Finding best fit');
        for i = 1:10000
            labeling = randperm(numLabels);
            p = zeros(numLabels,1);
            for l = 1:numLabels;
                p(l) = pdf(gm{l},laboptions(labeling(l),:));
            end
            tp = sum(p);
            tpw = sum(p.*lcounts)./sum(lcounts);
            if(tp>bestp)
                bestp = tp;
                bestlabeling = labeling;
            end
            if(tpw>bestpw)
                bestpw = tpw;
                bestlabelingw = labeling;
            end
            if(mod(i,100)==0)
                ProgressBar(pfig,i,1000);
            end
        end
        %}
        labelColor = rgboptions(bestlabelingw,:);
        save(saveFile,'labelColor');
    else
        load(saveFile);
    end
    labelColors{ls} = labelColor;
end


for k = 1:length(WEBLABELSETS)
    [foo setname] = fileparts(WEBLABELSETS{k});
    saveFile = fullfile(HOME,[setname 'Colors.mat']);
    labelColor = labelColors{k};
    if(~exist(saveFile,'file'))
        save(saveFile,'labelColor');
    end
end