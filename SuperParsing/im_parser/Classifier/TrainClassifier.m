function [classifiers] = TrainClassifier(HOMEDATA, HOMELABELSETS, fileList, indexs, Labels, labelMask, params)
num_nodes = params.num_nodes;
stopval = params.stopval;
subSample = 0;
clFold = 'Classifier';

for Kndx=1:size(indexs,2)
    for i = 1:length(Labels)
        [foo labelSet] = fileparts(HOMELABELSETS{i});
        index = indexs{i,Kndx};
        if(~isempty(index) && params.subSample > 0)
            saveFile = fullfile(HOMEDATA,clFold,'Partial',labelSet,sprintf('subSampleNdx%d-%d.mat',params.subSample,params.balancedsubSample));make_dir(saveFile);
            if(exist(saveFile,'file'))
                load(saveFile);
            else
                if(params.balancedsubSample) %balanced
                    [ls count] = UniqueAndCounts(index.label);
                    ptspl = params.subSample/length(ls);
                    ndxs = [];
                    for l = ls(:)'
                        ndx = find(index.label==l);
                        rndx = randperm(length(ndx));
                        ndxs = [ndxs ndx(rndx(1:min(end,ptspl)))];
                    end
                else
                    ndxs = randperm(length(index.image));
                    ndxs = ndxs(1:params.subSample);
                end
                ndxs = sort(ndxs);
                save(saveFile,'ndxs');
            end
            index.image = index.image(ndxs);
            index.sp = index.sp(ndxs);
            index.label = index.label(ndxs);
            index.spSize = index.spSize(ndxs);
        end
        indexs{i,Kndx} = index;
    end
end

classifiers = cell(size(indexs,2),length(Labels));
trees = params.num_iterations;
numGood = 0;
for Kndx=1:size(indexs,2)
    for i = 1:length(Labels)
        [foo labelSet] = fileparts(HOMELABELSETS{i});
        index = indexs{i,Kndx};
        rmmask = zeros(size(index.label))==1;
        rmlind = find(labelMask{i}==0);
        labels = Labels{i}(labelMask{i});
        for k = 1:length(labels)
            labels{k} = FixLabelName(labels{k});
        end
        for k = rmlind(:)'
            rmmask = rmmask|index.label==k;
        end
        index.label(rmmask)=[];
        index.image(rmmask)=[];
        index.sp(rmmask)=[];
        usedL = unique(index.label);
        semTmp = MakeFileName(HOMEDATA,clFold,labelSet,params,Kndx);make_dir(semTmp);
        %{
        semTmp = fullfile(HOMEDATA,clFold,'Partial',labelSet,sprintf('ClassK%03d%s-N%03dS%.4f%s-%d',params.K(Kndx),params.segSuffix,num_nodes,stopval,params.init_weight,testsetnum));make_dir(semTmp);
        if(params.subSample>0)
            semTmp = [semTmp sprintf('-ss%d',params.subSample)];
            if(params.balancedsubSample)
                semTmp = [semTmp '-bal'];
            end
        end
        %}
        if(length(labels)==2)
            usedL = 1;
        end
        labelsFinished = 0;
        for j = usedL(:)'
            tempSave = sprintf('%s-%s.mat',semTmp,labels{j});
            if(exist(tempSave,'file'))
                load(tempSave);
                if(size(wcs,1)>=trees)
                    labelsFinished = labelsFinished +1;
                elseif (length(aveconf)>=t-1 && t>10 && (aveconf(t)-aveconf(t-10) < stopval))
                    labelsFinished = labelsFinished +1;
                else
                    fprintf('notdone\n');
                end
            end
        end
        if(labelsFinished>=length(usedL))
            labels2Train = 1:length(labels);
            if(length(labels)==2)
                labels2Train = 1;
            end
            classifiers{Kndx,i} = train_boosted_dt_mc([], [], labels(index.label)', trees, num_nodes, stopval, [], subSample, labels2Train, semTmp, labels);
            numGood = numGood + 1;
        end
    end
end
if(numGood == numel(classifiers))
    return;
end

for trees = min(100,params.num_iterations):100:params.num_iterations
    for Kndx=1:size(indexs,2)
        for i = 1:length(Labels)
            [foo labelSet] = fileparts(HOMELABELSETS{i});
            index = indexs{i,Kndx};
            rmmask = zeros(size(index.label))==1;
            rmlind = find(labelMask{i}==0);
            labels = Labels{i}(labelMask{i});
            for k = 1:length(labels)
                strind = strfind(labels{k},'/');
                labels{k}(strind) = '-';
            end
            for k = rmlind(:)'
                rmmask = rmmask|index.label==k;
            end
            index.label(rmmask)=[];
            index.image(rmmask)=[];
            index.sp(rmmask)=[];
            usedL = unique(index.label);
            labels2Train = 1:length(labels);
            labelsFinished = 0;
            semTmp = MakeFileName(HOMEDATA,clFold,labelSet,params,Kndx);make_dir(semTmp);
            %{
            semTmp = fullfile(HOMEDATA,clFold,'Partial',labelSet,sprintf('ClassK%03d%s-N%03dS%.4f%s-%d',params.K(Kndx),params.segSuffix,num_nodes,stopval,params.init_weight,testsetnum));make_dir(semTmp);
            if(params.subSample>0)
                semTmp = [semTmp sprintf('-ss%d',params.subSample)];
                if(params.balancedsubSample)
                    semTmp = [semTmp '-bal'];
                end
            end
            %}
            if(length(usedL)==2)
                usedL = 1;
            end
            if(length(labels)==2)
                labels2Train = 1;
            end
            for j = usedL(:)'
                tempSave = sprintf('%s-%s.mat',semTmp,labels{j});
                if(exist(tempSave,'file'))
                    load(tempSave);
                    if(t>=trees)
                        labelsFinished = labelsFinished +1;
                    end
                end
            end
            if(labelsFinished==length(usedL))
                classifiers{Kndx,i} = train_boosted_dt_mc([], [], labels(index.label)', trees, num_nodes, stopval, [], subSample, labels2Train, semTmp, labels);
            else
                inputLabels = labels(index.label)';
                labelList = labels;
                retSetSPDesc = LoadSegmentDesc(fileList,index,HOMEDATA,params.segmentDescriptors,params.K(Kndx),params.segSuffix);
                features = GetFeaturesForClassifier(retSetSPDesc);
                clear retSetSPDesc;
                cat_features = [];
                init_weights = [];
                if(strcmp(params.init_weight,'cFreq'))
                    [a b] = UniqueAndCounts(index.label);
                    b = 1./b;
                    init_weights = b(index.label);
                elseif(strcmp(params.init_weight,'cFreq2'))
                    [a b] = UniqueAndCounts(index.label);
                    b = (1./b).^2;
                    init_weights = b(index.label);
                    init_weights = init_weights.*length(labels)./sum(init_weights);
                elseif(strcmp(params.init_weight,'DupData'))
                    startInLabels = index.label;
                    [a b] = UniqueAndCounts(startInLabels);
                    tnp = max(b);
                    newf = cell(size(a));
                    newl = cell(size(a));
                    for l = a(:)'
                        numaddpts = tnp - b(l);
                        lndx = find(startInLabels==l);
                        toadd = randperm(length(lndx));
                        toadd = lndx(toadd);
                        toadd = repmat(toadd,[1 ceil(numaddpts/length(toadd))]);
                        toadd = toadd(1:numaddpts);
                        newf{l} = features(toadd,:);
                        newl{l} = ones(size(toadd))*l;
                    end
                    inputLabels = labels([startInLabels cell2mat(newl')])';
                    features = [features; cell2mat(newf)];
                    clear newf;
                end
                labels2Train = 1:length(labels);
                if(length(labels)==2)
                    labels2Train = 1;
                end
                classifiers{Kndx,i} = train_boosted_dt_mc(features, cat_features, inputLabels, trees, num_nodes, stopval, init_weights, subSample, labels2Train, semTmp, labelList); %#ok<CCAT>
            end
        end
    end
end

function semTmp = MakeFileName(HOMEDATA,clFold,labelSet,params,Kndx)
    %semTmp = fullfile(HOMEDATA,clFold,'Partial',labelSet,sprintf('ClassK%03d%s-N%03dS%.4f%s-%d',params.K(Kndx),params.segSuffix,params.num_nodes,params.stopval,params.init_weight,params.testsetnum));
    %semTmp = fullfile(HOMEDATA,clFold,'Partial',labelSet,sprintf('ClassK%03d%s-N%03dS%.4f%s-%d',params.K(Kndx),params.segSuffix,params.num_nodes,0              ,params.init_weight,params.testsetnum));
    semTmp = fullfile(HOMEDATA,clFold,'Partial',labelSet,sprintf('ClassK%03d%s-N%03d%s-%d',params.K(Kndx),params.segSuffix,params.num_nodes,params.init_weight,params.testsetnum));
    if(params.subSample>0)
        semTmp = [semTmp sprintf('-ss%d',params.subSample)];
        if(params.balancedsubSample)
            semTmp = [semTmp '-bal'];
        end
    end

