function [classifiers] = TrainRandForest(HOMEDATA, HOMELABELSETS, fileList, indexs, Labels, labelMask, params)
num_nodes = params.num_nodes;
stopval = params.stopval;
subSample = 0;
clFold = 'Classifier';

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
        if(length(usedL)==2)
            usedL = 1;
        end
        if(length(labels)==2)
            labels2Train = 1;
        end
        
        inputLabels = index.label';
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
        classifiers{Kndx,i} = classRF_train(features,inputLabels);
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

