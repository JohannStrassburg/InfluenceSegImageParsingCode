function classifier = train_boosted_dt_mc(features, cat_features, labels, ...
    num_iterations, num_nodes, stopval, init_weights, subSample, labels2train, tempFile, varargin)
% Train a classifier based on boosted decision trees.  Boosting done by the
% logistic regression version of Adaboost (Adaboost.L - Collins, Schapire,
% Singer 2002).  At each
% iteration, a set of decision trees is created for each class, with
% confidences equal to 1/2*ln(P+/P-) for that class, according to the
% weighted distribution.  Final classification is based on the largest
% confidence label (possibly incorporating a prior as h0(c) =
% 1/2*ln(Pc/(1-Pc)).  Weights are assigned as
% w(i,j) = 1 / (1+exp(sum{t in iterations}[yij*ht(xi, j)])).  

if length(varargin) == 1  % class names supplied
    gn = varargin{1};
    gid = zeros(size(labels));
    for c = 1:length(gn)
        ind = find(strcmp(labels, gn{c}));
        gid(ind) = c;
        if (all(~isnan(init_weights)) && ~isempty(init_weights))
            disp([gn{c} ': ' num2str(sum(init_weights(ind)))]);
        else
            disp([gn{c} ': ' num2str(length(ind))]);
        end
    end
    ind = find(gid==0);
    if(~isempty(ind))
        gid(ind) = [];
        labels(ind) = [];
        features(ind, :) = [];
    end
else    
    [gid, gn] = grp2idx(labels);    
    gn
end
clear labels

if ~exist('stopval', 'var') || isempty(stopval)
    stopval = 0;
end
if ~exist('init_weights', 'var') 
    init_weights = [];
end

classifier.names = gn;

num_classes = length(gn);
num_data = length(gid);

if isempty(init_weights)
    init_weights = ones(num_data, 1)/num_data;
else
    init_weights = init_weights / sum(init_weights);
end

% if no examples from a class are present, create one dummy example for
% that class with very small weight
for c = 1:numel(gn)
    if (~any(gid==c) && ~isempty(features))
        disp(['warning: no examples from class ' gn(c)])
        gid(end+1) = c;
        features(end+1, :) = zeros(size(features(end, 1)));
        num_data = num_data + 1;
        init_weights(end+1) = min(init_weights)/2;        
    end
end

classCounts = zeros(num_classes,1);
for i = 1:num_classes
    classCounts(i) = sum(gid==i);
end


all_conf = zeros(num_data, num_classes);

if (isempty(labels2train))
    labels2train = 1:num_classes;
end

for c = labels2train(:)'

    disp(['class: ' num2str(gn{c})]);    
    y = (gid == c)*2-1;
    cl = [-1 1];
    nc = 2;
    w = zeros(num_data, 1);
    cw = zeros(num_classes, 1);  
    posind = find(y==1);
    negind = find(y==-1);
    for i = 1:2
        indices = find(y==cl(i));
        %count = sum(init_weights(indices));
        %w(indices) = init_weights(indices) / count / 2;
        w(indices) = init_weights(indices);
        %w(indices) = init_weights(indices)./(2*sum(init_weights(indices)));
        
        if cl(i)==1
            %classifier.h0(c) = log(count / (1-count));
            classifier.h0(c) = 0;
        end
        
    end
        
    data_confidences = zeros(num_data, 1);
    aveconf = [];
    
    maxPos = 10000;
    if(subSample&&length(posind)>maxPos)
        subPosInd = posind(randperm(length(posind)));
        posind = subPosInd(1:min(maxPos,length(subPosInd)));
    end
    if(subSample&&length(negind)>2*length(posind))
        subNegInd = negind(randperm(length(negind)));
        subNegInd = subNegInd(1:min(2*length(posind),length(subNegInd)));
        inds = sort([subNegInd(:)' posind(:)']);
    end
    
    tempSave = sprintf('%s-%s.mat',tempFile,gn{c});
    startT = 1;
    if(exist(tempSave,'file'))
        load(tempSave);
        copySize = min(size(wcs,1),num_iterations);
        if(copySize>10)
            newSize = find(aveconf(11:end)-aveconf(1:end-10)<stopval,1);
            copySize = min([copySize newSize+10]);
        end
        classifier.wcs(1:copySize, c) = wcs(1:copySize,:);
        startT = t+1;
    end
    skip = 0;
    if(startT>num_iterations)
        skip = 1;
    end
    if(startT<num_iterations && isempty(features))
        skip = 1;
    end
    if ~skip && length(aveconf)>=startT-1 && startT>11 && (aveconf(startT-1)-aveconf(startT-11) < stopval)
        disp(num2str(aveconf))
        disp(['Stopping after ' num2str(startT-1) ' trees'])            
        skip = 1;
    end
    busyFile = sprintf('%s-%s.busy.mat',tempFile,gn{c});a = 1;
    if(exist(busyFile,'file'))
        fprintf('Skipping %s Busy\n',gn{c});
    elseif(skip)
    else
        try
            save(busyFile,'a');
            tic
            for t = startT:num_iterations
                % learn decision tree based on weighted distribution
                if(subSample&&length(negind)>2*min(maxPos,length(posind)))
                    ftemp = features(inds,:);
                    ytemp = y(inds);
                    wtemp = w(inds)./sum(w(inds));
                    dt = treefitw(ftemp, ytemp, wtemp, 1/length(inds)/2, 'catidx', cat_features, 'method', 'classification', 'maxnodes', num_nodes*4);
                else
                    dt = treefitw(features, y, w, 1/num_data/2, 'catidx', cat_features, 'method', 'classification', 'maxnodes', num_nodes*4);
                end

                [tmp, level] = min(abs(dt.ntermnodes-num_nodes));
                dt = treeprune(dt, 'level', level-1);

                % assign partition confidences
                classNames = dt.classname;
                pi = (strcmp(classNames{1},'1')) + (2*strcmp(classNames{2},'1'));
                ni = (strcmp(classNames{1},'-1')) + (2*strcmp(classNames{2},'-1'));
                cp = dt.classprob;
                confidences = 1/2*(log(cp(:, pi)) - log(cp(:, ni)));

                % assign weights
                [class_indices, nodes, classes] = treeval(dt, features);        
                data_confidences = data_confidences + confidences(nodes);

                w = 1 ./ (1+exp(y.*data_confidences));        
                w = w / sum(w);
                %w(y==1) = w(y==1)./(sum(w(y==1))*2);
                %w(y==-1) = w(y==-1)./(sum(w(y==-1))*2);

                disp(['c: ' num2str(mean(1 ./ (1+exp(-y.*data_confidences)))) '  e: ' num2str(mean(y.*data_confidences < 0)) '   w: ' num2str(max(w))]);  

                classifier.wcs(t, c).dt = dt;
                classifier.wcs(t, c).confidences = confidences;       


                aveconf(t) = mean(1 ./ (1+exp(-y.*data_confidences)));
                wcs = classifier.wcs(1:t, c);
                make_dir(tempSave);save(tempSave,'wcs','w','data_confidences','class_indices','nodes','confidences','cp','ni','pi','dt','level','t','aveconf');
                if t>10 && (aveconf(t)-aveconf(t-10) < stopval)
                    disp(num2str(aveconf))
                    disp(['Stopping after ' num2str(t) ' trees'])            
                    break;
                end
                fprintf('it: %d error: %.4f mins: %.2f\n',t,aveconf(t),toc/60);
            end
        catch ERR
            if(exist(busyFile,'file'))
                delete(busyFile);
            end
            throw(ERR)
        end
        if(exist(busyFile,'file'))
            delete(busyFile);
        end
    end
    if(length(y)< length(data_confidences))
        y(end+1:length(data_confidences)) = -1;
    end
    if(length(y) ~= length(data_confidences))
        y = -1*ones(size(data_confidences));y(1) = 1;
        all_conf = zeros(length(data_confidences), num_classes);
    end
    finalconf = 1 ./ (1+exp(-y.*data_confidences));
    finalerr = (y.*data_confidences < 0);
    disp(['confidence:: mean: ' num2str(mean(finalconf)) ...
        '  pos: ' num2str(mean(finalconf(y==1))) ...
        '  neg: ' num2str(mean(finalconf(y~=1)))]);
    disp(['training error:: mean: ' num2str(mean(finalerr)) ...
        '  pos: ' num2str(mean(finalerr(y==1))) ...
        '  neg: ' num2str(mean(finalerr(y~=1)))]);    
    if(size(all_conf,1)<length(data_confidences))
        all_conf = [all_conf; zeros(length(data_confidences)-size(all_conf,1),size(all_conf,2))];
    end
    all_conf(:, c) = data_confidences+classifier.h0(c);
  
end

% compute and display training error
[tmp, assigned_label] = max(all_conf, [], 2);
conf_matrix = zeros(num_classes, num_classes);
if(length(gid)>size(all_conf,1))
    gid = gid(1:size(all_conf,1));
end
for c = 1:num_classes    
    indices = find(gid==c);
    for c2 = 1:num_classes
        conf_matrix(c, c2) = mean(assigned_label(indices)==c2);
    end
    disp([gn{c} ' error: ' num2str(mean(assigned_label(indices)~=c))]);
end
disp('Confusion Matrix: ');
disp(num2str(conf_matrix));
if(length(gid)<length(assigned_label))
    assigned_label(length(gid)+1:end) = [];
end
disp(['total error: ' num2str(mean(assigned_label~=gid))]);


        