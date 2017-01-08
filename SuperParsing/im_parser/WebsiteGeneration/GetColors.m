function [ labelColors ] = GetColors(HOME, HOMECODE, WEBLABELSETS, Labels)
%GETCOLORS Summary of this function goes here
%   Detailed explanation goes here
    labelColors = cell(size(WEBLABELSETS));
    for k = 1:length(WEBLABELSETS)
        [foo setname] = fileparts(WEBLABELSETS{k});
        saveFile = fullfile(HOME,[setname 'colors.mat']);
        if(~exist(saveFile,'file'))
            numL = length(Labels{k});
            h = (1/numL):(1/numL):1;
            s = repmat([1 .5],[1 ceil(numL/2)]);s = s(1:numL);
            v = repmat([1 1],[1 ceil(numL/2)]);v = v(1:numL);
            rndx = numL:-1:1;%randperm(numL);
            labelColor = [hsv2rgb([h(rndx)' s(rndx)' v(rndx)']); [0 0 0]];
        else
            load(saveFile);
        end
        labelColors{k} = labelColor;
    end
    %}

    for k = 1:length(WEBLABELSETS)
        [foo setname] = fileparts(WEBLABELSETS{k});
        saveFile = fullfile(HOMECODE,'Colors.txt');
        if(exist(saveFile,'file'))
            data = importdata(saveFile);
            for i = 1:length(data.textdata)
                labelname = lower(data.textdata{i});
                lndx = find(strcmp(lower(Labels{k}),labelname));
                if(~isempty(lndx))
                    labelColors{k}(lndx,:) = data.data(i,:)./255;
                    fprintf('%s Got: %s\n',setname,labelname);
                else
                    %fprintf('%s Missed: %s\n',setname,labelname);
                end
            end
            fprintf('\n');
        end
    end

    for k = 1:length(WEBLABELSETS)
        [foo setname] = fileparts(WEBLABELSETS{k});
        saveFile = fullfile(HOME,[setname 'Colors.mat']);
        labelColor = labelColors{k};
        if(~exist(saveFile,'file'))
            save(saveFile,'labelColor');
        end
    end

end

