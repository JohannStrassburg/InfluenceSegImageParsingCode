function [ fg ] = GetGroundTruthBB( HOMEIMAGES,HOMELABELSET,fileName )
%GETGROUNDTRUTHBB Summary of this function goes here
%   Detailed explanation goes here
    [fold base] = fileparts(fileName);
    saveFile = fullfile(HOMELABELSET,fold,[base '.bb.mat']);
    if(exist(saveFile,'file'))
        load(saveFile);
    else
        [imSP, ll, names] = GenerateGroundTruthSegs(fullfile(HOMELABELSET,fold,[base '.mat']));
        count = 1;
        clear fg;
        fg.I = fullfile(HOMEIMAGES,fileName);
        fg.curid = fullfile(fold,base);
        for j = 1:length(ll)
            if(ll(j) == 0)
                continue;
            end
            [y x] = find(imSP==j);
            obj.bbox = [min(x) min(y) max(x) max(y)];
            obj.cls = names{ll(j)};
            obj.objectid = j;
            fg.obj(count) = obj;
            count = count +1;
        end
        save(saveFile,'fg');
    end
end

