function [ fg ] = GetDetectorStream(HOMEIMAGES,HOMELABELSET, trainFileList, stream_params)

fg = cell(0,1);
for i = 1:length(trainFileList)
    [fold, base, ext] = fileparts(trainFileList{i});
    gtbb = GetGroundTruthBB(HOMEIMAGES,HOMELABELSET,trainFileList{i});
    names =  {gtbb.obj(:).cls};
    lind = find(strcmp(names,stream_params.cls));
    for j = lind(:)'
        res = [];
        res.I = gtbb.I;
        res.curid = gtbb.curid;
        res.bbox = gtbb.obj(j).bbox;
        res.cls = gtbb.obj(j).cls;
        res.objectid = gtbb.obj(j).objectid;
    	res.filer = sprintf('%s.%d.%s.mat', res.curid, res.objectid, res.cls);
        fg{end+1} = res;
        if length(fg) == stream_params.stream_max_ex
            return;
        end
    end
end
end

