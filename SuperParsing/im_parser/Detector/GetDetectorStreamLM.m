function [ fg ] = GetDetectorStreamLM(HOMEIMAGES,D, stream_params)

fg = cell(0,1);
for i = 1:length(D)
    fileName = LMfilename(D(i).annotation);
    [fold, base] = fileparts(fileName);
    [bbs jc] = LMobjectboundingbox(D(i).annotation,stream_params.cls);
    for j = 1:size(bbs,1);
        %LMplot(D, i, HOMEIMAGES);
        I = LMimread(D,i,HOMEIMAGES);
        if(bbs(j,1)+2>size(I,2) || bbs(j,2)+2>size(I,1))
            continue;
        end
        res = [];
        res.I = I;
        res.curid = [fold '/' base];
        res.bbox = bbs(j,:);
        res.cls = stream_params.cls;
        res.objectid = jc(j);
    	res.filer = sprintf('%s.%d.%s.mat', res.curid, res.objectid, res.cls);
        res.polygon = D(i).annotation.object(jc(j)).polygon;
        fg{end+1} = res;
        if length(fg) == stream_params.stream_max_ex
            return;
        end
    end
end
end

