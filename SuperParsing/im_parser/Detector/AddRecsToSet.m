function imSet = AddRecsToSet(HOMEANNOTATIONS,imSet,imSetBase,cls)


for i = 1:length(imSet)
    [fold base] = fileparts(imSetBase{i});
    xmlFile = fullfile(HOMEANNOTATIONS,fold,[base '.xml']);
    [ann] = LMread(xmlFile);
    [bbs jc] = LMobjectboundingbox(ann,cls);
    %clsNums = cellfun(@(x)str2num(x),{ann.object(j).namendx});
    recs.folder = fold;
    recs.filename = base;
    recs.source = '';
    recs.size.width = ann.imagesize.ncols;
    recs.size.height = ann.imagesize.nrows;
    recs.size.depth = 3;
    recs.segmented = 0;
    recs.imgname = sprintf('%08d',i);
    recs.imgsize = [ann.imagesize.ncols ann.imagesize.nrows 3];
    recs.database = '';
    clear object;
    for j = 1:length(jc)
        object(j).class = cls;
        object(j).view = '';
        object(j).truncated = 0;
        object(j).occluded = 0;
        object(j).difficult = 0;
        object(j).label = cls;
        object(j).bbox = bbs(j,:);%[sub2 sub1 sub2+size(A,2) sub1+size(A,1) ];
        object(j).bndbox.xmin =object(j).bbox(1);
        object(j).bndbox.ymin =object(j).bbox(2);
        object(j).bndbox.xmax =object(j).bbox(3);
        object(j).bndbox.ymax =object(j).bbox(4);
        object(j).polygon = [];
    end
    recs.objects = object;
    I = imSet{i};
    imSet{i} = [];
    imSet{i}.I = I;
    imSet{i}.recs = recs;
end

end