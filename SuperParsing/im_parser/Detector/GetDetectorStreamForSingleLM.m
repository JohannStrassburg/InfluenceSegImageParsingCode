function [ fg ] = GetDetectorStreamForSingleLM(HOMEIMAGES,HOMEDATA, xmlFile, detectorParams)

fg = cell(0,1);

[ann] = LMread(xmlFile,HOMEIMAGES);
fileName = LMfilename(ann);
[fold, base] = fileparts(fileName);
if(detectorParams.preComputeHOG)
    imSet = LoadImAndPyr(HOMEIMAGES,HOMEDATA,{fileName},detectorParams);
    I = imSet{1};
    [ro co ch] = size(I.I);
else
    I = fullfile(HOMEIMAGES,fileName);
    im = convert_to_I(I);
    [ro co ch] = size(im);
end
[bbs jc] = LMobjectboundingbox(ann);
for j = 1:size(bbs,1);
    %LMplot(D, i, HOMEIMAGES);
    if(bbs(j,1)+2>co || bbs(j,2)+2>ro)
        continue;
    end
    res = [];
    res.I = I;
    res.curid = [fold '/' base];
    res.bbox = bbs(j,:);
    res.cls = ann.object(jc(j)).name;
    res.clsNum = str2num(ann.object(jc(j)).namendx);
    res.objectid = ann.object(jc(j)).id;
    res.filer = sprintf('%s.%s.%s.mat', res.curid, res.objectid, res.cls);
    res.polygon = ann.object(jc(j)).polygon;
    fg{end+1} = res;
end

end

