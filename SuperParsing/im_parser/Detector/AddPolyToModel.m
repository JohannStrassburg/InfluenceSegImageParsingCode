function [ models ] = AddPolyToModel( HOMEANN, models, modelFile )
%ADDPOLYTOMODEL Summary of this function goes here
%   Detailed explanation goes here

lastXML = '';
modified = false;
for i = 1:length(models)
    if(isfield(models{i},'polygon') && isfield(models{i},'clsNum'))
        continue;
    end
    [fold base ext] = fileparts(models{i}.curid);
    base = [base ext];
    xmlFile = fullfile(HOMEANN,fold,[base '.xml']);
    if(~strcmp(xmlFile,lastXML))
        [ann] = LMread(xmlFile);
        objIds = str2double({ann.object.id});
        id2Obj = zeros(max(objIds),1);
        id2Obj(objIds) = 1:length(objIds);
    end
    models{i}.polygon = ann.object(id2Obj(str2double(models{i}.objectid))).polygon;
    models{i}.clsNum = str2num(ann.object(id2Obj(str2double(models{i}.objectid))).namendx);
    lastXML = xmlFile;
    modified = true;
end

if(exist('modelFile','var') && modified)
    save(modelFile,'models');
end

end

