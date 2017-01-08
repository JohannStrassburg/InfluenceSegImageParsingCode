function EvaluateRetrievalSet(HOMEDATA,trainFileList,testParams)


retFiles = dir_recurse(fullfile(HOMEDATA,testParams.TestString,'RetrievalSet','*.mat'),0);
folders = cell(size(retFiles));
for i = 1:length(retFiles);folders{i} = fileparts(retFiles{i});end
imageClasses = unique(folders);
classRates = zeros(length(imageClasses),length(testParams.retSetSize));
for j = 1:length(testParams.retSetSize)
    classStats = zeros(length(imageClasses),2);
    for i = 1:length(retFiles)
        [imageClass name] = fileparts(retFiles{i});
        retInds = FindRetrievalSet([],[],fullfile(HOMEDATA,testParams.TestString),fullfile(imageClass,name),testParams);

        inds = strmatch([imageClass filesep],trainFileList(retInds(1:testParams.retSetSize(j))));
        classind = strmatch(imageClass,imageClasses,'exact');
        classStats(classind,1) = classStats(classind,1) + length(inds);
        classStats(classind,2) = classStats(classind,2) + testParams.retSetSize(j);
    end
    classRates(:,j) = classStats(:,1)./classStats(:,2);
end

fid = fopen(fullfile(HOMEDATA,testParams.TestString,'RetrievalSet.txt'),'w');

fprintf(fid,'%s\t','Class Name');
for j = 1:length(testParams.retSetSize)
    fprintf(fid,'%d\t',testParams.retSetSize(j));
end
fprintf(fid,'\n');
for i = 1:length(imageClasses)
    fprintf(fid,'%s\t',imageClasses{i});
    for j = 1:length(testParams.retSetSize)
        fprintf(fid,'%.2f%%\t',100*classRates(i,j));
    end
    fprintf(fid,'\n');
end

fclose(fid);