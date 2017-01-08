function usedRetInds = FindAllRetInds(HOMEDATA,testFileList,trainGlobalDesc,testGlobalDesc,testParams)

DataDir = fullfile(HOMEDATA,testParams.TestString);
range = 1:length(testFileList);
usedRetInds = [];
for i = range
    [folder file] = fileparts(testFileList{i});
    baseFName = fullfile(folder,file);
    %Get Retrieval Set
    retIndsAll = FindRetrievalSet(trainGlobalDesc,SelectDesc(testGlobalDesc,i,1),DataDir,baseFName,testParams);
    usedRetInds = [usedRetInds retIndsAll(1:min(end,max(testParams.retSetSize)))];
end

usedRetInds = unique(usedRetInds);