function fullTestFileList = GetFullVideoFiles(HOMEIMAGESALL,fileList,imageFileType)
dirList = (FileList2DirList(fileList));
dirs = unique(dirList);
fullTestFileList = cell(0);
if(~exist('imageFileType','var')); imageFileType = '*.*'; end
for d = 1:length(dirs)
    orgDir = pwd;
    cd(HOMEIMAGESALL);
    files = dir_recurse(fullfile(dirs{d},imageFileType));
    cd(orgDir);
    inFiles = sort(fileList(strcmp(dirs{d},dirList)));
    startNdx = find(strcmp(inFiles{1},files));
    endNdx = find(strcmp(inFiles{end},files));
    fullTestFileList = [fullTestFileList files(startNdx:endNdx)];
end