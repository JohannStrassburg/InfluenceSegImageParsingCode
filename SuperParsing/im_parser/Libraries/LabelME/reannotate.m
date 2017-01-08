
HOME = '/home/jstrassb/im_parser/Barcelona';
HOMEANNOTATIONS = '/home/jstrassb/im_parser/Barcelona/Annotations';
HOMEIMAGES = fullfile(HOME,'Images');
NEWFOLDER = fullfile(HOME,'Annotations_sorted');
if (~exist(NEWFOLDER))
        mkdir(NEWFOLDER);
    end

d = dir(HOMEANNOTATIONS);
isub = [d(:).isdir]; %# returns logical vector
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];

for i = 1:length(nameFolds)
	dd = dir(fullfile(HOMEANNOTATIONS,cell2mat(nameFolds(i))));
	dirIndex = [dd.isdir];  %# Find the index for directories
  	fileList = {dd(~dirIndex).name}';  %'# Get a list of the files
    folder = fullfile(NEWFOLDER,cell2mat(nameFolds(i)));
    if (~exist(folder))
        mkdir(folder);
    end
	
	for j = 1:length(fileList)
		filename = fullfile(HOMEANNOTATIONS,cell2mat(nameFolds(i)),cell2mat(fileList(j)));
		[a, xml] = loadXML(filename);
		s = fullfile(HOMEIMAGES, a.annotation.folder, a.annotation.filename);
		s = imread(s);
		[b, k, l] = LMsortlayers(a.annotation,s);
		c=struct;
		c.annotation = b;
        [q,name,ext] = fileparts(filename);
		filename_out = fullfile(NEWFOLDER, a.annotation.folder,strcat(name,ext));
		writeXML(filename_out,c);
    end
    display(i);
end

%fileList = dir_recurse(fullfile(HOMEIMAGES,'*.*'),0);

%filename =



