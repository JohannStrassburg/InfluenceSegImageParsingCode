function LMphotoalbum(varargin);
%
% LMphotoalbum(D, pagename);
%
% or
%
% for i = 1:length(D);
%    folderlist{i} = D(i).annotation.folder;
%    filelist{i} = D(i).annotation.filename;
% end
% LMphotoalbum(folderlist, filelist, pagename);
%
% This function allows building an interface that communicates with the
% LabelMe web annotation tool. 
%
% It can be used to label specific images.
% 
% You can use this to label images that have some characteristic that you
% want. You can use this function in combination with the LMquery function.
%
% For instance, if you want to create a web page with images only of
% kitchens so that the thumbnails are connected to the LabelMe web
% annotation tool online, you can do the next thing:
%
% [D,j] = LMquery(D, 'folder', 'kitchen');
% LMphotoalbum(D, 'myphotoalbum.html');
%
% This will create a web page with thumbnails of the selected images. But
% more importantly, the images will be link with the LabelMe online tool.
% So, whenever you will click on one image it will call the annotation tool
% and will open that specific image showing the annotations available
% online (not the local ones that you have). Now you can label more objects
% in that image and download later the annotations.


if nargin == 2
    D = varargin{1}
    pagename   = varargin{2};
    for i = 1:length(D)
        folderlist{i} = D(i).annotation.folder;
        filelist{i} = D(i).annotation.filename;
    end
    ADDSUMMARY = 1;
end
if nargin > 2
    folderlist = varargin{1};
    filelist   = varargin{2};
    pagename   = varargin{3};
    ADDSUMMARY = 0;
end


Nimages = length(folderlist);
webpage = 'http://labelme.csail.mit.edu/tool.html?collection=LabelMe&mode=i'

% Hearder
page = {};
page = addline(page, '<html><head><title>LabelMe photoalbum</title><body>');
page = addline(page, '<img src="http://labelme.csail.mit.edu/Icons/LabelMeNEWtight198x55.gif" height=26 alt="LabelMe" /><br>');

if ADDSUMMARY
    [names, counts]  = LMobjectnames(D);
    [foo, ndxn] = sort(-counts);
    names = names(ndxn);
    counts = counts(ndxn);

    page = addline(page, '<b>Database summary:</b><br>');
    page = addline(page, sprintf('There are %d images<br>', Nimages));
    page = addline(page, sprintf('There are %d polygons<br>', sum(counts)));
    page = addline(page, sprintf('There are %d descriptions<br>', length(names)));
    page = addline(page, sprintf('Last update: %s<br>', date));
    page = addline(page, ' List of objects:<br>');
    page = addline(page, '<div style="width:800px;height:100px;overflow-y:scroll;overflow-x:none;border:thin solid;">');
    for no = 1:length(names)
        page = addline(page, sprintf('<li> %s (%d instances)<br>', names{no}, counts(no)));
    end

    page = addline(page, '</div>');
    page = addline(page, '<br><hr align="Left" size="1">');

else
    page = addline(page, '<b><font size=5> Photoalbum</font></b><br>');
    page = addline(page, '<hr align="Left" size="1"><br>');
    page = addline(page, sprintf('<b><font size=4>There are %d images.</font></b><br>', Nimages));
    page = addline(page, '<b><font size=4>Click on an image to visualize it with the online annotation tool</font></b><br>');
end

% Create links for each image
for i = 1:Nimages
    imageline = sprintf('<IMG src="http://people.csail.mit.edu/brussell/research/LabelMe/Thumbnails/%s/%s" height=96 border=2>', ...
        folderlist{i}, filelist{i});
    page = addline(page, ...
        [sprintf('%d) <a href="%s&folder=%s&image=%s" target="_blank">%s</a>', ...
        i, webpage, folderlist{i}, filelist{i}, imageline)]);
end
page = addline(page, '<hr align="Left" size="1"><br>');

% Close web page
page = addline(page, '</body></html>');

% write web page
fid = fopen(pagename,'w');
for i = 1:length(page); 
    fprintf(fid, [page{i} '\n']); 
end
fclose(fid);


function page = addline(page, line)

page = [page {line}];
