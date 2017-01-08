
HOME = 'D:\im_parcer\Core\Data\BSEdata\';
HOMEWEB = fullfile(HOME,'Web');
HOMEIMAGES = fullfile(HOME,'Images');
fileList1 = dir_recurse(fullfile(HOMEIMAGES,'*.jpg'),0);
fileList2 = dir_recurse(fullfile(HOMEIMAGES,'*.jpeg'),0);

fileList = {fileList2{:} fileList1{:}};

make_dir(fullfile(HOMEWEB,'index.html'));
fid = fopen(fullfile(HOMEWEB,'index.html'),'w');
fprintf(fid,'<table>\n');
for i = 1:length(fileList)
    [folder base ext] = fileparts(fileList{i});
    %con = read_array(fullfile(HOMEIMAGES,[fileList{i} '.pb']));
    segCon = read_array(fullfile(HOMEIMAGES,[fileList{i} '.pbs']));
    seg2 = read_array_int(fullfile(HOMEIMAGES,[fileList{i} '.2.seg']));
    seg15 = read_array_int(fullfile(HOMEIMAGES,[fileList{i} '.15.seg']));
    
    
    fprintf(fid,'<tr><td>\n');
    
    make_dir(fullfile(HOMEWEB,'Images',folder,[base '.jpg']));
    copyfile(fullfile(HOMEIMAGES,fileList{i}),fullfile(HOMEWEB,'Images',folder,[base '.jpg']));
    fprintf(fid,'<img src="%s/%s/%s">\n','Images',folder,[base '.jpg']);
    
    outfile = fullfile(HOMEWEB,'Contour1',folder,[base '.png']);make_dir(outfile);
    if(~exist(outfile,'file'))
        show(segCon{1},1);
        set(gcf,'PaperPositionMode','auto');print(outfile,'-dpng','-r96');
    end
    fprintf(fid,'<img src="%s/%s/%s">\n','Contour1',folder,[base '.png']);
    
    outfile = fullfile(HOMEWEB,'Contour2',folder,[base '.png']);make_dir(outfile);
    if(~exist(outfile,'file'))
        show(segCon{2},1);
        set(gcf,'PaperPositionMode','auto');print(outfile,'-dpng','-r96');
    end
    fprintf(fid,'<img src="%s/%s/%s">\n','Contour2',folder,[base '.png']);
    
    %{
    outfile = fullfile(HOMEWEB,'Seg15',folder,[base '.png']);make_dir(outfile);
    if(~exist(outfile,'file'))
        show(seg15,1);
        set(gcf,'PaperPositionMode','auto');print(outfile,'-dpng','-r96');
    end
    fprintf(fid,'<img src="%s/%s/%s">','Seg15',folder,[base '.png']);
    
    outfile = fullfile(HOMEWEB,'Seg2',folder,[base '.png']);make_dir(outfile);
    if(~exist(outfile,'file'))
        show(seg2,1);
        set(gcf,'PaperPositionMode','auto');print(outfile,'-dpng','-r96');
    end
    fprintf(fid,'<img src="%s/%s/%s"><br><br>','Seg2',folder,[base '.png']);
    %}
    
    fprintf(fid,'</td></tr>\n');
end
fprintf(fid,'</table>\n');
fclose(fid);