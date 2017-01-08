function ExtractServerSegments(HOME,level)
    HOMEVIDEO = fullfile(HOME, 'Videos');
    HOMEDESC = fullfile(HOME,'Data','Descriptors',['SP_Desc_k' num2str(level) '_vidSeg']);
    files = dir_recurse(fullfile(HOMEVIDEO,'*.pb'),0,0);
    pfig = ProgressBar('Extracting Segments');
    oldfold = pwd;
    for i = 1:length(files)
        [fold base] = fileparts(files{i});
        pbfile = fullfile(HOMEVIDEO,files{i});
        make_dir(fullfile(HOMEDESC,'super_pixel_ims',base,'t.t'));
        cd(fullfile(HOMEDESC,'super_pixel_ims',base));
        cmd = ['d:\segment_converter.exe ' pbfile ' --bitmap_ids=' num2str(level)];
        system(cmd);
        imFiles = dir_recurse(fullfile(HOMEDESC,'super_pixel_ims',base,'*.png'),1,0);
        for j = 1:length(imFiles)
            [imfold imbase] = fileparts(imFiles{j});
            segIm = uint32(imread(imFiles{j}));
            superPixels = 1+bitshift(segIm(:,:,1),16)+bitshift(segIm(:,:,2),8)+segIm(:,:,3);
            saveFile = fullfile(HOMEDESC,'super_pixels',base,sprintf('frame%05d.mat',j-1));make_dir(saveFile);
            save(saveFile,'superPixels');
        end
        ProgressBar(pfig,i,length(files));
    end
    close(pfig);
    cd(oldfold);
end