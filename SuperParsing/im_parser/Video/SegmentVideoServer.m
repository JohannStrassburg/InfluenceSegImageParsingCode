function SegmentVideoServer( HOMEVIDEO )
    files = dir_recurse(fullfile(HOMEVIDEO,'*.mp4'),1,0);
    pfig = ProgressBar('Segmenting Videos');
    for i = 1:length(files)
        [fold base] = fileparts(files{i});
        pbfile = fullfile(fold,[base '.pb']);
        if(~exist(pbfile,'file'))
            cmd = ['curl -F "file_name=@' files{i} '" http://neumann.cc.gt.atl.ga.us/videosegmentation/segment_video?source=UPLOAD'];
            [~, result] = system(cmd);
            pos = strfind(result,'[SUCCESS]: ');
            key = result(pos+11:end);
            [~, result] = system(['curl http://neumann.cc.gt.atl.ga.us/videosegmentation/query_job?id=' key]);
            while(isempty(strfind(result,'[DONE]')))
                pause(10);
                [~, result] = system(['curl http://neumann.cc.gt.atl.ga.us/videosegmentation/query_job?id=' key]);
                pos = strfind(result,'[');
                fprintf('%s\n',result(pos:end-2));
            end
            urlwrite(['http://neumann.cc.gt.atl.ga.us/videosegmentation/stream_result?id=' key],pbfile);
        end
        ProgressBar(pfig,i,length(files));
    end
    close(pfig);
end

