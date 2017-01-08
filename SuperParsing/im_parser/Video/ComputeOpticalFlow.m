function ComputeOpticalFlow (HOMEIMAGESALL,HOMEDATA,canSkip)

if(~exist('canSkip','var'))
    canSkip = 1;
end

videoDirs = dir_recurse(fullfile(HOMEIMAGESALL,'*'),0,0);

alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;
para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];
if(matlabpool('size')==0)
    matlabpool;
end
numThreads = matlabpool('size');
chunkSize = numThreads * floor(100/numThreads);
pfig = sp_progress_bar('Computing Optical Flow');
for i = 1:length(videoDirs)
    files = dir_recurse(fullfile(HOMEIMAGESALL,videoDirs{i},'*.png'),0);
    tic
    numFiles = length(files);
    videoDir = videoDirs{i};
    files2 = files([1 1:end]);
    filerange = 2:numFiles; 
    ranges = accumarray(floor((0:numel(filerange)-1)/chunkSize)'+1,filerange,[],@(x) {x});
    %make list of outputfiles
    outfile = cell(numFiles,1);
    for j = 1:numFiles ; [fold base ext] = fileparts(files2{j}); outfile{j} = fullfile(HOMEDATA,'OpticalFlow',videoDir,fold,[base '.mat']); end
    %split up into ranges and run
    for rndx = 1:length(ranges)
        filerange = ranges{rndx};
        vxc = cell(size(filerange));
        vyc = cell(size(filerange));
        for j = 1:length(filerange)
            if(~exist(outfile{filerange(j)},'file')&&canSkip)
                im1 = imread(fullfile(HOMEIMAGESALL,videoDir,files2{filerange(j)}));
                im2 = imread(fullfile(HOMEIMAGESALL,videoDir,files{filerange(j)}));
                %if(0); show(im1,1); show(im2,2); figure(3);clf;subplot(1,2,1);imagesc(vxc{j}),colorbar;subplot(1,2,2);imagesc(vyc{j}),colorbar; end
                fprintf('%04d of %04d\n',filerange(j)-1,numFiles-1);
                [vx,vy] = Coarse2FineTwoFrames(im1,im2,para);
                vxc{j}=single(vx);vyc{j}=single(vy);
            end
        end
        for j = 1:length(filerange)
            if(~isempty(vxc{j}))
                vx=vxc{j}; vy=vyc{j};
                make_dir(outfile{filerange(j)});save(outfile{filerange(j)},'vx','vy');
            end
        end
        sp_progress_bar(pfig,i,length(videoDirs),max(filerange),numFiles-1,['Video: ' videoDirs{i} ' (Computing Flow)']);
    end
end

end