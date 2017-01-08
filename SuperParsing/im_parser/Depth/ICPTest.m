if(~exist('preload','var')||~preload)
    datasetDir = 'D:\im_parser\NYUDepth\clips';
    % The name of the scene to demo.
    sceneName = 'bathroom_0002';
    % The absolute directory of the 
    sceneDir = fullfile(datasetDir, sceneName);
    % Reads the list of frames.
    frameList = get_synched_frames(sceneDir);
end
preload = true;
iter=40;

dense = cell(length(frameList),1);
model = cell(length(frameList),1);
d = swapbytes(imread(fullfile(sceneDir, frameList(1).rawDepthFilename)));
model{1} = get_projected_3d(d,10,16)';
m = cell2mat(model);
T = cell(length(frameList),1);
T{1} = eye(4);
Tlast = eye(4);
figure(1),clf,plot3(m(1,:),m(3,:),m(2,:),'.b','MarkerSize',.5);
pfig = ProgressBar('ICP');
accData = zeros(4,length(frameList));
for i = 1:length(frameList)
    %accData(:,i) = get_accel_data(fullfile(sceneDir, frameList(i).accelFilename));
    %fprintf('%d %d %d %d %d \n',i, accData);
    show(imread(fullfile(sceneDir, frameList(i).rawRgbFilename)),2);
    %{
    d = swapbytes(imread(fullfile(sceneDir, frameList(i).rawDepthFilename)));
    data = get_projected_3d(d,10,16)';
    n_data = size(data,2);
    dataP = Tlast*[data; ones(1,n_data)];
    dataP = dataP(1:3,:)./repmat(dataP(4,:),[3 1]);
    
    %figure(1),plot3(model(1,:),model(2,:),model(3,:),'r.',dataP(1,:),dataP(2,:),dataP(3,:),'c.'),hold off;
    
    %matlab only version
    %[R T] = icp(model,data);
    weights=ones(1,n_data);
    rndvec=uint32(randperm(n_data)-1);
    sizernd=ceil(1.45*n_data);
    m = cell2mat(model(max(1,i-30):i-1)');
    [tmp, tmp, TreeRoot] = kdtree( m', []);
    % Run the ICP algorithm.
    [Ro,To]=icpCpp(m,dataP,weights,rndvec,sizernd,TreeRoot,iter);
    % Free allocated memory for the kd-tree.
    kdtree([],[],TreeRoot);
    Tnew = [Ro To;0 0 0 1];
    Tlast = Tnew*Tlast;
    data=Tlast*[data; ones(1,n_data)];
    data = data(1:3,:)./repmat(data(4,:),[3 1]);
    %figure(2),plot3(model(1,:),model(2,:),model(3,:),'r.',data(1,:),data(2,:),data(3,:),'b.'),hold off;
    
    
    %data = get_projected_3d(d)';
    %dense{i} = data;%Rlast*data+repmat(Tlast,1,size(data,2));
    model{i} = data;
    T{i} = Tlast;
    fprintf('%.3f\n',max(max(abs(Tlast-eye(4)))));
    %}
    ProgressBar(pfig,i,length(frameList));
    %m = cell2mat(model');
    %figure(1),clf,plot3(m(1,:),m(3,:),m(2,:),'.b','MarkerSize',.5);
end
close(pfig);

m = cell2mat(model(:,1:1000:end)');
figure(1),clf,plot3(m(1,:),m(3,:),m(2,:),'.b','MarkerSize',.5);
cc = zeros(3,length(T));
for i = 1:length(T);    cc(:,i) = T{i}(1:3,4);  end
figure(2),clf,plot3(cc(1,:),cc(3,:),cc(2,:));




