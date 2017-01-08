function [VW, sptHist] = LMdenseVisualWords(D, HOMEIMAGES, VWparam)
%
% Compute dense visual words and spatial pyramid histogram.
%
% The SIFT grid + visual words will be defined by the parameters VWparam:
%   VWparam.imagesize = 256; % normalized image size (images will be scaled so that the maximal axis has this dimension before computing the sift features)
%   VWparam.grid_spacing = 1; % distance between grid centers
%   VWparam.patch_size = 16; % size of patch from which to compute SIFT descriptor (it has to be a factor of 4)
%   VWparam.NumVisualWords = 200; % number of visual words
%   VWparam.Mw = 2; % number of spatial scales for spatial pyramid histogram
%
% Run demoVisualWords.m to see an example of how it works.
%
% The SIFT descriptor at each location has 128 dimensions and is vector quantized.
%
% This function can be called as:
%
% [VW, sptHist] = LMdenseVisualWords(img, HOMEIMAGES, param);
% [VW, sptHist] = LMdenseVisualWords(D(n), HOMEIMAGES, param);
% [VW, sptHist] = LMdenseVisualWords(filename, HOMEIMAGES, param);
%
%
% Antonio Torralba, 2008


if nargin==4
    precomputed = 1;
    % get list of folders and create non-existing ones
    %listoffolders = {D(:).annotation.folder};
else
    precomputed = 0;
    HOMESIFT = '';
end

if nargin<3
    % Default parameters
    VWparam.grid_spacing = 1; % distance between grid centers
    VWparam.patch_size = 16; % size of patch from which to compute SIFT descriptor (it has to be a factor of 4)
end

Nfeatures = 128;

if isstruct(D)
    % [gist, param] = LMdenseVisualWords(D, HOMEIMAGES, param);
    Nscenes = length(D);
    typeD = 1;
end
if iscell(D)
    % [gist, param] = LMdenseVisualWords(filename, HOMEIMAGES, param);
    Nscenes = length(D);
    typeD = 2;
end
if isnumeric(D)
    % [gist, param] = LMdenseVisualWords(img, HOMEIMAGES, param);
    Nscenes = size(D,4);
    typeD = 3;
end

if Nscenes >1
    fig = figure;
end

% Loop: Compute visual words for all scenes
if isfield(VWparam, 'imagesize')
    n = VWparam.imagesize-VWparam.patch_size+2; m = n;
else
    % read one image to check size. This will only work if all the images
    % have the same size
    switch typeD
        case 1
            img = LMimread(D, 1, HOMEIMAGES);
        case 2
            img = imread(fullfile(HOMEIMAGES, D{1}));
        case 3
            img = D(:,:,:,1);
    end
    n = size(img,1)-VWparam.patch_size+2;
    m = size(img,2)-VWparam.patch_size+2;
end

VW = zeros([n m Nscenes], 'uint16');
sptHist = zeros([VWparam.NumVisualWords*((4^VWparam.Mw-1)/3) Nscenes], 'single');
for n = 1:Nscenes
    g = [];
    todo = 1;
    % otherwise compute gist
    if todo==1
        disp([n Nscenes])

        % load image
        try
            switch typeD
                case 1
                    img = LMimread(D, n, HOMEIMAGES);
                case 2
                    img = imread(fullfile(HOMEIMAGES, D{n}));
                case 3
                    img = D(:,:,:,n);
            end
        catch
            disp(D(n).annotation.folder)
            disp(D(n).annotation.filename)
            rethrow(lasterror)
        end        
        
        % Reshape image to standard format
        if isfield(VWparam, 'imagesize')
            img = imresizecrop(img, VWparam.imagesize, 'bilinear');
        end

        %M = max(size(img,1), size(img,2));
        %if M~=VWparam.imagesize
        %    img = imresize(img, VWparam.imagesize/M, 'bilinear');            
        %end

        % get SIFT descriptors
        sift = single(LMdenseSift(img, HOMEIMAGES, VWparam));
        [nrows ncols nf] = size(sift);
        sift = reshape(sift, [size(sift,1)*size(sift,2) Nfeatures]);
        
        % vector quantization
        [fitd, w] = min(distMat(single(VWparam.visualwordcenters), sift));
        
        VW(:,:,n) = uint16(reshape(w, [nrows ncols]));

        % Compute spatial histogram
        sptHist(:,n) = spatialHistogram(VW(:,:,n), VWparam.Mw, VWparam.NumVisualWords);
        
        
        if Nscenes >1
            figure(fig);
            subplot(121)
            imshow(uint8(img))
            subplot(122)
            imagesc(VW(:,:,n))
            axis('equal')
            axis('off')
        end
    end

    drawnow
end



% 
% function [sift_arr, grid_x, grid_y] = dense_sift(I, SIFTparam)
% % Svetlana Lazebnick
% % Antonio Torralba: modified using convolutions to speed up the
% % computations.
% 
% grid_spacing = SIFTparam.grid_spacing;
% patch_size = SIFTparam.patch_size;
% 
% I = double(I);
% I = mean(I,3);
% I = I /max(I(:));
% 
% % parameters
% num_angles = 8;
% num_bins = 4;
% num_samples = num_bins * num_bins;
% alpha = 9; %% parameter for attenuation of angles (must be odd)
% 
% if nargin < 5
%     sigma_edge = 1;
% end
% 
% angle_step = 2 * pi / num_angles;
% angles = 0:angle_step:2*pi;
% angles(num_angles+1) = []; % bin centers
% 
% [hgt wid] = size(I);
% 
% [G_X,G_Y]=gen_dgauss(sigma_edge);
% 
% % add boundary:
% I = [I(2:-1:1,:,:); I; I(end:-1:end-1,:,:)];
% I = [I(:,2:-1:1,:) I I(:,end:-1:end-1,:)];
% 
% I = I-mean(I(:));
% I_X = filter2(G_X, I, 'same'); % vertical edges
% I_Y = filter2(G_Y, I, 'same'); % horizontal edges
% 
% I_X = I_X(3:end-2,3:end-2,:);
% I_Y = I_Y(3:end-2,3:end-2,:);
% 
% I_mag = sqrt(I_X.^2 + I_Y.^2); % gradient magnitude
% I_theta = atan2(I_Y,I_X);
% I_theta(find(isnan(I_theta))) = 0; % necessary????
% 
% % grid 
% grid_x = patch_size/2:grid_spacing:wid-patch_size/2+1;
% grid_y = patch_size/2:grid_spacing:hgt-patch_size/2+1;
% 
% % make orientation images
% I_orientation = zeros([hgt, wid, num_angles], 'single');
% 
% % for each histogram angle
% cosI = cos(I_theta);
% sinI = sin(I_theta);
% for a=1:num_angles
%     % compute each orientation channel
%     tmp = (cosI*cos(angles(a))+sinI*sin(angles(a))).^alpha;
%     tmp = tmp .* (tmp > 0);
% 
%     % weight by magnitude
%     I_orientation(:,:,a) = tmp .* I_mag;
% end
% 
% % Convolution formulation:
% weight_kernel = zeros(patch_size,patch_size);
% r = patch_size/2;
% cx = r - 0.5;
% sample_res = patch_size/num_bins;
% weight_x = abs((1:patch_size) - cx)/sample_res;
% weight_x = (1 - weight_x) .* (weight_x <= 1);
% 
% for a = 1:num_angles
%     %I_orientation(:,:,a) = conv2(I_orientation(:,:,a), weight_kernel, 'same');
%     I_orientation(:,:,a) = conv2(weight_x, weight_x', I_orientation(:,:,a), 'same');
% end
% 
% % Sample SIFT bins at valid locations (without boundary artifacts)
% % find coordinates of sample points (bin centers)
% [sample_x, sample_y] = meshgrid(linspace(1,patch_size+1,num_bins+1));
% sample_x = sample_x(1:num_bins,1:num_bins); sample_x = sample_x(:)-patch_size/2;
% sample_y = sample_y(1:num_bins,1:num_bins); sample_y = sample_y(:)-patch_size/2;
% 
% sift_arr = zeros([length(grid_y) length(grid_x) num_angles*num_bins*num_bins], 'single');
% b = 0;
% for n = 1:num_bins*num_bins
%     sift_arr(:,:,b+1:b+num_angles) = I_orientation(grid_y+sample_y(n), grid_x+sample_x(n), :);
%     b = b+num_angles;
% end
% clear I_orientation
% 
% 
% % Outputs:
% [grid_x,grid_y] = meshgrid(grid_x, grid_y);
% [nrows, ncols, cols] = size(sift_arr);
% 
% % normalize SIFT descriptors
% 
% %sift_arr = reshape(sift_arr, [nrows*ncols num_angles*num_bins*num_bins]);
% %sift_arr = normalize_sift(sift_arr);
% %sift_arr = reshape(sift_arr, [nrows ncols num_angles*num_bins*num_bins]);
% 
% 
% ct = .1;
% sift_arr = sift_arr + ct;
% tmp = sqrt(sum(sift_arr.^2, 3));
% sift_arr = sift_arr ./ repmat(tmp, [1 1 size(sift_arr,3)]);
% 
% function [GX,GY]=gen_dgauss(sigma)
% 
% % laplacian of size sigma
% %f_wid = 4 * floor(sigma);
% %G = normpdf(-f_wid:f_wid,0,sigma);
% %G = G' * G;
% G = gen_gauss(sigma);
% [GX,GY] = gradient(G); 
% 
% GX = GX * 2 ./ sum(sum(abs(GX)));
% GY = GY * 2 ./ sum(sum(abs(GY)));
% 
% 
% function G=gen_gauss(sigma)
% 
% if all(size(sigma)==[1, 1])
%     % isotropic gaussian
% 	f_wid = 4 * ceil(sigma) + 1;
%     G = fspecial('gaussian', f_wid, sigma);
% %	G = normpdf(-f_wid:f_wid,0,sigma);
% %	G = G' * G;
% else
%     % anisotropic gaussian
%     f_wid_x = 2 * ceil(sigma(1)) + 1;
%     f_wid_y = 2 * ceil(sigma(2)) + 1;
%     G_x = normpdf(-f_wid_x:f_wid_x,0,sigma(1));
%     G_y = normpdf(-f_wid_y:f_wid_y,0,sigma(2));
%     G = G_y' * G_x;
% end


function D=distMat(P1, P2)
%
% Euclidian distances between vectors

if nargin == 2
    X1=repmat(single(sum(P1.^2,2)),[1 size(P2,1)]);
    X2=repmat(single(sum(P2.^2,2)),[1 size(P1,1)]);
    R=P1*P2';
    D=X1+X2'-2*R;
else
    % each vector is one column
    X1=repmat(sum(P1.^2,1),[size(P1,2) 1]);
    R=P1'*P1;
    D=X1+X1'-2*R;
    D = sqrt(D);
end



function h = spatialHistogram(W, Mw, Nwords)
% Mw = number of spatial windows for computing histograms
coef = 1./[2^(Mw-1) 2.^(Mw-(1:(Mw-1)))];

h = [];
for M = 1:Mw
    lx = round(linspace(1, size(W,2)-1, 2^(M-1)+1));
    ly = round(linspace(1, size(W,1)-1, 2^(M-1)+1));
    for x = 1:2^(M-1)
        for y = 1:2^(M-1)
            ww = W(ly(y)+1:ly(y+1), lx(x)+1:lx(x+1));
            hh = hist(ww(:), 1:Nwords);
            h = [h coef(M)*hh];
        end
    end
end

% store words
h = h /sum(h);




























