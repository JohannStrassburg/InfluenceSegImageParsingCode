% Modification of Code provided by B. Schauerte 2012,2013:
%   B. Schauerte, R. Stiefelhagen, "How the Distribution of Salient Objects
%   in Images Influences Salient Object Detection". In Proceedings of the
%   20th International Conference on Image Processing (ICIP), 2013.
%
% Reads images from folder 'Images' and stores saliency-images to folder 'segments'
%
%
%% Options
do_visualize_differences    = true; % do you want to visualize the differences between the models?
colormap_arguments          = {'hot'};%{'default'};% % pick your favorite color map
do_save_saliency_maps       = false; % do you want to save the saliency maps?
save_saliency_maps_folder   = 'smaps';
do_save_difference          = false; % do you want to save the difference maps?
save_difference_maps_folder = 'dmaps';

%% Allow for different center bias combinations
%    CB_LINEAR  = 0,
%    CB_PRODUCT = 1
%    CB_MAX     = 2
%    CB_MIN     = 3
cbctids = {'CB_LINEAR','CB_PRODUCT','CB_MAX','CB_MIN'};                                              % the types
f_cbctids_idx = @(s) (sum(strcmp(s,{'CB_LINEAR','CB_PRODUCT','CB_MAX','CB_MIN'}) .* [1 2 3 4]) - 1); % get the index of the string

%% define the methods that you want to compare here
smethods = { ...
%    {'RC'}% ...   % Region Contrast
%    {'LDRC'} ... % Locally Debiased Region Contrast
%    ... % in the following, LDRC with an added/multiplied/min'ed/max'ed center bias
    {'LDRCCB',0.5,50,50,0.5,0.5,0.5,0.5,f_cbctids_idx('CB_PRODUCT')} ...
%    {'LDRCCB',0.5,50,50,0.5,0.5,0.5,0.5,f_cbctids_idx('CB_LINEAR')} ...
%    {'LDRCCB',0.5,50,50,0.5,0.5,0.5,0.5,f_cbctids_idx('CB_MIN')} ...
%    {'LDRCCB',0.5,50,50,0.5,0.5,0.5,0.5,f_cbctids_idx('CB_MAX')} ...
    };
nmethods = numel(smethods);

%% Calculate the saliency maps for all methods
img_dir = 'Images';
files   = dir(fullfile(img_dir,'*.jpg'));
diffs   = zeros(nmethods,nmethods,length(files));
figure('name','image');
for i = 1:length(files)
    I_path = fullfile(img_dir,files(i).name);
    
    I      = imread(I_path);
    I_orig = I;
    %I      = imresize(I,[400 NaN]);
    IS     = im2single(I);
    
    smaps  = cell(1,numel(smethods));
    
    for m=1:nmethods
        tic;
        S=region_saliency_mex(IS,smethods{m}{:});
        [a,b,c] = fileparts(files(i).name)
        outFileName = fullfile('segments',sprintf('%s.mat',b));
        save(outFileName,'S')
        t=toc;
        subplot(3,ceil((nmethods+1)/3),1+m); imshow(mat2gray(S)); colormap(colormap_arguments{:}); title([smethods{m}{1} ' (' num2str(t) ')']);

    end
    drawnow;
end


