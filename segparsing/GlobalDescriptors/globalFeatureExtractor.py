'''
Created on 23.04.2014
IGNORE THIS FILE ## DOES NOTHING
REIMPLEMENTATION OF GLOBAL FEATURES based on Lazebnik's MATLAB Code to SuperParsing not started
@author: Johann Strassburg
'''

import numpy as np
class globalFeatureExtractor(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''

        pass
        
    
    def buildTinyIm(self, imageFileList, imageBaseDir, dataBaseDir, canSkip ):
        '''
        '''
        #if(nargin<4)
        #    canSkip = 0
        I = np.zeros((256,256,3))
        #tinyIm = imresize(I,[16 16]); tinyIm = tinyIm(:);
        #desc_all = np.zeros(len(imageFileList),len(tinyIm));

#         ##tic
#         ##pfig = ProgressBar('Building Tiny Image');
#         for f in range(len(imageFileList)):
#             #% load image
#             imageFName = imageFileList{f}
#             [dirN base] = fileparts(imageFName)
#             baseFName = [dirN filesep base];
#     outFName = fullfile(dataBaseDir,sprintf('%s_tinyIm.mat',  baseFName));
#     imageFName = fullfile(imageBaseDir, sprintf('%s', imageFName));
#     
#     
#     if(mod(f,100)==0)
#         ProgressBar(pfig,f,length(imageFileList));
#     end
#     if(size(dir(outFName),1)~=0 && canSkip)
#         %fprintf('Skipping file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
#         load(outFName);
#         desc_all(f,:) = tinyIm(:)';
#         continue;
#     end
#     
#     try
#         I = imread(imageFName);
#         if(size(I,3)==1)
#             I = repmat(I,[1 1 3]);
#         end
#         %fprintf('Working on file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
#         %I = im2double(I);
# 
#         tinyIm = imresize(I,[16 16]); 
#         
#         make_dir(outFName);
#         save(outFName, 'tinyIm');
# 
#         desc_all(f,:) = tinyIm(:)';
#     catch ME
#         ME
#     end
# end
# close(pfig);
# end


    def buildLabelHist(self, imageFileList, labelBaseDirs, dataBaseDir, canSkip ):
        '''
        '''
        pass
# close all;
# if(nargin<4)
#     canSkip = 0;
# end
# desc_all = cell(length(labelBaseDirs),1);
# for ls = 1:length(labelBaseDirs)
#     labelBaseDir = labelBaseDirs{ls};
#     [foo labelSet] = fileparts(labelBaseDir);
#     tic
#     pfig = ProgressBar('Building Label Histogram');
#     for f = 1:length(imageFileList)
#         % load image
#         imageFName = imageFileList{f};
#         [dirN base] = fileparts(imageFName);
#         baseFName = [dirN filesep base];
#         outFName = fullfile(dataBaseDir,labelSet,sprintf('%s.mat',  baseFName));
#         labelFName = fullfile(labelBaseDir, sprintf('%s.mat', baseFName));
#         if(mod(f,100)==0)
#             ProgressBar(pfig,f,length(imageFileList));
#         end
#         if(exist(outFName,'file') && canSkip)
#             %fprintf('Skipping file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
#             load(outFName);
#             if(isempty(desc_all{ls}))
#                 desc_all{ls} = zeros(length(imageFileList),length(labelHist));
#             end
#             desc_all{ls}(f,:) = labelHist;
#             continue;
#         end
# 
#         try
#             load(labelFName);
#             if(~exist('S','var')&&exist('L','var'))
#                 S = L;
#                 names = labelList;
#             end
#             [l counts] = UniqueAndCounts(S);
#             %counts(l<1) = [];
#             %l(l<1) = [];
#             labelHist = zeros(1,length(names)+1);
#             labelHist(l+1) = counts;
#             %labelHist = labelHist./sum(labelHist);
#             make_dir(outFName);
#             save(outFName, 'labelHist');
#             
#             if(isempty(desc_all{ls}))
#                 desc_all{ls} = zeros(length(imageFileList),length(labelHist));
#             end
#             desc_all{ls}(f,:) = labelHist;
#         catch ME
#             ME
#         end
#     end
#     close(pfig);
# end
# end


    def buildGist(self, imageFileList, imageBaseDir, dataBaseDir, canSkip, color ):
        '''
        '''
        pass
# %function [ gist_all ] = BuildGist( imageFileList, imageBaseDir,dataBaseDir, canSkip )
# %BUILDGIST Summary of this function goes here
# %   Detailed explanation goes here
# 
# 
# %% Parameters:
# imageSize = 256; 
# orientationsPerScale = [8 8 4];
# numberBlocks = 4;
# 
# if(~exist('color','var'))
#     color=0;
# end
# if(~exist('canSkip','var'))
#     canSkip = 0;
# end
# 
# %% Precompute filter transfert functions (only need to do this one, unless image size is changes):
# %createGabor(orientationsPerScale, imageSize); % this shows the filters
# G = createGabor(orientationsPerScale, imageSize);
# 
# if(color)
#     I = zeros([imageSize imageSize 3]);
# else
#     I = zeros(imageSize,imageSize);
# end
# output = prefilt(I,4);
# gist = gistGabor(output,numberBlocks,G);
# gist_all = zeros(length(imageFileList),size(gist,1));
# 
# tic
# pfig = ProgressBar('Building Gist');
# for f = 1:length(imageFileList)
#     %% load image
#     imageFName = imageFileList{f};
#     [dirN base] = fileparts(imageFName);
#     baseFName = [dirN filesep base];
#     outFName = fullfile(dataBaseDir,sprintf('%s_gist.txt',  baseFName));
#     imageFName = fullfile(imageBaseDir, sprintf('%s', imageFName));
#     
#     if(mod(f,100)==0)
#         ProgressBar(pfig,f,length(imageFileList));
#     end
#     if(size(dir(outFName),1)~=0 && canSkip)
#         %fprintf('Skipping file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
#         gist = load(outFName, '-ascii');
#         gist_all(f,:) = gist;
#         continue;
#     end
#     
#     %try
#         I = load_image(imageFName,color);
#         I = imresizecrop(I,[imageSize imageSize], 'bicubic');
#         %fprintf('Working on file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
#         %I = im2double(I);
# 
#         output = prefilt(I,4);
#         gist = gistGabor(output,numberBlocks,G);
# 
#         make_dir(outFName);
#         save(outFName, 'gist', '-ascii');
# 
#         gist_all(f,:) = gist;
#     %catch ME
#     %    ME
#     %end
# end
# close(pfig);
# 
# end


    def buildColorHist(self, imageFileList, imageBaseDir, dataBaseDir, canSkip ):
        '''
        '''
        pass
# %function [ gist_all ] = BuildGist( imageFileList, imageBaseDir,dataBaseDir, canSkip )
# %BUILDGIST Summary of this function goes here
# %   Detailed explanation goes here
# 
# 
# 
# if(nargin<4)
#     canSkip = 0;
# end
# I = zeros(256,256,3);
# coHist = color_hist( I, ones(size(I,1)*size(I,2),1), ones(size(I,1)*size(I,2),1));
# desc_all = zeros(length(imageFileList),length(coHist));
# 
# tic
# pfig = ProgressBar('Building Color Histogram');
# for f = 1:length(imageFileList)
#     % load image
#     imageFName = imageFileList{f};
#     [dirN base] = fileparts(imageFName);
#     baseFName = [dirN filesep base];
#     outFName = fullfile(dataBaseDir,sprintf('%s_coHist.mat',  baseFName));
#     imageFName = fullfile(imageBaseDir, sprintf('%s', imageFName));
#     
#     
#     if(mod(f,100)==0)
#         ProgressBar(pfig,f,length(imageFileList));
#     end
#     if(size(dir(outFName),1)~=0 && canSkip)
#         %fprintf('Skipping file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
#         load(outFName);
#         desc_all(f,:) = coHist(:)';
#         continue;
#     end
#     
#     try
#         I = imread(imageFName);
#         if(size(I,3)==1)
#             I = repmat(I,[1 1 3]);
#         end
#         %fprintf('Working on file: %s %.2f%% done Time: %.2f/%.2f\n',baseFName,100*f/length(imageFileList),toc/60,toc*length(imageFileList)/f/60);
#         %I = im2double(I);
# 
#         coHist = color_hist( I, true(size(I,1)*size(I,2),1), true(size(I,1)*size(I,2),1));
#         
#         make_dir(outFName);
#         save(outFName, 'coHist');
# 
#         desc_all(f,:) = coHist(:)';
#     catch ME
#         ME
#     end
# end
# close(pfig);
# end



        
        