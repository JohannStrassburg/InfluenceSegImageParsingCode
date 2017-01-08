function [ centers ] = gist_centers( fileList, HOMEIMAGES, dictionarySize )
%GIST_CENTERS Summary of this function goes here
%   Detailed explanation goes here

imageSize = 64; 
orientationsPerScale = [8 8 4];


%% Precompute filter transfert functions (only need to do this one, unless image size is changes):
centers = createGabor(orientationsPerScale, imageSize);


end
