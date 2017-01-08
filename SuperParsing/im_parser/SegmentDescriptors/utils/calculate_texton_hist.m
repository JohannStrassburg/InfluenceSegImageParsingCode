function [ descHist ] = calculate_texton_hist( textons, dictionarySize )
%CALCULATE_TEXTON_HIST Summary of this function goes here
%   Detailed explanation goes here

if(~isempty(textons))
    descHist = hist(textons(:),1:dictionarySize)./length(textons);
else
    descHist = zeros(dictionarySize,1);
end