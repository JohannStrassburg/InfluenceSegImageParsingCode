function [clConfidenceMap B] = MakeClassifierConfidenceMap(classifierOutput,labels,binSize,disp)

if(~exist('binSize','var'))
    binSize = std(classifierOutput)/5;
end
if(~exist('disp','var'))
    disp = 1;
end
minCO = min(classifierOutput);
maxCO = max(classifierOutput);
minInBin = 5;%length(classifierOutput)/100;

bins = minCO:binSize:maxCO;
clConfidenceMap = [minCO 0];
for i = 2:length(bins)
    map = classifierOutput>clConfidenceMap(end,1)&classifierOutput<=bins(i);
    if(sum(map)<minInBin)
        continue;
    end
    clConfidenceMap = [clConfidenceMap; [bins(i) sum(labels(map))./sum(map)]];
end

B=mnrfit(classifierOutput,~labels+1);

if(disp)
    plot(clConfidenceMap(:,1),clConfidenceMap(:,2),'r.-');hold on;
    CORange = minCO:(maxCO-minCO)/50:maxCO;
    Val = mnrval(B,CORange(:));
    plot(CORange,Val(:,1),'b-');
    plot(classifierOutput,[labels],'g.');
    legend({'Bin Confidence','Logistic Regression','Ground Truth'},'Location','NorthWest');hold off;
end