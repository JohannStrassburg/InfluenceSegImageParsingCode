function [AUC cutoffPt] = ROC(scores,classes,labels,WEBBASE,labelset,steps,plotsPerGraph)

if(~exist('steps','var'))
    steps = 1000;
end
if(~exist('plotsPerGraph','var'))
    plotsPerGraph = 10;
end

numLabels = length(labels);
pts = zeros(steps,2,numLabels);
ranges = zeros(steps,numLabels);
AUC = zeros(numLabels,1);
cutoffPt = zeros(numLabels,1);
for l = 1:numLabels
    score = scores(:,l);
    class = classes==l;
    scoreMin = min(score);
    scoreMax = max(score);

    stepSize = (scoreMax-scoreMin)/(steps-1);
    numPos = sum(class==1);
    numNeg = sum(class==0);
    range = scoreMax:-stepSize:scoreMin;
    ranges(:,l) = range(:);
    for i = range
        res = score>=i;
        falsePos = sum(res(:)==1 & class(:)==0)./numNeg;
        truePos = sum(res(:)==1 & class(:)==1)./numPos;
        pts(range==i,:,l) = [falsePos truePos];
        if(falsePos>(1-truePos) && cutoffPt(l) == 0)
            cutoffPt(l) = i;
        end
    end
    n = size(pts,1);
    AUC(l) = sum((pts(2:n,1,l) - pts(1:n-1,1,l)).*(pts(2:n,2,l)+pts(1:n-1,2,l)))/2;
end
colorOrder = get(gca,'ColorOrder');
lineStyleOrder = {'-','--',':','-.'};

webFile = fullfile(WEBBASE,'index.htm');
make_dir(webFile);
fid = fopen(webFile,'a');
fprintf(fid,'\n<h2><a href=%s>%s</a><h2>\n',[labelset '\index.htm'],labelset);
fprintf(fid,'\n<table border="0">\n');
fprintf(fid,'\t<tr>\n');
webFile = fullfile(WEBBASE,labelset,'index.htm');
make_dir(webFile);
fidLS = fopen(webFile,'w');
fprintf(fidLS,'\n<table border="0">\n');
fprintf(fidLS,'\t<tr>\n');

[foo lOrder] = sort(AUC,'descend');


numGroups = ceil(numLabels/plotsPerGraph);
for i = 1:numGroups
    labelGroup = lOrder((1+(i-1)*plotsPerGraph):min(numLabels,(i*plotsPerGraph)));labelGroup = labelGroup(:)';
    figure(i);
    clf;
    hold on;
    
    c = 1;ls = 1;
    for l = labelGroup
        plot(pts(:,1,l),pts(:,2,l),lineStyleOrder{ls},'color',colorOrder(c,:));
        c = c+1;
        if(c>size(colorOrder,1))
            c = 1; ls = ls+1;
            if(ls>length(lineStyleOrder)); ls = 1; end
        end
    end
    legend(labels(labelGroup));
    c = 1;ls = 1;
    for l = labelGroup
        range = ranges(:,l);
        [foo ind] = min(abs(range));
        plot(pts(ind,1,l),pts(ind,2,l),'.','color',colorOrder(c,:));
        c = c+1;
        if(c>size(colorOrder,1))
            c = 1; ls = ls+1;
            if(ls>length(lineStyleOrder)); ls = 1; end
        end
    end

    plot([0 1],[1 0],'color','k');

    %set(gca,'XGrid','on');
    %set(gca,'YGrid','on');
    %set(gca,'YTick',0:1:maxy);
    %set(gca,'XTick',0:1:maxx);
    set(gca,'DataAspectRatioMode','manual');
    set(gca,'DataAspectRatio',[1 1 1]);
    scrsz = get(0,'ScreenSize');
    dispSize = 600;
    set(gcf,'Position',[0 scrsz(4)-(dispSize+120) (dispSize) (dispSize)]);
    hold off;
    plotFileName = sprintf('Plot%d.png',i);
    set(gcf,'PaperPositionMode','auto') 
    print('-dpng','-r0',fullfile(WEBBASE,labelset,plotFileName));
    %saveas(gcf,fullfile(WEBBASE,labelset,plotFileName));
    fprintf(fid,'\t\t<td><img width="%d" src="%s/%s">\n',dispSize,labelset,plotFileName);
    fprintf(fidLS,'\t\t<td><img width="%d" src="%s">\n',dispSize,plotFileName);
    for l = labelGroup
        fprintf(fid,'\t\t<br>%s: AUC: %.3f Cut-off: %.4f',labels{l},AUC(l),cutoffPt(l));
        fprintf(fidLS,'\t\t<br>%s: AUC: %.3f Cut-off: %.4f',labels{l},AUC(l),cutoffPt(l));
    end
    fprintf(fid,'\t\t </td>\n');
    fprintf(fidLS,'\t\t </td>\n');
end

fprintf(fid,'\t</tr>\n');
fprintf(fid,'\n</table>\n');
fprintf(fidLS,'\t</tr>\n');
fprintf(fidLS,'\n</table>\n');

clConfidenceMap = cell(length(lOrder),1);
B = zeros(2,length(lOrder));
for i = lOrder(:)'
    figure(1);clf;
    [clConfidenceMap{i} B(:,i)] = MakeClassifierConfidenceMap(scores(:,i),classes(:)==i);
    scrsz = get(0,'ScreenSize');
    dispWidth = 600;
    dispHeight = 100;
    set(gcf,'Position',[0 scrsz(4)-(dispHeight+120) (dispWidth) (dispHeight)]);
    hold off;
    plotFileName = sprintf('Confidence%s.png',FixLabelName(labels{i}));
    set(gcf,'PaperPositionMode','auto') 
    print('-dpng','-r0',fullfile(WEBBASE,labelset,plotFileName));
    %saveas(gcf,fullfile(WEBBASE,labelset,plotFileName));
    fprintf(fidLS,'<br>%s: AUC: %.3f Cut-off: %.4f:\n',labels{i},AUC(i),cutoffPt(i));
    fprintf(fidLS,'<br><img src="%s"><br>\n',plotFileName);
end




