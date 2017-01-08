function [imLabeled] = DrawImLabelsVideo(im,labels,labelColors,labelNames,outfile,dLegend,dTrans,figureNo,maxDim,mask,title)
if(exist('mask','var') && isempty(mask))
    clear mask;
end
if(exist('maxDim','var'))
    [ro co ch] = size(labels);
    scaling = min(1,maxDim/max(ro,co));
    if(scaling<1)
        im = imresize(im,scaling,'bicubic');
        labels = imresize(labels,scaling,'nearest');
        if(exist('mask','var'));
            mask = imresize(mask,scaling,'nearest');
        end
    end
end
if(dTrans==404)
    show(im,figureNo),hold on;
    set(gcf,'PaperPositionMode','auto');
    if(~isempty(outfile))
        print(outfile,'-dpng','-r96');%saveas(gcf,outfname);
    end
    hold off;
    imLabeled = im;
    return;
end
allLabels = labelNames;
allColors = labelColors;
unlabs = unique(labels);l2ul = zeros(length(labelNames),1);l2ul(unlabs) = 1:length(unlabs);
labels = l2ul(labels);labelColors = labelColors(unlabs,:);labelNames = labelNames(unlabs);
if(~exist('dLegend','var'));dLegend = 0;end;
if(~exist('dTrans','var'));dTrans = 0;end;
if(~exist('figureNo','var'));figureNo = 1;end;
imLabeled = labelColors(labels,:);
if(exist('mask','var'));
    lineMask = 1==(imdilate(mask,strel('disk',1))-mask);
    imLabeled(lineMask(:),:) = 1;
end
imLabeled = reshape(imLabeled,[size(labels) 3]);
%imLabeled(1,1:end,:) = 0;imLabeled(end,1:end,:) = 0;
%imLabeled(1:end,1,:) = 0;imLabeled(1:end,end,:) = 0;
if(dTrans);imLabeled = repmat(rgb2gray(im2double(im)),[1 1 3])./2 + imLabeled./2;end;
if(dLegend>1)
    tmp = ones([size(imLabeled,1) dLegend size(imLabeled,3)]);
    imLabeled = [imLabeled tmp];
end
show(imLabeled,figureNo),hold on;
if(dLegend)
    for i = 1:length(allLabels);plot([0 0],'LineWidth', 8,'Color',allColors(i,:));end;
    legend(allLabels),hold off;drawnow;
end
if(exist('title','var'))
    text(size(imLabeled,2)/2,size(imLabeled,1),title,'FontSize',24,'HorizontalAlignment','center','VerticalAlignment','bottom');
end
set(gcf,'PaperPositionMode','auto');
hold off;
if(~isempty(outfile))
    %saveas(gcf,outfile);
    print(outfile,'-dpng','-r96');%saveas(gcf,outfname);
end
end