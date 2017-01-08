function [smoothingMatrix] = BuildSmoothingMatrix(labelPenality,labelSmoothing,interLabelSmoothing,labelPenalityFun,interLabelPenalityFun)
if(length(labelSmoothing)==1)
    labelSmoothing = repmat(labelSmoothing,size(labelPenality,1));
end
if(length(interLabelSmoothing)==1)
    interLabelSmoothing = repmat(interLabelSmoothing,[size(labelPenality,1), 1]);
end
for i = 1:size(labelPenality,1)
    for j = 1:size(labelPenality,2)
        if(i==j)
            smoothing = labelSmoothing(i);
            penalityFun = labelPenalityFun;
        else
            smoothing = interLabelSmoothing(i,j);
            penalityFun = interLabelPenalityFun;
        end
        if(strcmp(penalityFun,'pots'))
            labelPenality{i,j} = labelPenality{i,j}>.1;
        elseif(strcmp(penalityFun,'metric'))
            mask = labelPenality{i,j}>0;
            if(length(unique(labelPenality{i,j}(mask)))==1)
                labelPenality{i,j}(mask) = 1;
            else
                labelPenality{i,j}(mask) = labelPenality{i,j}(mask)-min(labelPenality{i,j}(mask));
                labelPenality{i,j}(mask) = .5*labelPenality{i,j}(mask)/max(labelPenality{i,j}(mask));
                labelPenality{i,j}(mask) = .5+labelPenality{i,j}(mask);
            end
        end
        labelPenality{i,j} = labelPenality{i,j}*smoothing;
    end
end
smoothingMatrix = cell2mat(labelPenality);
end