function [fs] = svmShortListSoftMax(SVM,globalDesc,svmDescNames)

for j = 1:length(SVM)
    [fo fs(j,:)] = svmShortList(SVM{j},globalDesc,svmDescNames(j),1);
end

fs = prod(softmax(fs'),2);
