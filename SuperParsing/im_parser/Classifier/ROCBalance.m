function b = ROCBalance(classifierOutput, correctClass)
    correctClass=correctClass>0;
    npos=sum(correctClass==1);
    nneg=sum(correctClass==0);
    
    [classifierOutput,ind] = sort(classifierOutput);
    correctClass       = correctClass(ind);    
    
    tpr = 1-cumsum(correctClass)/sum(correctClass);
    fpr = 1-cumsum(1-correctClass)/sum(1-correctClass);
    tpr = [1 ; tpr ; 0];
    fpr = [1 ; fpr ; 0];
    figure(1);clf;plot(fpr,tpr);
    n = size(tpr, 1);
    AUC = sum((fpr(2:n) - fpr(1:n-1)).*(tpr(2:n)+tpr(1:n-1)))/2;
    b=[min(classifierOutput)-1;classifierOutput];
    [aux,indice]=min(abs(1-fpr-tpr)); % intersection entre la courbe roc et la diagonale.
    b=b(indice) + eps;
    fprintf('AUC: %.3f b: %.2f\n',AUC,b);
end