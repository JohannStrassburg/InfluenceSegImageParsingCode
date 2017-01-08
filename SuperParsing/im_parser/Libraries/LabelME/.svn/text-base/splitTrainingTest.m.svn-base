function [jtraining, jtest, jvalidation] = splitTrainingTest(class, ptrain, ptest, pval)

c = unique(class);
Nclasses = length(c);

jtraining = [];
jtest = [];
jvalidation = [];


for n = 1:Nclasses
    j = find(class == c(n));
    N = length(j);
    
    r = randperm(N);
    j = j(r);
    
    if ptrain<1
    else
        jtraining = [jtraining j(1:ptrain)];
        if nargin < 3
            jtest = [jtest j(ptrain+1:end)];
        else
            jtest = [jtest j(ptrain+1:ptrain+ptest)];
            if nargin == 4
                jvalidation = [jvalidation j(ptrain+ptest+1:ptrain+ptest+pval)];
            end
        end
    end
end


