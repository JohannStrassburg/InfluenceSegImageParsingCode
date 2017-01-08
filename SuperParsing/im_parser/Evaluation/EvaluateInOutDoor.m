[foo labelset] = fileparts(HOMELABELSETS{UseLabelSet(1)});
folds = dir_recurse(fullfile(HOMEDATA,testParams.TestString,testParams.MRFFold,labelset,'*'),1,0);
good = true;
for i = 1:length(folds)
    files = dir_recurse(fullfile(folds{i},'*.mat'),0,1);
    if(length(files)<length(testFileList))
        good = false;
        fprintf('Eval: missing files in: %s\n',folds{i});
        %break;
    end
end
if(good)
    clear metadata;
    EvaluateTests(HOMEDATA,HOMELABELSETS(UseLabelSet),{testParams.TestString},testParams.MRFFold,[],[],testParams.MRFFold);
    if(exist('inoutEval','var') && inoutEval)
        metadata.inout = 'indoor';
        EvaluateTests(HOMEDATA,HOMELABELSETS(UseLabelSet),{testParams.TestString},[testParams.MRFFold 'indoor'],metadata,[],testParams.MRFFold);
        metadata.inout = 'outdoor';
        EvaluateTests(HOMEDATA,HOMELABELSETS(UseLabelSet),{testParams.TestString},[testParams.MRFFold 'outdoor'],metadata,[],testParams.MRFFold);
    end
end