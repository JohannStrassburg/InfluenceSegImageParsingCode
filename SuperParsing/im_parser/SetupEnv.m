if(~exist('DOSETUP','var')||DOSETUP~=0)
    addpath(genpath(HOMECODE));
    
    tmp = pwd;
    cd(fullfile(HOMECODE,'Libraries','anigaussm'));
    mex anigauss.c anigauss_mex.c 
    cd(tmp);


    tmp = pwd;
    cd(fullfile(HOMECODE,'Libraries','segment'));
    mex segmentmex.cpp
    cd(tmp);
    
    tmp = pwd;
    cd(fullfile(HOMECODE,'Libraries','boostDt','boost'));
    mex treevalc.c
    cd(tmp);

    GCO_UnitTest
    
    DOSETUP=0; 
    
end
