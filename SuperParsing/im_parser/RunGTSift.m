
if(~exist('HOMECODE','var'))
    HOMECODE = pwd;
end

SetupEnv;

RunSiftFlowECCV10;
