function [] = RunBarca(experiment)

if(~exist('HOMECODE','var'))
    HOMECODE = pwd;
end

EXPERIMENT = experiment; %Experiment Folder e.g. Quick_Shift_2_48_0.05

SetupEnv;

RunGTBarcelonaParallel;
