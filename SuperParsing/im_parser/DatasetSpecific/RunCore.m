HOME = 'D:\im_parcer\Core';
HOMECODE = 'D:\Perforce\im_parser\Release';
HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'LabelsAnimalVehicle'),fullfile(HOME,'LabelsSemantic'),fullfile(HOME,'LabelsMaterial'),fullfile(HOME,'LabelsParts')};%fullfile(HOME,'LabelsForgroundBK'), % 
HOMEDATA = fullfile(HOME,'Data');
UseClassifier = [0 0 0 0];



clear testParams;
testParams.LabelSmoothing = {[  2 8 2;
                                4 8 1;
                                8 16 4;
                                2 2 1]}; %% best NN params
testParams.LabelSmoothing = {[  %0 0 0;
                                2 32 4;
                                2 8 4;%8 16 4;%
                                1 8 2;
                                2 2 1]}; %% best classifier params
                            
testParams.LabelSetWeights = [1 2 4 8 16 1 2 4 8 16;
                              1 1 1 1 1  2 2 2 2 2;
                              1 1 1 1 1  1 1 1 1 1;
                              1 1 1 1 1  1 1 1 1 1];
testParams.LabelSetWeights = [1 2 16 1 2 16;
                              1 1 1  2 2 2;
                              1 1 1  1 1 1;
                              1 1 1  1 1 1];  
testParams.LabelPenality = {'conditional'};%'metric','pots'
testParams.InterLabelSmoothing = [0 .25, .5, 1, 2, 4, 8, 16, 32, 64, 128];
testParams.LabelSetWeights = [1; 1; 1; 1];
testParams.LabelSmoothing = [0];
testParams.InterLabelSmoothing = [0];
testParams.labelSubSetsWeights = [0];%8 128
testParams.InterLabelPenality = {'conditional'};%'conditional',,'pots'
testParams.TestString = ['CoreTest' num2str(testSetNum)];
preset=1;
RunFullSystem;
