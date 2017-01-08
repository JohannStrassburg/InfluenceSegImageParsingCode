HOME = 'D:\im_parcer\Core';
HOMECODE = 'D:\Perforce\im_parser\Release';
HOMEIMAGES = fullfile(HOME,'Images');
HOMEANNOTATIONS = fullfile(HOME,'Annotations');
HOMELABELSETS = {fullfile(HOME,'LabelsForgroundBK'),fullfile(HOME,'LabelsAnimalVehicle'),fullfile(HOME,'LabelsSemantic'),fullfile(HOME,'LabelsMaterial'),fullfile(HOME,'LabelsParts')};%};%
HOMEDATA = fullfile(HOME,'Data');
UseClassifier = [1 1 1 1 1];%];%

testSetNum = 7;
preset=1;
if(~exist('loadDone','var'))
    LoadData;
end
loadDone = true;

clear testParams;
testParams.K = K;
testParams.segmentDescriptors = {'centered_mask_sp','bb_extent','pixel_area',...'centered_mask', %Shape
    'absolute_mask','top_height',...'bottom_height', %Location
    'int_text_hist_mr','dial_text_hist_mr',...'top_text_hist_mr','bottom_text_hist_mr','right_text_hist_mr','left_text_hist_mr' %Texture
    'sift_hist_int_','sift_hist_dial','sift_hist_bottom','sift_hist_top','sift_hist_right','sift_hist_left'... %Sift
    'mean_color','color_std','color_hist','dial_color_hist',... %Color
    'color_thumb','color_thumb_mask','gist_int'}; %Appearance
testParams.TestString = ['ClassifierValidationTest' num2str(testSetNum)];

PreComputClassifierOutput(HOMEDATA,HOMELABELSETS,testFileList,classifiers,testParams);
EvaluateClassifiers(HOMEDATA,HOMELABELSETS,testFileList,testIndex,Labels,classifiers,testParams)