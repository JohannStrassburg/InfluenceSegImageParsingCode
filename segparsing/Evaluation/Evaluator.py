'''
Created on 25 Apr 2014

@author: johann strassburg
Reimplementation of Evaluation, based on Lazebnik et al.'s SuperParsing paper (MATLAB-code)

WARNING: CODE IS UNFINISHED, USE AT YOUR OWN RISK
'''

import Loader
import numpy as np
from utils.utils import Logger, write_dict_to_matfile, read_dict_from_matfile,\
    read_arr_from_matfile, write_arr_to_file, write_arr_to_matfile
from scipy.spatial.distance import cdist
import os.path
from numpy import exp

class Evaluator(object):
    '''
    classdocs
    '''


    def __init__(self, loader):
        '''
        Constructor
        '''
        self.loader = loader
        
        self.log = Logger()
        
        
        #image variables
        self.image_array_list = self.loader.image_array_list
        self.image_path_list = self.loader.image_path_list
        self.image_list = self.loader.image_list
        
        #global descriptor variables
        self.global_descriptor_dict = self.loader.global_descriptor_dict
        self.centers = self.loader.centers
    
        #superpixels variables
        self.sp_array_list = self.loader.sp_array_list
        
        
        #segment descriptors variables
        self.segment_descriptor_dict = self.loader.segment_descriptor_dict
        self.feature_list = self.loader.feature_list
        
        #image label variables
        self.image_label_array_list = self.loader.image_label_array_list
        
        self.semantic_labels = self.loader.semantic_labels
        self.segment_label_index = self.loader.segment_label_index
        
        #evaluation variables
        self.test_file_names = self.loader.test_file_names
        self.test_file_indices = self.loader.test_file_indices
        self.train_file_indices = self.loader.train_file_indices
        
        #retrieval_set variables
        self.rank_set = {}
        
        print "test"
        
        self.labels = self.loader.labels
        
        self.search_rs = {}
        
        
#         #segment descriptors variables
#         self.feature_list = ['sift_hist_int_', 'sift_hist_left', 
#                              'sift_hist_right', 'sift_hist_top', 'top_height',
#                              'absolute_mask', 'bb_extent',
#                              'centered_mask_sp', 'color_hist',
#                              'color_std', 'color_thumb', 'color_thumb_mask',
#                              'dial_color_hist', 'dial_text_hist_mr',
#                              'gist_int', 'int_text_hist_mr', 
#                              'mean_color', 'pixel_area',
#                              'sift_hist_bottom', 'sift_hist_dial']
#         self.segment_descriptor_dict_path = self.save_path + \
#                                             "segment_descriptor_dict.mat"

        self.adj_file_list = self.loader.adj_file_list

    def parse_test_images(self, base_path, retrieval_path, descriptor_path, label_sets):
        '''
        '''
        
        
        homelabelsets = [1, 2]
#execute: #ParseTestImages(HOMEDATA, HOMEDATA,    HOMELABELSETS(UseLabelSet),testFileList,  testGlobalDesc,  trainFileList,  trainGlobalDesc,  trainIndex(UseLabelSet),  trainCounts(UseLabelSet),  labelPenality(UseLabelSet,UseLabelSet),  Labels(UseLabelSet),  classifierTemp(UseLabelSet),  globalSVMTemp(UseLabelSet,:),  testParams,  fullSPDesc(UseLabelSet));
#function: ParseTestImages(HOMEDATA, HOMETESTDATA,HOMELABELSETS,             testFileList,  testGlobalDesc,  trainFileList,  trainGlobalDesc,  trainIndex,               trainCounts,               labelPenality,                           Labels,               classifiers,                  globalSVM,                     testParams,  fullSPDesc)

# UseLabelSet = [1 2]
#HOMEDATA, HOMEDATA,    HOMELABELSETS(UseLabelSet),testFileList,  testGlobalDesc,  trainFileList,  trainGlobalDesc,  trainIndex(UseLabelSet),  trainCounts(UseLabelSet),  labelPenality(UseLabelSet,UseLabelSet),  Labels(UseLabelSet),  classifierTemp(UseLabelSet),  globalSVMTemp(UseLabelSet,:),  testParams,  fullSPDesc(UseLabelSet)
#execute: #ParseTestImages(HOMEDATA, HOMEDATA,    HOMELABELSETS(UseLabelSet),
#testFileList: FileList (path) to test files
#testGlobalDesc: struct of global Descriptors to test files: 'spatialPryScaled', 'colorGist', 'coHist' je 200x4200/960/24
#trainFileList: FileList (path) to train files (2488)
#trainGlobalDesc: struct of global Descriptors to train files: 'spatialPryScaled', 'colorGist', 'coHist' je 2488x4200/960/24  
#trainIndex(UseLabelSet): 2x1 struct to index arrays of specified Labels (Geo, Semantic), each with 'label', 'sp', 'spSize', 'image'
#trainCounts(UseLabelSet): 2x1 struct of counts per label
#labelPenality(UseLabelSet,UseLabelSet): 2x2 cell (3x3, 3x33; 33x3, 33x33)  
#Labels(UseLabelSet): 1x2 cell:
#    #sky, horizontal, vertical
#    #['awning','balcony', 'bird', 'boat', 'bridge', 'building', 'car', 'cow', 'crosswalk', 'desert', 'door', 'fence', 'field', 'grass', 'moon', 'mountain', 'person', 'plant', 'pole', 'river', 'road', 'rock', 'sand', 'sea', 'sidewalk', 'sign', 'sky', 'staircase', 'streetlight', 'sun', 'tree', 'window']
#classifierTemp(UseLabelSet): 1x2 cell: 1: names, h0, wcs (Geo...labeled), 2: []
#globalSVMTemp(UseLabelSet,:): [] []
#testParams: testParams
#fullSPDesc(UseLabelSet)): 2x1: 1:[] 2:segDesc





#ParseTestImages(HOMEDATA,HOMETESTDATA,HOMELABELSETS,testFileList,testGlobalDesc,trainFileList,trainGlobalDesc,trainIndex,trainCounts,labelPenality,Labels,classifiers,globalSVM,testParams,fullSPDesc)
################# Irrelevant - 'fullSPDesc' is already loaded ################## 
# DataDir = fullfile(HOMETESTDATA,testParams.TestString);
# if(~exist('fullSPDesc','var'))
#     fullSPDesc = cell(length(HOMELABELSETS),length(testParams.K));
# end
################################################################################
        train_index = {}
        for key in self.train_file_indices: # self.loader.label_thread_index.keys() #Load train data from files or label_thread
            train_index[key] = self.loader.label_thread_index[key]
        
        test_index = {}
        for key in self.test_file_indices: # self.loader.label_thread_index.keys() #Load train data from files or label_thread
            test_index[key] = self.loader.label_thread_index[key]
        
        bincount_train = np.array([np.bincount(value) for key,value in train_index])
        train_counts = np.array([sum([len(bincount==i) 
                                      for bincount in bincount_train]) 
                                 for i in np.array(
                                                range(len(self.labels[1])))+1])
        train_counts = np.array([np.bincount(value) for key,value in train_index])
        
        prob_type = 'ratio'; #probType = testParams.probType
        
        gl_suffix = ''

        for im_num in self.test_file_indices:
            
            #im = imread(fullfile(HOMEDATA,'..','Images',testFileList{i}));
            im = self.image_array_list[im_num]
            (ro,co,ch) = im.shape
            #[folder file] = fileparts(testFileList{i});
            #baseFName = fullfile(folder,file);
            
            #calculate Retrieval Set:
            (ret_inds, rank) = self.find_retrieval_set(im_num, retrieval_path)
            self.rank_set[im_num] = rank
         
            
                        

            short_list_mask = {}
            for label_type in len(label_sets):
                
                short_list_ind = np.ones(self.labels[label_type].shape) #Check!!! Labels! (horizontal...sky,tree...)
                short_list_mask[label_type]= short_list_ind == 1
                ###################### No SVM I guess --- is empty!!!!  ##########################
                #                 if(~isempty(globalSVM{labelType}))
                #                     svmstr = testParams.SVMType;
                #                     if(isfield(testParams,'SVMSoftMaxCutoff'))
                #                         fs = svmShortListSoftMax(globalSVM(labelType,:),SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs);
                #                         shortListInds = fs>=testParams.SVMSoftMaxCutoff{labelType};
                #                         if(all(~shortListInds))
                #                             [foo ind] = max(fs);
                #                             shortListInds(ind) = 1;
                #                 end
                ################################################################################
                ##################### SVMType == SVMVal ########################################
                #             elseif(strcmp(testParams.SVMType,'SVMPerf'))
                #                 dataFile = fullfile(HOMELABELSETS{labelType},folder,[file '.mat']);
                #                 load(dataFile);
                #                 shortListInds = zeros(size(names));
                #                 ind = unique(S(:));
                #                 ind(ind<1) = [];
                #                 shortListInds(ind)=true;
                #             elseif(strcmp(testParams.SVMType,'SVMTop10'))
                #                 [~, ind] = sort(trainCounts{labelType},'descend');
                #                 shortListInds = zeros(size(trainCounts{labelType}));
                #                 shortListInds(ind(1:min(end,10)))=true;
                #             elseif(strcmp(testParams.SVMType,'SVMLPBP'))
                #                 svmstr = [testParams.SVMType num2str(testParams.svmLPBPItt)];
                #                 [shortListInds] = svmShortListLPBPSoftMax(globalSVM{labelType}(testParams.svmLPBPItt),SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs);
                ################################################################################
                ######################## irrelevant!!! ##############################################
                #             elseif(length(globalSVM{labelType}) ==2)
                #                 svmOutput = [];
                #                 [shortListInds svmOutput.svm]  = svmShortList(globalSVM{labelType}{1},SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,0);
                #                 [shortListInds fs] = svmShortList(globalSVM{labelType}{2},svmOutput,{'svm'},0);
                #             else
                #                 shortListInds = svmShortList(globalSVM{labelType},SelectDesc(testGlobalDesc,i,1),testParams.SVMDescs,0);
                #             end
                
                
                ######################## no SVMOutput ##########################################
                #         elseif(isfield(testParams,'SVMOutput'))
                #             shortListInds = testParams.SVMOutput{labelType}(i,:)>0;
                #             minSL = 1;
                #             if(sum(shortListInds)<minSL)
                #                 [a inds] = sort(testParams.SVMOutput{labelType}(i,:),'descend');
                #                 shortListInds(inds(1:minSL)) = true;
                #             end
                #             svmstr = [testParams.SVMType 'Min' num2str(minSL)];
                #             Labels{labelType}(shortListInds)
                #         else
                ####################################################################
            

            
                        
                #             shortListInds = ones(size(Labels{labelType}));
                #         end
                #         shortListMask{labelType} = shortListInds==1;
                #     end
                
                
                ########################## no RetrievalMetaMatch ###############################     
                #     if(isfield(testParams,'RetrievalMetaMatch'))
                #         if(strcmp('GroundTruth',testParams.RetrievalMetaMatch))
                #             svmstr = [svmstr 'RetGT'];
                #             metaFields = fieldnames(testParams.TrainMetadata);
                #             totalMask = ones(size(retInds))==1;
                #             for f = 1:length(metaFields)
                #                 mask = strcmp(testParams.TrainMetadata.(metaFields{f})(retInds),testParams.TestMetadata.(metaFields{f}){i});
                #                 totalMask = mask&totalMask;
                #             end
                #             retInds = retInds(totalMask);
                #         end
                #         if(strcmp('SVM',testParams.RetrievalMetaMatch))
                #             svmstr = [svmstr 'RetSVM'];
                #             metaFields = fieldnames(testParams.TrainMetadata);
                #             totalMask = ones(size(retInds))==1;
                #             for f = 1:length(metaFields)
                #                 mask = strcmp(testParams.TrainMetadata.(metaFields{f})(retInds),testParams.TestMetadataSVM.(metaFields{f}){i});
                #                 totalMask = mask&totalMask;
                #             end
                #             retInds = retInds(totalMask);
                #         end
                #     end
                ################################################################################
            
            fg_set = 0
            k_ndx = 1
            classifier_str = np.tile(np.array(['0']), [1, len(label_sets)])
            
            test_im_sp_desc = {}
            #get descriptors for test image
            for feature in self.feature_list:
                test_im_sp_desc[feature] = \
                        self.segment_descriptor_dict[feature][im_num]
            
            im_sp = self.sp_array_list[im_num]
            prob_per_label = {}
            data_cost = {}
            
            #     FGSet = 0;Kndx = 1;
            #     classifierStr = repmat('0',[1 length(HOMELABELSETS)]);
            #     preStr = '';
            #     probSuffix = sprintf('K%d',testParams.K(Kndx));
            #     clear imSP testImSPDesc;
            #     [testImSPDesc imSP] = LoadSegmentDesc(testFileList(i),[],HOMETESTDATA,testParams.segmentDescriptors,testParams.K,testParams.segSuffix);
            #     probPerLabel = cell(size(HOMELABELSETS));
            #     dataCost = cell(size(probPerLabel));
            
            
            
            for label_type in len(label_sets):
                
                label_set = label_sets[label_type]
                
            #     for labelType=1:length(HOMELABELSETS)
            #         [foo labelSet] = fileparts(HOMELABELSETS{labelType});
            
            
            #         if(isempty(classifiers{labelType})) #emtpy for SemanticLabels
            #             if(testParams.retSetSize == length(trainFileList) )
            #                 retSetIndex = trainIndex{labelType,Kndx};
            #             else
            #                 [retSetIndex descMask] = PruneIndex(trainIndex{labelType,Kndx},retInds,testParams.retSetSize,testParams.minSPinRetSet);
            #             end
                (ret_set_ind, desc_mask) = self.prune_index(train_index, ret_inds, [200], 1500)(np.zeros((123,1)),np.zeros((123,1)))#prune_index()
            
                        
                
                label_nums = range(len(train_counts)) 
                prob_per_label[label_type] = self.get_all_prob_per_label(base_path, self.image_list[im_num], ret_set_ind, np.array([]), label_nums, train_counts, type, 1, 1)

                if len(prob_per_label[label_type])>0 & prob_per_label[label_type].shape[0]!=test_im_sp_desc['sift_hist_dial'].shape[0]:
                    prob_per_label[label_type] = np.array([]) 


                if len(prob_per_label[label_type])==0:
                    raw_nns = self.do_rnn_search(test_im_sp_desc, np.array([]),base_path, self.image_list[im_num])
                    
            
                ret_set_sp_desc = self.load_segment_desc(self.train_file_indices, ret_set_ind, descriptor_path, self.feature_list)

                    
#                                         retSetSPDesc = [];
#                                         for dNdx = 1:length(testParams.segmentDescriptors)
#                                             retSetSPDesc.(testParams.segmentDescriptors{dNdx}) = fullSPDesc{labelType,Kndx}.(testParams.segmentDescriptors{dNdx})(descMask,:);
#                                         end
#                                     end
#                                 end

                (raw_nns, total_matches) = self.do_rnn_search(test_im_sp_desc, ret_set_sp_desc, base_path, self.image_list[im_num])

                prob_per_label[label_type] = self.get_all_prob_per_label(base_path, self.image_list[im_num], ret_set_ind, raw_nns, label_nums, train_counts, prob_type,1, 1)
      
            
            
                #NORMALIZATION
                #%nomalize the datacosts for mrf. This is especiall important 
                #when using some classifier or when some labelsets are 
                #under represented
                prob_per_label[label_type]\
                    [:,short_list_mask[label_type]==False] = min(prob_per_label[label_type].flatten(1))-1
                data_cost[label_type] = 1000*(1-1.0/\
                    (1+np.exp(-0.1*prob_per_label[label_type]+0))) 
                #maxPenalty testparam is 1000, BConst is [0; .1]
                data_cost[label_type]\
                    [:,short_list_mask[label_type]==False] = 1000 #maxPenalty
            
                ###### if you use Classifier #######################################
                #             probCacheFile = fullfile(DataDir,'ClassifierOutput',[labelSet testParams.CLSuffix],[baseFName  '.mat']);
                #             classifierStr(labelType) = '1';
                #             if(~exist(probCacheFile,'file'))
                #                 features = GetFeaturesForClassifier(testImSPDesc);
                #                 prob = test_boosted_dt_mc(classifiers{labelType}, features);
                #                 make_dir(probCacheFile);save(probCacheFile,'prob');
                #             else
                #                 clear prob;
                #                 load(probCacheFile);
                #             end
                #             probPerLabel{labelType} = prob;
                #             if(size(prob,2) == 1 && length(Labels{labelType}) ==2)
                #                 probPerLabel{labelType}(:,2) = -prob;
                #             end
                #             if(labelType == FGSet)
                #                 probPerLabel{labelType}(:,1) = probPerLabel{labelType}(:,1);
                #                 probPerLabel{labelType}(:,2) = probPerLabel{labelType}(:,2);
                #             end
                #             
                #             %nomalize the datacosts for mrf. This is especiall important when using some classifier or when some labelsets are under represented
                #             probPerLabel{labelType}(:,~shortListMask{labelType}) = min(probPerLabel{labelType}(:))-1;
                #             dataCost{labelType} = testParams.maxPenality*(1-1./(1+exp(-(testParams.BConstCla(2)*probPerLabel{labelType}+testParams.BConstCla(1)))));
                #             dataCost{labelType}(:,~shortListMask{labelType}) = testParams.maxPenality;
                #             %for k = 1:size(probPerLabel{labelType},2)
                #             %    temp = mnrval(testParams.BConstCla,probPerLabel{labelType}(:,k));
                #             %    dataCost{labelType}(:,k) = testParams.maxPenality*(1-temp(:,1));
                #             %end
                #         end
                #     end
            
            
            
            
            ####################################################################
            #     
            #     dataCostPix = cell(size(probPerLabel));

            ########################  NOPE  ####################################
            #     if(testParams.PixelMRF)
            #         spTransform = imSP;%SPtoSkel(imSP);%
            #         spTransform(spTransform==0) = size(dataCost{1},1)+1;
            #         for labelType=1:length(HOMELABELSETS)
            #             temp = dataCost{labelType};
            #             temp(end+1,:) = 0;
            #             dataCostPix{labelType} = temp(spTransform,:);
            #         end
            #     end
            ####################################################################
            
            data_cost_pix = {}    
            #     adjFile = fullfile(HOMETESTDATA,'Descriptors',sprintf('SP_Desc_k%d%s',testParams.K,testParams.segSuffix),'sp_adjacency',[baseFName '.mat']);
            adj_file = self.adj_file_list[im_num]
            #     load(adjFile);
            #     useLabelSets = 1:length(HOMELABELSETS);
            #
            mean_sp_colors = np.zeros((prob_per_label.shape[0],ch))
            im_flat = np.reshape(im, (ro*co, ch))
                
            for sp_ndx in range(mean_sp_colors.shape[0]):
                mean_sp_colors[sp_ndx, :] = np.mean(im_flat[
                                            im_sp.flatten(1)==sp_ndx,:],axis=0)
                
                 
                    
            #     meanSPColors = zeros(size(probPerLabel{1},1),ch);
            #     imFlat = reshape(im,[ro*co ch]);
            #     for spNdx = 1:size(meanSPColors,1)
            #         meanSPColors(spNdx,:) = mean(imFlat(imSP(:)==spNdx,:),1);
            #     end
            #     
            
            
            #     endStr = [];
            #     preStr = [preStr probSuffix glSuffix]; %testParams.CLSuffix
            
            
######################    MACHEN!!! ############################################
            for label_type in range(len(label_sets)):
                #testParam.weightbysize is 1
                sp_size = np.bincount(im_sp)[1:]
                data_cost[label_type] = data_cost[label_type]*\
                        np.tile(sp_size.flatten(1),
                        (1, data_cost[label_type].shape[1]))/np.mean(sp_size)
                data_cost[label_type] = int(data_cost[label_type])
                data_cost_pix[label_type] = int(data_cost_pix[label_type])
            #     for labelType=1:length(HOMELABELSETS)
            #         if(testParams.weightBySize)
            #             [foo spSize] = UniqueAndCounts(imSP);
            #             dataCost{labelType} = dataCost{labelType}.*repmat(spSize(:),[1 size(dataCost{labelType},2)])./mean(spSize);
            #         end
            #         dataCost{labelType} = int32(dataCost{labelType});
            #         dataCostPix{labelType} = int32(dataCostPix{labelType});
            #     end
            
            
################################################################################

            #     for labelSmoothingInd = 1:length(testParams.LabelSmoothing) lenght is 1 content is [0]
            
            ################### Nope ###########################################    
            #         if(iscell(testParams.LabelSmoothing))
            #             labelSmoothing = testParams.LabelSmoothing{labelSmoothingInd};
            #             lsStr = myN2S(labelSmoothingInd,3);
            ####################################################################
            label_smoothing = 0
            
                #label_smoothing = ...
            #         else
            #             labelSmoothing = repmat(testParams.LabelSmoothing(labelSmoothingInd),[length(dataCost) 3]);
            #             lsStr = myN2S(labelSmoothing(1),3);
            #         end
            ################inter_label_smoothing_mat = np.tile(0,label_pen)    
            #         for interLabelSmoothing  = testParams.InterLabelSmoothing ##### is [0]
                #inter_label_smoothing_mat = ...
            #             interLabelSmoothingMat = repmat(interLabelSmoothing,size(labelPenality,1));
            
            #             for lPenNdx = 1:length(testParams.LabelPenality) # is ['conditional']
            
            
            #                 if((sum(sum(labelSmoothing==0))==numel(labelSmoothing))&&lPenNdx>1); continue; end lPenNdx == 1
            
            #                 for ilPenNdx = 1:length(testParams.InterLabelPenality) ['pots']
            
            #                     if(all(interLabelSmoothingMat(:)==0)&&ilPenNdx>1); continue; end # True Irrelevant???
            
            #                     ilsStr = myN2S(interLabelSmoothing,3);
            #                     
            #                     etNdx = 1; epNdx = 1;
            
            #                     if(~testParams.PixelMRF) #TRUE
            #                         endStr = sprintf('Seg WbS%s', num2str(testParams.weightBySize)); #That's it
            #                     end
            
            
            #                     testName = sprintf('%s C%s %s S%s IS%s P%s IP%s %s%s',preStr,classifierStr,testParams.NormType,lsStr,ilsStr,...
            #                         testParams.LabelPenality{lPenNdx}(1:3),testParams.InterLabelPenality{ilPenNdx}(1:3),svmstr,endStr);
            
            
            
            #                                                 
            #                     %testName = ['Pixel ' testName];
            
            #                     interLabelSmoothingMattemp = interLabelSmoothingMat; # 2x2 mat of 0
            
            ######################################## No ########################
            #                     if(FGSet>0 && interLabelSmoothingMat(FGSet,FGSet) < 1 && length(HOMELABELSETS)>1)
            #                         interLabelSmoothingMattemp(FGSet,:) = 1;
            #                         interLabelSmoothingMattemp(:,FGSet) = 1;
            #                     end
            ####################################################################
            
            
            
            # 
            #                     smoothingMatrix = BuildSmoothingMatrix(labelPenality,labelSmoothing(:,1),interLabelSmoothingMattemp,testParams.LabelPenality{lPenNdx},testParams.InterLabelPenality{ilPenNdx});
            
            #                     mrfParams = [];
            #                     mrfParams.labelSubSetsWeight = 0;
            #                     mrfParams.edgeType = testParams.edgeType{etNdx};
            #                     mrfParams.edgeParam = testParams.edgeParam(epNdx);
            #                     mrfParams.maxPenality = testParams.maxPenality;
            #                     mrfParams.connected = testParams.connected;
            #                     timing(i,4)=toc;
            
            #                     if(~testParams.PixelMRF) #  YES
            
            #ls = self.multi_level_seg_mrf(home_data, home_label_sets, test_name, base_name, labels, im_sp, adj_pairs, data_cost, smoothing_matrix, label_subsets, label_sub_sets_weight)
            #                         [Ls Lsps] = MultiLevelSegMRF(fullfile(DataDir,testParams.MRFFold),HOMELABELSETS(useLabelSets),testName,baseFName,Labels,imSP,adjPairs,dataCost,smoothingMatrix,0,0);
            #                     else # NO
            #                         [Ls Lsps] = MultiLevelPixMRF(fullfile(DataDir,testParams.MRFFold),HOMELABELSETS(useLabelSets),testName,baseFName,Labels,imSP,im,dataCostPix,smoothingMatrix,0,mrfParams);
            #                     end
            #                     timing(i,4)=toc-timing(i,4);
            #                 end
            #             end
            #         end
            #     end
            #     %}

        
#         self.segment_descriptor_dict = {}
#         
#         self.sfe = segmentFeatureExtractor()


    def find_retrieval_set(self, im_num, path):
        '''
        @param path: path to "HOME/DATA/BASE/RETRIEVALSET"
        '''
        
        
# function [retInds rank ranksort] = FindRetrievalSet(trainGlobalDesc,testGlobalDesc,DataDir,fileName,testParams,suffix)
# if(~exist('suffix','var'));suffix = '';end
# outfile = fullfile(DataDir,'RetrievalSet',suffix,[fileName '.mat']);
# if(exist(outfile,'file'))
#     %retInds = []; rank = []; ranksort = [];
#     %return;
#     load(outfile);
        if not os.path.isfile(path+"/"+
                                 self.image_list[im_num]+".mat"):
            keys = self.global_descriptor_dict.keys()
            rank = {}
             
            for key in keys():
                train_global_desc = self.global_descriptor_dict[key][self.train_file_indices]
                test_data = self.global_descriptor_dict[key][im_num]
                n_pts = test_data.shape[0]
                dists = np.zeros((train_global_desc.shape[0],n_pts))
                for d in range(0,n_pts,100):
                    #for d = 1:100:nPts:
                    dists[:,d:min(d+100,n_pts+1)] = cdist(train_global_desc,
                                            test_data[d:min(d+100,n_pts+1),:], 
                                            'euclidean')
                    #dists(:,d:min(d+99,nPts)) = dist2(trainGlobalDesc.(fields{i}),testData(d:min(d+99,nPts),:));
                ndx = np.argsort(dists)
                #[dists ndx] = sort(dists);
                rank[key] = ndx[ndx].reshape(ndx.shape)
            
            write_dict_to_matfile(rank, path+"/"+
                                 self.image_list[im_num]+".mat")
        else:
            rank = read_dict_from_matfile(path+"/"+
                                 self.image_list[im_num]+".mat")
            
            #rank[key] = np.ones(ndx.shape);
            #rank[key](ndx) = 1:size(ndx);
        #save file
# else
#     fields = fieldnames(trainGlobalDesc);
#     rank = [];
#     for i = 1:length(fields)
#         if(~iscell(trainGlobalDesc.(fields{i})))
#             testData = testGlobalDesc.(fields{i});
#             nPts = size(testData,1);
#             dists = zeros(size(trainGlobalDesc.(fields{i}),1),nPts);
#             for d = 1:100:nPts
#                 dists(:,d:min(d+99,nPts)) = dist2(trainGlobalDesc.(fields{i}),testData(d:min(d+99,nPts),:));
#             end
#             [dists ndx] = sort(dists);
#             rank.(fields{i}) = ones(size(ndx));
#             rank.(fields{i})(ndx) = 1:size(ndx);
#         end
#     end
#     make_dir(outfile);save(outfile,'rank');
# end
# 
        fields = ['spatialPryScaled','colorGist','coHist']
        rankfinal = np.ones((len(fields),rank[fields[0]].shape[0]))
        for i in range(len(fields)):
            rankfinal[i,:] = rank[fields[i]]
            
        rank2 = min(rankfinal,[],1)
        ret_inds = np.argsort(rank2)
        
        return (ret_inds, rank)
# fields = testParams.globalDescriptors;
# rankfinal = ones(length(fields),size(rank.(fields{1}),1));
# for i = 1:length(fields)
#     rankfinal(i,:) = rank.(fields{i});
# end
# 
# rank2 = min(rankfinal,[],1);
# [ranksort retInds] = sort(rank2);

#probPerLabel{labelType} = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,probSuffix,retSetIndex,[],labelNums,trainCounts{labelType,Kndx},probType,testParams.smoothingConst,1); %#ok<AGROW>
    def get_all_prob_per_label(self, path, base_name, ret_set_index, raw_nns, filtered_labels, filt_lab_counts, type, smoothing_const, canskip):
        '''
        '''
        
#probPerLabel{labelType} = GetAllProbPerLabel(fullfile(DataDir,labelSet),baseFName,probSuffix,retSetIndex,[],labelNums,trainCounts{labelType,Kndx},probType,testParams.smoothingConst,1); %#ok<AGROW>

# function [probPerLabel,totalTime,probPerDescPerLabel] = GetAllProbPerLabel(HOMEDATA,baseFName,suffix,index,rawNNs,filteredLabels,filtLabCounts,type,smoothingConst,canSkip)
#
################# 

# if(~exist('canSkip','var'))
#     canSkip = 1;
# end

# if(type==0) ######## Type is ratio
#     type='sum';
# elseif(type==1)
#     type='ratio';
# end

        cur_index_len = len(ret_set_index['image'])
# curIndexLength = length(index.image);


#     outfilename = fullfile(HOMEDATA,['probPerLabel' suffix],[baseFName '.mat']); # YES
#         if False:
#             #test if file exists and load it
#             pass
#         else:
            
        if os.path.isfile(path + '/' + base_name + '.mat'):
            out_dict = read_dict_from_matfile(path + '/' + base_name + '.mat')
            if out_dict['indexLenght']:
                index_len = out_dict['indexLenght']
                if index_len == cur_index_len:
                    if out_dict['probPerDescPerLabel']:
                        prob_per_label = out_dict['ProbPerLabel']
                        if prob_per_label.shape[1] == len(filtered_labels):
                            prob_per_desc_per_label = out_dict['probPerDescPerLabel']
                            return prob_per_label, prob_per_desc_per_label
# if(exist(outfilename,'file')&&canSkip)
#     load(outfilename);
#     if(exist('indexLength','var') && indexLength == curIndexLength) %#ok<NODEF>
#         if(exist('probPerDescPerLabel','var'))
#             if(size(probPerLabel,2)==length(filteredLabels))
#                 return;
#             end
#         end
#     end
# end
        if len(raw_nns)==0:
            prob_per_label = []
            prob_per_desc_per_label = []
            return (prob_per_label, prob_per_desc_per_label)
        
        
        
# 
# if(isempty(rawNNs))
#     probPerLabel = [];
#     probPerDescPerLabel=[];
#     return;
# end

        
        prob_per_label = np.zeros(len(raw_nns), len(filtered_labels))
        prob_per_desc_per_label = np.zeros(len(raw_nns), len(filtered_labels), len(raw_nns[0].keys()))
# probPerLabel = zeros(length(rawNNs),length(filteredLabels));
# probPerDescPerLabel = zeros(length(rawNNs),length(filteredLabels),length(fieldnames(rawNNs(1))));
# 
        for i in range(len(raw_nns)):
            (ppl, ppdpl) = self.get_prob_per_label(ret_set_index, raw_nns[i], filtered_labels, filt_lab_counts, type, smoothing_const)
            prob_per_label[i,:] = ppl
            prob_per_desc_per_label[i,:,:] = ppdpl
#     
# for i = 1:length(rawNNs)
#     [probPerLabel(i,:) probPerDescPerLabel(i,:,:)] = GetProbPerLabel(index,rawNNs(i),filteredLabels,filtLabCounts,type,smoothingConst);
# end
        index_len = cur_index_len
        if(canskip):
            out_dict = {}
            out_dict['probPerLabel'] = prob_per_label
            out_dict['probPerDescPerLabel'] = prob_per_desc_per_label
            out_dict['indexLenght'] = index_len
            
            write_dict_to_matfile(out_dict, path + '/' + base_name + '.mat')
            
# totalTime = totalTime+etime(clock,startTime);
# indexLength = curIndexLength;
# if(canSkip)
#     make_dir(outfilename);
#     save(outfilename,'probPerLabel','probPerDescPerLabel','indexLength');
# end
# end


#     [probPerLabel(i,:) probPerDescPerLabel(i,:,:)] = GetProbPerLabel(index,rawNNs(i),filteredLabels,filtLabCounts,type,smoothingConst);
    def get_prob_per_label(self, sp_index, raw_nns, filtered_labels, filt_lab_counts, type, smoothing_const):
        '''
        '''
# function [segLabelProb segDescLabelProb] = GetProbPerLabel(spIndex,rawNNs,filteredLabels,filtLabCounts,type,smoothingConst)
# 
        desc_funs = raw_nns.keys()
        seg_label_prob = np.zeros(len(filt_lab_counts),1)
        seg_desc_label_prob = np.zeros(len(filt_lab_counts),1)
        
        for j in range(len(desc_funs)):
            nns = raw_nns[desc_funs[j]]['nns']
            
            #if 'label' in sp_index.keys():
                
# descFuns = fieldnames(rawNNs);
# segLabelProb = zeros(length(filtLabCounts),1); %ML
# segDescLabelProb = zeros(length(filtLabCounts),length(descFuns));
# 
# for j = 1:length(descFuns)
#     nns = rawNNs.(descFuns{j}).nns;
#     if(isfield(spIndex,'label'))
#         if(strfind(type,'extreme')==1)
                
            #Type is ratio
                
                
                
#             prob = GetProbPerLabelPerDescSimp(spIndex,nns,filteredLabels,filtLabCounts,type,smoothingConst);
#             ps = sort(prob,'descend');
#             wbparams = wblfit(ps(2:min(end,6)));
#             segDescLabelProb(:,j) = log(wblcdf(prob,wbparams(1),wbparams(2))+.001);%./length(descFuns);
#         else
            seg_desc_label_prob[:,j] = np.log(self.get_prob_per_label_per_desc_simp(sp_index, nns, filtered_labels, filt_lab_counts, type, smoothing_const))
#             segDescLabelProb(:,j) = log(GetProbPerLabelPerDescSimp(spIndex,nns,filteredLabels,filtLabCounts,type,smoothingConst))';
#         end
#     end
        seg_label_prob = seg_label_prob + seg_desc_label_prob[:,j]
        return seg_label_prob, seg_desc_label_prob
#     segLabelProb = segLabelProb + segDescLabelProb(:,j);
# end
# end


    def get_prob_per_label_per_desc_simp(self, sp_index, nns, filtered_labels, filt_lab_counts, type, smoothing_const):
        '''
        '''
        pass
   
        sc = smoothing_const
        desc_votes = np.zeros(len(filt_lab_counts),1)
        
        label_num = np.unique(sp_index['label'][nns])
        votes = np.bincount(sp_index['label'][nns])
        
        


        label_num = np.intersect1d(label_num, filtered_labels)

        votes = votes[label_num]

        desc_votes[label_num] = votes

        tv = np.sum(votes)
        tc = np.sum(filt_lab_counts)
        desc_prob = ((sc+desc_votes)/(sc+tv-desc_votes))*\
                                        (tc/filt_lab_counts.flatten(1));

        desc_prob[desc_votes==0] = sc/(tv+sc)
        return desc_prob, desc_votes




    def do_rnn_search(self, im_desc, desc_train, path, base_name):
        '''
        '''
        
        can_skip = 1
        
        
        

        rs = self.search_rs
        target_nn = 80 #testparams
        desc_funs = self.feature_list
        
        file_name = path + '/rNNSearchR200K200TNN80-SPscGistCoHist/' + \
                        base_name+'.mat'
        
        if os.path.isfile(file_name):
            raw_nns = read_arr_from_matfile(file_name, "rawNNs")
            return raw_nns, np.array([])
        else:    
            if len(desc_train)==0:
                raw_nns = np.array([])
                total_matches = np.array([])
                return raw_nns, total_matches
            
            total_matches = 0
            for feature in desc_funs:
                search_desc = desc_train[feature]
                query_desc = im_desc[feature]
                search_r = self.search_rs[feature]['Rs']\
                        [self.search_rs[feature]['numNNs'] == target_nn]
                
                num_queries = query_desc.shape[0]
                dist_all = np.zeros((num_queries, search_desc.shape[0]))
                
                for j in range(np.ceil(num_queries/100.0)):
                    finx = 1+(j-1)*100
                    sinx = min(num_queries,j*100)
                    dist_all[finx:sinx+1,:] = np.sqrt(cdist(
                                    query_desc[finx:sinx+1,:]*1.0,
                                    search_desc*1.0))
                
                for j in range(num_queries):
                    nns,_ = (dist_all[j,:]<=search_r).nonzero()
                    dist = dist_all[j,nns]
                    raw_nns[j][feature]['dists'] = np.sort(dist)
                    ind = np.argsort(dist)
                    raw_nns[j][feature]['nns'] = nns[ind]
                    total_matches = total_matches + \
                            len(raw_nns[j][feature]['nns'])

            
            if can_skip==1:
                write_arr_to_matfile(raw_nns, file_name, 'rawNNs')
        




    def prune_index(self, train_index, image_nns, minImages, minSegments):
        '''
        '''
        

        min_images = len(image_nns)

        min_segments = 0
        min_images = min(min_images, len(image_nns))
        desc_mask = np.zeros(train_index['image'].shape) == 1
        for i in range(min_images):
            desc_mask[train_index['image']==image_nns[i]] = 1
        
        
        m = desc_mask==1
        i = 0
        while len(m[m])<min_segments:
            if i>len(image_nns):
                break
        
            desc_mask = desc_mask | train_index['image']==image_nns[i]
            m = desc_mask==1
            i = i+1
        
        
            
            
                                             
        ret_set_index = {}
        for key in train_index.keys():
            ret_set_index[key] = train_index[key][desc_mask]
                 
        return (ret_set_index, desc_mask)
    
    def calculate_rs(self, path, desc_path, train_file_list, train_index, desc_funs):    
        '''
        '''
        
        if os.path.isfile(path + "/SPsearchRs_k200.mat"):
            self.search_rs = read_dict_from_matfile(path + "/SPsearchRs_k200.mat")
            return
        else:
            self.search_rs = {}
            
        search_rs = {}
        
        num_nns = np.array(range(20,201,20)) #[20,40...200]
        rand_ind = np.array([])
        num_rand = 1000
        dists = np.zeros((num_rand, len(num_nns)))
        
        cont_count = 0
        for feature in desc_funs:
            if search_rs[desc_funs]:
                cont_count = cont_count + 1
                continue
            #if len(ret_)
            im_ind = np.random.permutation(len(self.train_file_list))
            ret_set_ind = self.prune_index(train_index, im_ind, 
                                           min(len(im_ind),2000), 0)
            ret_set_sp_desc = self.load_segment_desc(self.train_file_indices, 
                                                     ret_set_ind, desc_path, desc_funs)

            data = ret_set_sp_desc[feature]
            num_rand = min(num_rand, data.shape[0])
            if len(rand_ind) == 0:
                rand_ind = np.random.permutation(data.shape[0])
                rand_ind = rand_ind[range(num_rand)]
            
            
                
            for j in range(num_rand):
                query = data[rand_ind[j],:]
                dist = np.sqrt(cdist(query, data, 'euclidean'))
                dist = np.sort(dist)
                for k in range(len(num_nns)):
                    dist[j,k] = dist[num_nns[k]]
            
            
            rs = np.median(dists)
            search_rs[feature]['Rs'] = rs
            search_rs[feature]['numNNs'] = num_nns
            
            self.search_rs = search_rs
            
        if cont_count<len(desc_funs):
            write_dict_to_matfile(search_rs, path + "/SPsearchRs_k200.mat")


    def load_segment_desc(self, file_list, index, path, desc_funs):
        '''
        returning seg_descs, superPixels, unLabeledSPDesc
        '''
        
        
        if len(index)>0:
            ran = np.unique(index['image'])
        
        
        else:
            ran = np.array(range(len(file_list)))
                        
        
        seg_descs = {}
        un_labeled_sp_desc = {}
        
        for feature in desc_funs:
            seg_descs[feature] = np.array([])
            un_labeled_sp_desc[feature] = np.array([])
        
                    
        current = 0
        current_un = 1
        for i in ran.flatten(1):
            
            if len(index)>0:
                num_sp = len(index['image'][index['image']==i])
                sp = self.sp_array_list[i]
                u_sp = np.unique(sp)
                sp_map = np.zeros(max(u_sp))
                sp_map[u_sp] = range(len(u_sp))
            
            all_desc = {}
            
            if os.path.isfile(path + "/AllDesc/" + self.image_list[i]+'.mat'):
                all_desc = read_dict_from_matfile(path + 
                                    "/AllDesc/" + self.image_list[i]+'.mat')
                
            covered = np.zeros(desc_funs.shape)==1
            for k in len(desc_funs):
                
                feature = desc_funs[k]


                if not all_desc[feature]:
                    all_desc[feature] = self.segment_descriptor_dict[feature][i]
                else:
                    desc = all_desc[feature]
                    covered[k] = True
                
                if len(index)>0:
                    if len(seg_descs[feature])==0:
                        seg_descs[feature] = np.zeros((len(index['image']),
                                                          desc.shape[0]))
                    
                    seg_descs[feature][current:current+num_sp,:] = \
                            desc[:, sp_map[index['sp'][index['image']==i]]].T
                    
                    un_mask = np.ones((desc.shape[1], 1)) == 1
                    un_mask[index['sp'][index['image']==i]] = False
                    un_labeled_sp_desc[feature][current_un:current_un+
                                                len(un_mask[un_mask]),:] = \
                                                desc[:, un_mask].T
                else:
                    seg_descs[feature] = np.concatenate(seg_descs[feature], 
                                                       desc.T, axis = 1)
                
            if (covered == False).any():
                write_dict_to_matfile(all_desc, path + "/AllDesc/" + 
                                      self.image_list[i]+'.mat')
            
            if len(index)>0:
                current = current + num_sp
                current_un = current_un + len(un_mask[un_mask])
            
        return seg_descs
# if(nargout>1) #### Haaaaeeeeeh??? letzte baseFName? oder was?
#     outSPName = fullfile(HOMEDATA,'super_pixels',sprintf('%s.mat',baseFName));
#     load(outSPName);
# end
        
             
    def multi_level_seg_mrf(self, home_data, home_label_sets, test_name, base_name, labels, im_sp, adj_pairs, data_cost, smoothing_matrix, label_subsets, label_sub_sets_weight):     
        '''
        '''
        
# function [LabelPixels LabelSPs] = MultiLevelSegMRF(HOMEDATA,HOMELABELSETS,testName,baseFName,Labels,imSP,adjPairs,dataCost,smoothingMatrix,labelSubSets,labelSubSetsWeight)
# 
# if(~exist('canskip','var'))
#     canskip = 1;
# end

        canskip = 1
        
         
        num_label_sets = len(data_cost)
        label_pixels = {}#cell(numLabelSets,1);
        label_sps = {}#cell(numLabelSets,1);
        good_files = 0;
        out_file_names = np.array(['' for i in range(num_label_sets)])#cell(numLabelSets,1);
        for i in range(num_label_sets):
            #[foo LabelFolder] = fileparts(HOMELABELSETS{i});
            #if(~isempty(baseFName))
#            outFileNames[i] = fullfile(HOMEDATA,LabelFolder,testName,sprintf('%s.mat',baseFName));
#                 if(exist(outFileNames{i},'file')&&canskip)
#                     load(outFileNames{i});
#                     LabelPixels{i} = L;
#                     LabelSPs{i} = Lsp;
#                     if(exist('labelList','var'))
#                         if(size(dataCost{i},2)==length(labelList))
#                             goodFiles = goodFiles + 1;
#                         end
#                     end
#                 end
#             end
#         end
#         if(goodFiles == numLabelSets)
#             return;
#         end
#          
#         %reduce the label set to only plausable labels
#         usedLs = cell(size(dataCost));
#         allUsedLs = [];
#         current = 0;
#         for ls = 1:numLabelSets
#             [foo Lmin] = min(dataCost{ls},[],2);
#             usedLs{ls} = unique([Lmin; 1; size(dataCost{ls},2)]);
#             allUsedLs = [allUsedLs; current+usedLs{ls}(:)];
#             current = current + size(dataCost{ls},2);
#             dataCost{ls} = dataCost{ls}(:,usedLs{ls});
#         end
#         coDataCost = dataCost;
#         smoothingMatrix = smoothingMatrix(allUsedLs,:);
#         smoothingMatrix = smoothingMatrix(:,allUsedLs);
#         numSP = size(dataCost{1},1);
#          
#         if(exist('labelSubSets','var') && labelSubSetsWeight>0)
#             currentStartLabel = 0;
#             flatlabelSubSets = cell(0);
#             for ls = 1:numLabelSets
#                 for i = 1:length(labelSubSets{ls})
#                     [foo subSetLocalNdx] = intersect(usedLs{ls},labelSubSets{ls}{i});
#                     if(~isempty(subSetLocalNdx))
#                         flatlabelSubSets{end+1} = int32(subSetLocalNdx+currentStartLabel);
#                     end
#                 end
#                 currentStartLabel = currentStartLabel + length(usedLs{ls});
#             end
#         end
#          
#         %create the adjacency graph between all nodes
#         sparseSmooth = sparse(adjPairs(:,1),adjPairs(:,2),ones(size(adjPairs,1),1),numSP*numLabelSets,numSP*numLabelSets);
#         for i = 1:numLabelSets-1
#             sparseSmooth = sparseSmooth + sparse(adjPairs(:,1)+numSP*i,adjPairs(:,2)+numSP*i,ones(size(adjPairs,1),1),numSP*numLabelSets,numSP*numLabelSets);
#         end
#         s = ones(numSP,1);
#         for i = 1:numLabelSets
#             for j = i+1:numLabelSets
#                 a = (i-1)*numSP+1:i*numSP;
#                 b = (j-1)*numSP+1:j*numSP;
#                 sparseSmooth = sparseSmooth + sparse(a,b,s,numSP*numLabelSets,numSP*numLabelSets);
#                 sparseSmooth = sparseSmooth + sparse(b,a,s,numSP*numLabelSets,numSP*numLabelSets);
#             end
#         end
#         sparseSmooth(sparseSmooth>0) = 1;
#         %compute scaling to correct for int rounding errors
#         minTarget = 100;
#         minDataCost = minTarget;
#         for ls = 1:numLabelSets
#             minDataCost = min(minDataCost,min(dataCost{ls}(:)));
#         end
#         multiplier = max(1,minTarget/minDataCost);
#         dataInt = cell(size(dataCost));
#         numInt = 0;
#         for ls = 1:numLabelSets
#             if(isinteger(dataCost{ls}))
#                 dataInt{ls} = int32(dataCost{ls}');
#                 numInt = numInt + 1;
#             else
#                 dataInt{ls} = int32(ceil(dataCost{ls}'*multiplier));
#             end
#         end
#         if(numInt==numLabelSets)
#             multiplier = 1;
#         end
#          
#         %find initial labeling
#         Lin = zeros(numSP*numLabelSets,1);
#         current = 0;
#         for ls = 1:numLabelSets
#             [foo Lin((ls-1)*numSP+1:numSP*ls)] = min(dataInt{ls},[],1);
#             Lin((ls-1)*numSP+1:numSP*ls) = Lin((ls-1)*numSP+1:numSP*ls)+current;
#             current = current + size(dataInt{ls},1);
#         end
#         numTotalLabs = current;
#         scInt = int32(ceil(smoothingMatrix*multiplier));
#         if(exist('flatlabelSubSets','var'))
#             labelSubSetsWeight = int32(labelSubSetsWeight*multiplier);
#         end
#         numItt = 5;
#         Lout = cell(numItt,1);
#         Energy = zeros(size(Lout));
#         dataIntTmp = dataInt;
#         for i=1:numItt
#             graph = GCO_Create(numLabelSets*numSP,length(scInt));
#             GCO_SetVerbosity(graph,0)
#             current = 0;
#             for ls = 1:numLabelSets
#                 numL = size(dataInt{ls},1);
#                 dataIntTmp{ls} = (dataInt{ls} + int32(coDataCost{ls}'))./2;
#                 for l = 1:numL
#                     GCO_SetDataCost(graph,int32([(ls-1)*numSP+1:numSP*(ls); dataIntTmp{ls}(l,1:numSP)]),l+current);
#                 end
#                 current = current+numL;
#             end
#             if(exist('flatlabelSubSets','var'))
#                 for j = 1:length(flatlabelSubSets)
#                     GCO_SetLabelCost(graph,labelSubSetsWeight,flatlabelSubSets{j})
#                 end
#             end
#             GCO_SetSmoothCost(graph,scInt);
#             stemp = ceil(sparseSmooth);
#             GCO_SetNeighbors(graph,stemp);
#             GCO_SetLabeling(graph,Lin);
#             GCO_SetLabelOrder(graph,1:numTotalLabs);%randperm(numTotalLabs))
#              
#             if(exist('flatlabelSubSets','var'))
#                 GCO_Expansion(graph);
#             else
#                 %GCO_Expansion(graph)
#                 GCO_Swap(graph);
#             end
#             Lout{i} = GCO_GetLabeling(graph);
#             Energy(i) = GCO_ComputeEnergy(graph);
#             current = 0;
#             GCO_Delete(graph);
#         end
#             
#         if(exist('colorModel','var')&& any(colorModel>0))
#             Energy = Energy(numItt);
#             Lout = Lout{numItt};
#         else
#             [foo ndx] = min(Energy);
#             Energy = Energy(ndx);
#             Lout = Lout{ndx};
#         end
#          
#         if(sum(scInt(:))==0 && (~exist('flatlabelSubSets','var') || labelSubSetsWeight==0) && (~exist('colorModel','var') || all(colorModel==0)) )
#             energy = 0;
#             for ls = 1:numLabelSets
#                 [foo] = min(dataInt{ls});
#                 energy = energy + sum(foo);
#             end
#             if(sum(Lout~=Lin)>0)
#                 current = 0;
#                 inEnergy = 0;
#                 outEnergy = 0;
#                 for ls = 1:numLabelSets
#                     inEnergy = inEnergy+sum(dataInt{ls}(sub2ind(size(dataInt{ls}),min(size(dataInt{ls},1),max(1,double(Lin((ls-1)*numSP+1:numSP*ls)')-current)),1:size(dataInt{ls},2))));
#                     outEnergy = outEnergy+sum(dataInt{ls}(sub2ind(size(dataInt{ls}),min(size(dataInt{ls},1),max(1,double(Lout((ls-1)*numSP+1:numSP*ls)')-current)),1:size(dataInt{ls},2))));
#                     current = current + size(dataInt{ls},1);
#                 end
#                 fprintf('!!!!!!!!!!!!!!ERROR Graph Cut Broken %d: %.1f vs. %.1f!!!!!!!!!!!!!!!!!\n',sum(Lout~=Lin), inEnergy, outEnergy);
#                 %Lout=Lin;
#                 %keyboard;
#             end
#         end
#          
#         current = 0;
#         for ls = 1:numLabelSets
#             L = Lout(1+(ls-1)*numSP:ls*numSP);
#             L = L-current;
#             %correct for glitches in the graph cut code
#             labelsFound = unique(L);
#             inds = find(labelsFound<1|labelsFound>length(usedLs{ls}));
#             if(~isempty(inds))
#                 for lfound = labelsFound(inds(:)')'
#                     L(L==lfound) = 1;
#                     fprintf('Label from other labels set used!!!\n');
#                 end
#             end
#              
#             L = usedLs{ls}(L);
#             Lsp = L;
#             if(~isempty(imSP))
#                 L = L(imSP);
#                 if(~isempty(baseFName))
#                     labelList = Labels{ls};
#                     make_dir(outFileNames{ls});
#                     save(outFileNames{ls},'L','labelList','Lsp');
#                 end
#                 LabelPixels{ls} = L;
#             end
#             LabelSPs{ls} = Lsp;
#             current = current+length(usedLs{ls});
#         end
#          
#         end
                                       
                
    