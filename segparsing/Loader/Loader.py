'''
Created on 25 Apr 2014

@author: johann strassburg
Loader.py: use this class to load images and calculate superpixels
'''

import numpy as np
import Image
from utils.utils import path_to_pathlist, read_arr_from_file, write_arr_to_file,\
    read_arr_from_matfile, write_arr_to_matfile, path_to_subfolder_pathlist,\
    read_dict_from_matfile, Logger, get_border_bounding_box,\
    write_dict_to_matfile, read_arr_from_txt
import pylab
import os.path
from os.path import basename
from SegmentDescriptors.segmentFeatureExtractor import segmentFeatureExtractor
import threading
from SuperPixels.SuperPixels import SPCalculator


class DataLoader(object):
    '''
    classdocs
    '''

    def __init__(self, save_path, save_mode = True, seq_reading = True):
        '''
        @param save_path:   Path to save data locally for faster 
                            reading/processing in further examination
                            
        @param save_mode:   option to save data locally. 
                            True can results in big data files on hdd
        
        @param seq_reading: Option to set sequential reading on/off. 
                            Sequential reading is recommended for huge data
                            Sequential reading on False stores data temporally into ram.
        Constructor
        '''
        
        self.log = Logger(verbose=False)
        self.log_lock = threading.Lock()
        self.save_mode = save_mode
        
        #setting paths
        self.save_path = save_path
        self.image_array_list_path = self.save_path + "image_array_list.mat"
        
        #sequential Reading
        self.seq_reading = seq_reading
        
        #image variables
        self.image_array_list = np.array([])
        self.image_path_list = np.array([])
        self.image_list = np.array([]) #list of image names
        
        #global descriptor variables
        self.global_descriptor_dict_path = self.save_path + \
                                            "global_descriptor_dict.mat"
        #self.global_descriptor_path_list = np.array([])
        self.global_descriptor_dict = {}
        
        self.centers_path = self.save_path + "centers.mat"
        self.centers = {}
    
        #superpixels variables
        self.sp_array_list_path = self.save_path + "sp_array_list.mat"
        self.sp_array_list = np.array([])
        self.sp_array_path_list = np.array([])
        
        
        #segment descriptors variables
        self.feature_list = ['sift_hist_int_', 'sift_hist_left', 
                             'sift_hist_right', 'sift_hist_top', 'top_height',
                             'absolute_mask', 'bb_extent',
                             'centered_mask_sp', 'color_hist',
                             'color_std', 'color_thumb', 'color_thumb_mask',
                             'dial_color_hist', 'dial_text_hist_mr',
                             'gist_int', 'int_text_hist_mr', 
                             'mean_color', 'pixel_area',
                             'sift_hist_bottom', 'sift_hist_dial']
        self.segment_descriptor_dict_path = self.save_path + \
                                            "segment_descriptor_dict.mat"
        self.segment_descriptor_dict = {}
        
        self.sfe = segmentFeatureExtractor()
        
        
        
        #image label variables
        self.image_label_array_list = np.array([])
        self.image_label_array_list_path = self.save_path + \
                                            "image_label_array_list.mat"
        self.semantic_labels = np.array([])
        self.segment_label_index = {}
        
        
        #Thread variables
        self.label_thread_index = {}
        self.label_thread_lock = threading.Lock()
        self.max_thread = threading.Semaphore(8)
        self.thread_lock = threading.Lock()
        
        self.feat_calc_thread_lock = threading.Lock()
        
        
        
        #evaluation variables
        self.test_file_names = np.array([])
        self.test_file_indices = np.array([])
        self.train_file_indices = np.array([])
        
        self.labels = {}
        self.labels[0] = np.array(['sky', 'horizontal', 'vertical'])
        self.labels[1] = np.array(['awning','balcony', 'bird', 'boat', 'bridge',
                                   'building', 'car', 'cow', 'crosswalk', 
                                   'desert', 'door', 'fence', 
                                   'field', 'grass', 'moon', 
                                   'mountain', 'person', 'plant', 
                                   'pole', 'river', 'road', 'rock', 
                                   'sand', 'sea', 'sidewalk', 'sign', 
                                   'sky', 'staircase', 'streetlight', 
                                   'sun', 'tree', 'window'])
        
        #adjacency
        self.adj_file_list = np.array([])
        
        
        
    
    def load_images(self, path):
        '''
        loads images from given path into an array list
        '''
        self.log.start("Load Images", 1, 1)
        path_list = path_to_subfolder_pathlist(path, filter=".jpg")
        self.image_path_list = path_list
        
        self.image_list = np.array([os.path.splitext(basename(filepath))[0] 
                                    for filepath in path_list])
        
        if not self.seq_reading:
            if os.path.isfile(self.image_array_list_path):
                self.image_array_list = read_arr_from_matfile(
                                self.image_array_list_path, 
                                os.path.splitext(basename(
                                            self.image_array_list_path))[0])
            
            else:
                self.image_array_list = np.array([np.array(Image.open(filepath))
                                              for filepath in path_list])
                write_arr_to_matfile(self.image_array_list, 
                                     self.image_array_list_path, 
                                     os.path.splitext(basename(
                                            self.image_array_list_path))[0])
        
        self.log.update()
        
    
    
    def load_global_descriptors(self, path, subfolder):#Deprecated
        '''
        loads global descriptors from mat files in given path
        '''
                        
        self.log.start('Load Global Descriptors', 7, update_num=1)
        if os.path.isfile(self.global_descriptor_dict_path):
            self.global_descriptor_dict =   read_dict_from_matfile(
                                        self.global_descriptor_dict_path)

            self.log.end()
        else:
            self.global_descriptor_dict = {}
            
            

            print "Loading Color Histogramm"
            co_hist = np.array([read_arr_from_file(filepath,
                                    'coHist')
                                    for filepath in 
                                    path_to_subfolder_pathlist(
                                            path + '/' + 'coHist', 
                                            filter = ".mat")])
            self.log.update("Loading colorGist")
            color_gist = np.array([read_arr_from_file(filepath, 'colorGist')
                                 for filepath in 
                                 path_to_subfolder_pathlist(
                                            path + '/' + 'colorGist', 
                                            filter = ".txt")]) 
            self.log.update("Loading Spatial Pyramid")
            spatial_pyramid_h = np.array([read_arr_from_matfile(filepath,'H')
                                 for filepath in 
                                 path_to_subfolder_pathlist(
                                            path + '/' + 'SpatialPyrDenseScaled', 
                                            filter = "hist_200.mat")])
            self.log.update()
            spatial_pyramid_pyr = np.array([read_arr_from_matfile(filepath,
                                                                  'pyramid')
                                 for filepath in 
                                 path_to_subfolder_pathlist(
                                            path + '/' + 'SpatialPyrDenseScaled', 
                                            filter = "pyramid_200.mat")])
            self.log.update()
            spatial_pyramid_texton = np.array([read_dict_from_matfile(filepath)
                                 for filepath in 
                                 path_to_subfolder_pathlist(
                                            path + '/' + 'SpatialPyrDenseScaled', 
                                            filter = "ind_200.mat")])
            
            self.log.update("Loading mr filter")
            mr_filter = np.array([read_arr_from_file(filepath,
                                    'texton')
                                    for filepath in 
                                    path_to_subfolder_pathlist(
                                            path + '/' + 'Textons/mr_filter', 
                                            filter = ".mat")])
            self.log.update("Loading Sift Textons")
            sift_textons = np.array([read_arr_from_file(filepath,
                                    'texton')
                                    for filepath in 
                                    path_to_subfolder_pathlist(
                                            path + '/' + 'Textons/sift_textons', 
                                            filter = ".mat")])
            
            
            
            self.global_descriptor_dict['coHist'] = co_hist
            self.global_descriptor_dict['colorGist'] = color_gist
            self.global_descriptor_dict['H'] = spatial_pyramid_h
            self.global_descriptor_dict['pyramid'] = spatial_pyramid_pyr
            self.global_descriptor_dict['texton_ind'] = spatial_pyramid_texton
            
            self.global_descriptor_dict['mr_filter'] = mr_filter
            self.global_descriptor_dict['sift_textons'] = sift_textons
            
            self.log.update()
            
            self.log.start("Writing Global Descriptor File to HDD", 1, 1)

            if self.save_mode:
                write_dict_to_matfile(self.global_descriptor_dict,
                                 self.global_descriptor_dict_path)
        
            self.log.update()
            
    def load_centers(self, path):#Deprecated
        '''
        loads centers from mat files in given path
        '''
        
        
        self.log.start('Load Centers', 3, update_num=1)
        if os.path.isfile(self.centers_path):
            self.centers = read_dict_from_matfile(
                                        self.centers_path)

            self.log.end()
        else:
            self.centers = {}
            
            sift_centers = read_arr_from_matfile(path+
                                            "/Dictionaries/sift_centers.mat", 
                                            'dic')
            self.centers['sift_centers'] = sift_centers
            
            self.log.update()
            mr_resp_centers = read_arr_from_matfile(path+
                                            "/Dictionaries/mr_resp_centers.mat", 
                                            'dic')
            self.centers['mr_resp_centers'] = mr_resp_centers
            
            self.log.update()
            gist_centers = read_arr_from_matfile(path+
                                            "/Dictionaries/gist_centers.mat", 
                                            'dic')
            self.centers['gist_centers'] = gist_centers
            
            self.log.update()
            
            self.log.start("Writing Centers File to HDD", 1, 1)
            if self.save_mode:
                write_dict_to_matfile(self.centers,
                                 self.centers_path)
        
            self.log.update()
            
    
    def load_super_pixel(self, path, seg_method):
        '''
        loads super pixels from given path or calculates them if not created yet
        @param path: Path to superpixels
        @param seg_method: segmantation method and parameters to calculate superpixels
        '''
        
        #calculate superpixels which are not created
        sp_calculator = SPCalculator(self)
        sp_calculator.calculate_sp(seg_method, path)
    
        #load superpixels if seq_reading is set to False
        self.log.start("Load SuperPixels", 1, 1)
        path_list = path_to_subfolder_pathlist(path, filter=".mat")
        
        self.sp_array_path_list = path_list
        if not self.seq_reading:
            if os.path.isfile(self.sp_array_list_path):
                self.sp_array_list = read_arr_from_matfile(
                                self.sp_array_list_path, 
                                os.path.splitext(basename(
                                            self.sp_array_list_path))[0])
            
            else:
                for i in range(len(path_list)): 
                    try:
                        sp = read_arr_from_matfile(path_list[i],'superPixels')
                    except:
                        print "Failure in File {0}".format(i)
                        print self.image_list[i]
                        raise
                
                self.sp_array_list = np.array([read_arr_from_matfile(filepath,'superPixels')
                                 for filepath in path_list])
                        
                if self.save_mode:
                    write_arr_to_matfile(self.sp_array_list, 
                                     self.sp_array_list_path, 
                                     os.path.splitext(basename(
                                            self.sp_array_list_path))[0])
        
        self.log.update()
        
    
    def load_segment_descriptors(self, path, seg_suffix):#Deprecated
        '''
        loads Segment Descriptors
        '''
        
        
        self.segment_descriptor_dict_path = self.save_path + \
            "segment_descriptor_dict_"+seg_suffix + ".mat"
        if os.path.isfile(self.segment_descriptor_dict_path):
            self.log.start("Load Segment Descriptors", 1, 1)
            self.segment_descriptor_dict = read_dict_from_matfile(
                                            self.segment_descriptor_dict_path)
            self.log.update()
        
        else:
            self.segment_descriptor_dict = {}
    
            self.calculate_segment_features(path)
            
            self.log.start("Load Segment Descriptors", len(self.feature_list), 
                           1)
            
            for feature in self.feature_list:
                image_features = np.array([read_arr_from_matfile(filepath,
                                                                  'desc')
                                 for filepath in 
                                 path_to_subfolder_pathlist(
                                            path + '/' + feature, 
                                            filter = ".mat")])
                self.segment_descriptor_dict[feature] = image_features
                self.log.update()
            
            self.log.start("Writing Segment Descriptor File to HDD", 1, 1)
            if self.save_mode:
                write_dict_to_matfile(self.segment_descriptor_dict,
                                 self.segment_descriptor_dict_path)
            self.log.update()
        
    
    def calculate_segment_features(self, path):#Deprecated
        '''
        calculates the segment feature segments
        '''
    
        print path
        self.log.start("Calculate Segment Features", len(self.image_list), 1)
        
        for feature in self.feature_list:
                    if not os.path.exists(path+"/"+feature+
                                         "/"):
                        os.makedirs(path+"/"+feature+
                                         "/")

        for im_num in range(len(self.image_list)):
            
            if not os.path.isfile(path+"/"+
                                 self.image_list[im_num]):
                
                if not os.path.isfile(path+"/busy_"+
                                     self.image_list[im_num]):
                    
                    write_arr_to_file(np.array([1]),path+"/busy_"+
                                 self.image_list[im_num])
                    
                    sp_array = self.sp_array_list[im_num]
                    sp_ind = np.unique(sp_array)
                    image = self.image_array_list[im_num]
                    descs = {}
                    for desc in self.feature_list:
                        descs[desc] = np.array([[]])
                      
                    log = Logger()

                    for sp_num in sp_ind:
                        mask = sp_array == sp_num
                        (borders, bb) = get_border_bounding_box(mask)
                        borders = borders[bb[0]:bb[1]+1,bb[2]:bb[3]+1,:]
                        mask_crop = mask[bb[0]:bb[1]+1,bb[2]:bb[3]+1]
                        im_crop = image[bb[0]:bb[1]+1,bb[2]:bb[3]+1,:];
                        textons_crop = {}
                        textons_crop['mr_filter'] = \
                            self.global_descriptor_dict['mr_filter'][im_num][bb[0]:bb[1]+1,
                                                                        bb[2]:bb[3]+1]
                              
                        textons_crop['sift_textons'] = \
                            self.global_descriptor_dict['sift_textons'][im_num][bb[0]:bb[1]+1,
                                                                        bb[2]:bb[3]+1]
                          
                        centers = self.centers
          
                          
                        #Calculating Descriptors
                          
                        #log.start("sift_hist_int",1,1)
                        sift_hist_int_ = self.sfe.sift_hist_int(mask_crop, centers, 
                                                           textons_crop).flatten(1)
                        sift_hist_int_ = sift_hist_int_.\
                            reshape((sift_hist_int_.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("sift_hist_left",1,1)
                        sift_hist_left = self.sfe.sift_hist_left(centers, textons_crop, 
                                                             borders).flatten(1)
                        sift_hist_left = sift_hist_left.\
                            reshape((sift_hist_left.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("sift_hist_right",1,1)    
                        sift_hist_right = self.sfe.sift_hist_right(centers, 
                                                                   textons_crop, 
                                                               borders).flatten(1)                                     
                        sift_hist_right = sift_hist_right.\
                            reshape((sift_hist_right.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("sift_hist_top",1,1)    
                        sift_hist_top = self.sfe.sift_hist_top(centers, textons_crop, 
                                                           borders).flatten(1)
                        sift_hist_top = sift_hist_top.\
                            reshape((sift_hist_top.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("top_height",1,1)    
                        top_height = self.sfe.top_height(mask, mask_crop, bb).flatten(1)
                        top_height = top_height.\
                            reshape((top_height.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("absolute_mask",1,1)    
                        absolute_mask = self.sfe.absolute_mask(im_crop, mask).flatten(1)
                        absolute_mask = absolute_mask.\
                            reshape((absolute_mask.shape[0],1)).astype(np.float32).astype(np.float32)
                        #log.update()    
                        #log.start("bb_extent",1,1)    
                        bb_extent = self.sfe.bb_extent(mask, mask_crop, bb).flatten(1)
                        bb_extent = bb_extent.\
                            reshape((bb_extent.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("color_hist",1,1)    
                        color_hist = self.sfe.color_hist(im_crop, mask_crop).flatten(1)
                        color_hist = color_hist.\
                            reshape((color_hist.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("centered_mask_sp",1,1)   
                        centered_mask_sp = self.sfe.centered_mask_sp(im_crop, mask, 
                                                                 mask_crop).flatten(1)
                        centered_mask_sp = centered_mask_sp.\
                            reshape((centered_mask_sp.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("color_std",1,1)    
                        color_std = self.sfe.color_std(im_crop, mask_crop).flatten(1)
                        color_std = color_std.\
                            reshape((color_std.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("color_thumb",1,1)    
                        color_thumb = self.sfe.color_thumb(im_crop, 
                                                           mask_crop).flatten(1)
                        color_thumb = color_thumb.\
                            reshape((color_thumb.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("color_thumb_mask",1,1)    
                        color_thumb_mask = self.sfe.color_thumb_mask(im_crop, 
                                                                mask_crop).flatten(1)
                        color_thumb_mask = color_thumb_mask.\
                            reshape((color_thumb_mask.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("dial_color_hist",1,1)    
                        dial_color_hist = self.sfe.dial_color_hist(im_crop, 
                                                                   borders).flatten(1)
                        dial_color_hist = dial_color_hist.\
                            reshape((dial_color_hist.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("dial_text_hist_mr",1,1)    
                        dial_text_hist_mr = self.sfe.dial_text_hist_mr(centers, 
                                                                   textons_crop, 
                                                                   borders).flatten(1)
                        dial_text_hist_mr = dial_text_hist_mr.\
                            reshape((dial_text_hist_mr.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("gist_int",1,1)
                        # import profile
                        #profile.run('self.sfe.gist_int(im_crop, mask, centers, image).flatten(1)')                                               
                        gist_int = self.sfe.gist_int(im_crop, mask, centers, 
                                                     image).flatten(1)                                           
                        gist_int = gist_int.\
                            reshape((gist_int.shape[0],1)).astype(np.float32)                                          
                        #log.update()    
                        #log.start("int_text_hist_mr",1,1)                                           
                        int_text_hist_mr = self.sfe.int_text_hist_mr(mask_crop, centers, 
                                                                textons_crop).flatten(1)                                           
                        int_text_hist_mr = int_text_hist_mr.\
                            reshape((int_text_hist_mr.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("mean_color",1,1)                                               
                        mean_color = self.sfe.mean_color(im_crop, mask_crop).flatten(1)
                        mean_color = mean_color.\
                            reshape((mean_color.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("pixel_area",1,1)    
                        pixel_area = self.sfe.pixel_area(mask, mask_crop).flatten(1)
                        pixel_area = pixel_area.\
                            reshape((pixel_area.shape[0],1)).astype(np.float32)
                        #log.update()    
                        #log.start("sift_hist_bottom",1,1)    
                        sift_hist_bottom = self.sfe.sift_hist_bottom(centers, 
                                                                     textons_crop, 
                                                                 borders).flatten(1)
                        sift_hist_bottom = sift_hist_bottom.\
                            reshape((sift_hist_bottom.shape[0],1)).astype(np.float32)
                          
                        #log.update()    
                        #log.start("sift_hist_dial",1,1)
                        sift_hist_dial = self.sfe.sift_hist_dial(centers, textons_crop, 
                                                             borders).flatten(1)
                        sift_hist_dial = sift_hist_dial.\
                            reshape((sift_hist_dial.shape[0],1)).astype(np.float32)
                        #log.update()    
                          
                          
                        #Writing Descriptors into dictionary
                        if sp_num != sp_ind[0]:
                            descs['sift_hist_int_'] = \
                                np.concatenate((descs['sift_hist_int_'],
                                                sift_hist_int_),axis=1)
                              
                            descs['sift_hist_left'] = \
                                np.concatenate((descs['sift_hist_left'], 
                                                sift_hist_left),axis=1)
          
                            descs['sift_hist_right'] = \
                                np.concatenate((descs['sift_hist_right'], 
                                                sift_hist_right),axis=1)
                      
                            descs['sift_hist_top'] = \
                                np.concatenate((descs['sift_hist_top'],sift_hist_top),
                                               axis=1)                              
          
                            descs['top_height'] = \
                                np.concatenate((descs['top_height'],top_height),axis=1)              
          
                            descs['absolute_mask'] = \
                                np.concatenate((descs['absolute_mask'],absolute_mask),
                                               axis=1)
                          
                            descs['bb_extent'] = \
                                np.concatenate((descs['bb_extent'],bb_extent),axis=1)              
          
                            descs['centered_mask_sp'] = \
                                np.concatenate((descs['centered_mask_sp'],
                                                centered_mask_sp),axis=1)
          
                            descs['color_hist'] = \
                                np.concatenate((descs['color_hist'],color_hist),axis=1)
          
                            descs['color_std'] = \
                                np.concatenate((descs['color_std'],color_std),axis=1)
                          
                            descs['color_thumb'] = \
                                np.concatenate((descs['color_thumb'],color_thumb),
                                               axis=1)
                          
                            descs['color_thumb_mask'] = \
                                np.concatenate((descs['color_thumb_mask'],
                                                color_thumb_mask),axis=1)
                                                                   
                            descs['dial_color_hist'] = \
                                np.concatenate((descs['dial_color_hist'],
                                                dial_color_hist),axis=1)
                          
                            descs['dial_text_hist_mr'] = \
                                np.concatenate((descs['dial_text_hist_mr'],
                                                dial_text_hist_mr),axis=1)
                          
                            descs['gist_int'] = \
                                np.concatenate((descs['gist_int'],gist_int),axis=1)
                          
                            descs['int_text_hist_mr'] = \
                                np.concatenate((descs['int_text_hist_mr'],
                                                int_text_hist_mr),axis=1)
                          
                            descs['mean_color'] = \
                                np.concatenate((descs['mean_color'],mean_color),axis=1)
                          
                            descs['pixel_area'] = \
                                np.concatenate((descs['pixel_area'],pixel_area),axis=1)
                          
                            descs['sift_hist_bottom'] = \
                                np.concatenate((descs['sift_hist_bottom'],
                                                sift_hist_bottom),axis=1)
                          
                            descs['sift_hist_dial'] = \
                                np.concatenate((descs['sift_hist_dial'],sift_hist_dial),
                                               axis=1)
                          
                        #create a new array in dictionary if it is empty    
                        else:
                            descs['sift_hist_int_'] = sift_hist_int_
                              
                            descs['sift_hist_left'] =  sift_hist_left
          
                            descs['sift_hist_right'] =  sift_hist_right
                      
                            descs['sift_hist_top'] = sift_hist_top                              
          
                            descs['top_height'] = top_height              
          
                            descs['absolute_mask'] = absolute_mask
                          
                            descs['bb_extent'] = bb_extent              
          
                            descs['centered_mask_sp'] = centered_mask_sp
          
                            descs['color_hist'] = color_hist
          
                            descs['color_std'] = color_std
                          
                            descs['color_thumb'] = color_thumb
                          
                            descs['color_thumb_mask'] = color_thumb_mask
                                                                   
                            descs['dial_color_hist'] = dial_color_hist
                          
                            descs['dial_text_hist_mr'] = dial_text_hist_mr
                          
                            descs['gist_int'] = gist_int
                          
                            descs['int_text_hist_mr'] = int_text_hist_mr
                          
                            descs['mean_color'] = mean_color
                          
                            descs['pixel_area'] = pixel_area
                          
                            descs['sift_hist_bottom'] = sift_hist_bottom
                          
                            descs['sift_hist_dial'] = sift_hist_dial
                  
                    
                    
                    
                    if not os.path.isfile(path+"/"+
                                 self.image_list[im_num]):
                        #store descriptor into files
                          
                        for feature in self.feature_list:

                            write_arr_to_matfile(descs[feature], path+"/"+feature+
                                                 "/"+
                                                 self.image_list[im_num]
                                                 +".mat", "desc")
                          
                        #if matrices are written, write a file for proove
                        write_arr_to_file(np.array([1]),path+"/"+
                                             self.image_list[im_num])
                        
                    os.remove(path+"/busy_"+
                                 self.image_list[im_num])
              
            self.log.update()

    def label_images(self, label_path, sp_label_path):#Deprecated
        '''
        label superpixels according to their image labels
        '''
        
        self.log.start("Load Image Labels", 1, 1)
        path_list = path_to_subfolder_pathlist(label_path, filter=".mat")
        
        
        if os.path.isfile(self.image_label_array_list_path):
            self.image_label_array_list = read_arr_from_matfile(
                            self.image_label_array_list_path, 
                            os.path.splitext(basename(
                                        self.image_label_array_list_path))[0])
        
        else:
            self.image_label_array_list = np.array([read_arr_from_matfile(filepath,
                                                                 'S')
                                 for filepath in path_list])
            
            write_arr_to_matfile(self.image_label_array_list, 
                                 self.image_label_array_list_path, 
                                 os.path.splitext(basename(
                                        self.image_label_array_list_path))[0])
        
        self.log.update()
        
        self.log.start("Labeling SuperPixels", len(self.image_array_list), 1)
        
        if os.path.isfile(sp_label_path + "/segIndex.mat"):
            self.segment_label_index = read_dict_from_matfile(sp_label_path +
                                                              "/segIndex.mat")
            self.log.end()
        else:
            self.segment_label_index = {} 
            self.segment_label_index['label'] = np.array([])
            self.segment_label_index['sp'] = np.array([])
            self.segment_label_index['image'] = np.array([])
            self.segment_label_index['spSize'] = np.array([])
            
            self.label_thread_lock.acquire()
            for i in range(len(self.image_label_array_list)):
                
                #print "Active Threads: {0}".format(8-self.max_thread)
                self.max_thread.acquire()
                try:
                    t = threading.Thread(target=self.label_thread, 
                                     args =(sp_label_path, i,))
                    t.start()
                except:
                    pass
                            
            print "Acquire label_thread_lock"
            self.label_thread_lock.acquire()
            self.log.start("Writing Segment Label File", 
                           len(self.image_label_array_list), 1)
            try:
                for i in range(len(self.image_label_array_list)):    
                    self.segment_label_index['label'] = \
                        np.concatenate((self.segment_label_index['label'], 
                                self.label_thread_index[i]['label']))
                    self.segment_label_index['sp'] = \
                        np.concatenate((self.segment_label_index['sp'], 
                                self.label_thread_index[i]['sp']))
                    self.segment_label_index['image'] = \
                        np.concatenate((self.segment_label_index['image'], 
                                np.ones(self.label_thread_index[i]['sp'].shape)*i))
                    self.segment_label_index['spSize'] = \
                        np.concatenate((self.segment_label_index['spSize'], 
                                self.label_thread_index[i]['spSize']))
                    self.log.update()
            finally:
                try:
                    self.label_thread_lock.release()
                except:
                    pass

            
            
             
            write_dict_to_matfile(self.segment_label_index, sp_label_path +
                                                              "/segIndex.mat")   
            
                    

    def label_thread(self, sp_label_path, i):#Deprecated
        '''
        '''

        self.label_thread_index[i] = {}

        if not os.path.isfile(sp_label_path + "/" + self.image_list[i]
                                      + ".mat"):
            image_label = self.image_label_array_list[i]
            sp_ind = np.unique(self.sp_array_list[i])
            

            for j in range(len(sp_ind)):
                
                #initiate mask for current sp index
                mask = (self.sp_array_list[i] == sp_ind[j])
                
                #count pixels for current segment
                num_pix = len(mask[mask])
                
                #get labels in current segment from labels
                ls = np.unique(image_label[mask])
                
                #count label in segment
                counts = np.bincount(image_label[mask])
                #take only labels with count > 0
                counts = counts[counts>0]

                #filter labels with more/equal counts as the half of the segment
                ls = ls[counts>=num_pix*0.5]
                counts = counts[counts>=num_pix*0.5]
                
                #take labels with number bigger/equal than 1
                counts = counts[ls>=1]
                ls = ls[ls>=1]

                #check if at least one label is chosen for segment
                if ls.size>0:
                    ind = np.argmax(counts) #index for major label
                    if self.label_thread_index[i]:
                        #add label to label_array of current image
                        self.label_thread_index[i]['label'] = np.concatenate((self.label_thread_index[i]['label'],
                                                    np.array([ls[ind]])))
                        #set index of current superPixel for the label
                        self.label_thread_index[i]['sp'] = np.concatenate((self.label_thread_index[i]['sp'],
                                                        np.array([j])))
                        #set SuperPixel-Size to current SuperPixel
                        self.label_thread_index[i]['spSize'] = np.concatenate((
                                                self.label_thread_index[i]['spSize'],
                                                np.array([num_pix])))
                    else:
                        self.label_thread_index[i]['label'] = np.array([ls[ind]])
                        self.label_thread_index[i]['sp'] = np.array([j])
                        self.label_thread_index[i]['spSize'] = np.array([num_pix])
                
            
            if not self.label_thread_index[i]:
                print "Empty file {0} with name {1}".format(i, 
                                                            self.image_list[i])
                self.label_thread_index[i]['label'] = np.array([], dtype=float)
                self.label_thread_index[i]['sp'] = np.array([], dtype=float)
                self.label_thread_index[i]['spSize'] = np.array([], dtype=float)
                
                
                
            
                        
            write_dict_to_matfile(self.label_thread_index[i], sp_label_path + "/" + 
                                          self.image_list[i] + ".mat")
        else:
            self.label_thread_index[i] = read_dict_from_matfile(sp_label_path + "/" + 
                                            self.image_list[i] + ".mat")
                    
                    

        self.log_lock.acquire()            
        try:
            self.log.update()
            if self.log.process_percentage>=100 :
                try:
                    self.label_thread_lock.release()
                except:
                    print 'here'
                    pass
        finally:
            self.log_lock.release()
        
        
        self.max_thread.release()
    
    
    def load_evaluation_set(self, test_set_file_path):#Deprecated
        '''
        loads evaluation set from given path
        '''
        self.log.start("Loading Evaluation Set", 1, 1)
        test_files = read_arr_from_txt(test_set_file_path, dtype = np.str)
        test_file_names = np.array([os.path.splitext(basename(test_file))[0]
                                    for test_file in test_files])
        
        self.test_file_indices = np.array([i for i in range(len(
                                        self.image_list)) if self.image_list[i]
                                           in test_file_names])
        
        print self.test_file_indices.shape
        
        self.train_file_indices = np.array([i for i in range(len(
                                        self.image_list)) if i not in 
                                            self.test_file_indices])
        print self.train_file_indices.shape
        
        self.log.update()
    
    
