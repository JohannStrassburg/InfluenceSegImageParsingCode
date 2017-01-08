'''
Created on 25 Apr 2014

@author: johann strassburg
superParsing.py: 
Starting point for superParsing testing
use superParsing.py to segment images
further process with Matlab-Code, provided by Lazebnik et al.

for usage instructions see usage()
'''
import Loader.Loader
from utils.utils import Logger, show_image, show_overlayed_superpixels,\
    show_superpixel_boundary, base_ubuntu, base_gnome,\
    sub_experiments
import numpy as np
from SegmentDescriptors.segmentFeatureExtractor import segmentFeatureExtractor
import os
import getopt

#from Evaluation.Evaluator import Evaluator
#from SuperPixels.SuperPixels import SPCalculator
#from utils.Statistics import SP_Statistic

    
def usage():
    '''
    prints the usage of the script if called wrong or asked for 'help'
    '''
    print "USAGE of the Script:\n"+\
                "-h: help\n"+\
                "first parameter:\n"+\
                "\t'qs': Quick_Shift\n"+\
                "\t'fh': Standard SuperParsing Method (Graph-Based)\n"+\
                "\t'gt': Ground Truth\n"+\
                "\t'grid': GRID\n"+\
                "\t'slic': SLIC\n"+\
                "\t'sal': Saliency\n"+\
                "further parameters:\n"+\
                "\tQuick Shift:\n"+\
                "\t\t[ratio, kernelsize, maxdist]\n"+\
                "\tSLIC:\n"+\
                "\t\t[region_size, regularizer]\n"+\
                "\tGround Truth:\n"+\
                "\t\t[seg_folder]"+\
                "\tSaliency:\n"+\
                "\t\t[k]"+\
                "\tGRID:\n"+\
                "\t\t[subdividing_factor]"

def super_parsing(argv):
    '''
    @param method: SuperPixel Method to test
        Possible Methods are:
            "fh", "qs", "slic"
    @param params: Params Array for given SuperPixels method
    implements the superParsing Pipeline
    '''
    
    #handling Arguments
    try:
        opts, args = getopt.getopt(argv, "h", ["help"])
        
    except:
        usage()
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()                     
            sys.exit(1)
    
    if len(args)>0:
        
        if args[0]=='fh': #Graph Based
            method = 'fh'
            
        elif args[0]=='qs': #Quick Shift
            if len(args)==4:
                method = args[0]
            else:
                usage()
                sys.exit(2)
        elif args[0]=='slic': #SLIC
            if len(args)==3:
                method = args[0]
            else:
                usage()
                sys.exit(2)
        elif args[0]=='gt': #Ground Truth
            if len(args)==1:
                method = args[0]
            else:
                usage()
                sys.exit(2)
        
        elif args[0]=='sal': #Saliency
            if len(args)==2:
                method = args[0]
            else:
                usage()
                sys.exit(2)
                
        elif args[0]=='grid': #GRID
            if len(args)==2:
                method = args[0]
            else:
                usage()
                sys.exit(2)
                
                        
    else:
        method = 'fh'
    
    #Setting up Environment
    log = Logger()
    sfe = segmentFeatureExtractor()
    
    using_ubuntu = False
    
    #setting paths
    #Ubuntu Paths
    if using_ubuntu:
        home_base = base_ubuntu 
        home_folder = home_base + sub_experiments
        
    #Debian Paths
    else:
        home_base = base_gnome 
        home_folder = home_base + sub_experiments
    

        
    
    seg_method = {}
    ############## Segmentation Methods ################
    
    if method == 'qs':
        #################### Quick Shift ###################
        seg_method['method'] = "Quick_Shift"
        seg_method['ratio'] = float(args[1])#0.05
        seg_method['kernelsize'] = int(args[2])#2
        seg_method['maxdist'] = int(args[3])#48
    if method == 'slic':
        ####################  SLIC  ########################
        seg_method['method'] = "SLIC"
        seg_method['region_size'] = int(args[1])#60
        seg_method['regularizer'] = int(args[2])#100
    if method == 'gt':
        #################### Ground Truth ##################
        seg_method['method'] = "Ground_Truth"
    if method == 'sal':
        #################### Saliency #######################
        seg_method['method'] = "Saliency"
        seg_method['k'] = int(args[1])
    if method == 'grid':
        #################### GRID ###########################
        seg_method['method'] = 'GRID'
        seg_method['k'] = int(args[1])
        
    elif method == 'fh':
        ####################### FH ##########################
        seg_method['method'] = "SP"
        seg_method['desc'] = "Desc"
        seg_method['k'] = "k200"
    #"SP_Desc_k200"
    
    
    ############################ PATH SETTINGS #################################
    seg_folder = seg_method['method']
    seg_params = [seg_method[key] 
                  for key in np.sort(np.array(seg_method.keys())) 
                                            if key!='method']
    
    for i in range(len(seg_params)):
        seg_folder = seg_folder + "_"+str(seg_params[i])
        
    
    if not os.path.exists(home_folder + seg_folder):
        os.makedirs(home_folder + seg_folder)
    
    home_images = home_base + "Images"
    home_label_sets = (home_folder + seg_folder+"/GeoLabels", 
                           home_folder + seg_folder+"/SemanticLabels")
    label_sets = ("GeoLabels", "SemanticLabels")
    home_sp_label_folder = (home_label_sets[0] + "/SP_Desc_k200",
                                home_label_sets[1] + "/SP_Desc_k200")
#   home_sp_label_folder = (home_folder + "SPGeoLabels",
#                                 home_folder + "SPSemanticLabels")
    home_data = home_folder + seg_folder + "/Data"
         
    home_descriptors = home_data + "/Descriptors"
         
    home_descriptors_global = home_descriptors + "/Global"
    descriptors_global_folder = np.array(["coHist", "colorGist", 
                                          "SpatialPyrDenseScaled",
                                          "Textons/mr_filter", 
                                          "Textons/sift_textons"])
         
    global_descriptors = np.array(['coHist', 'colorGist', 
                                       'SpatialPyrDenseScaled', 'Textons'])
    
    
    
    
    
    
    home_descriptors_segment = home_descriptors + "/SP_Desc_k200"# + "/" + seg_folder
    
    if not os.path.exists(home_descriptors_segment):
        os.makedirs(home_descriptors_segment)
        
    home_super_pixels = home_descriptors_segment + "/super_pixels"
    
    if not os.path.exists(home_super_pixels):
        os.makedirs(home_super_pixels)
    save_path = home_folder + seg_folder + "/Python/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    test_set_file_path = home_folder + "TestSet1.txt"
    
    
    if method == 'gt':
        seg_method['label_folder'] = home_label_sets[1]
    if method == 'sal':
        seg_method['saliency_folder'] = home_folder + seg_folder + '/saliency'
    ############################################################################
    
    
    ################# Sequential Reading #######################################
    #variable to setup a sequential reading of mat files (saving ram power)
    seq_reading = True
    
    #loading Images
    loader = Loader.Loader.DataLoader(save_path,save_mode = False, seq_reading=seq_reading)
    
    
    loader.load_images(home_images)
    

    #loading/Calculate SuperPixels from/as mat files
    loader.load_super_pixel(home_super_pixels, seg_method)
    
    
    #stats = SP_Statistic(loader)
    #stats.mean_sp_num()
    
        
#    evaluator = Evaluator(loader)

    ##### For testing purposes only. seq_reading needs to be set to False (careful with huge data) #####
#     for i in range(50):
#         show_superpixel_boundary(loader.image_array_list[i], loader.sp_array_list[i], 1.0-1.0/3, 1.0/3)
        #show_overlayed_superpixels(loader.image_array_list[i], loader.sp_array_list[i], 1.0-1.0/3, 1.0/3)
 
 
 
    ### Code to compute global and local Desriptors ###
    ### small differences in some descriptors ###
    ### Recommandation: Use Lazebnik et al.'s MATLAB Code instead ###
    
    #loading Global Descriptors from mat files
    #loader.load_global_descriptors(home_descriptors_global, descriptors_global_folder)
    #loader.load_centers(home_descriptors_global)
    
    #loading Segment Descriptors
    #!DEPRECATED!# Use Lazebnik's matlab code instead
    #loader.log.verbose=True
    #loader.log.verbose = True
    #loader.load_segment_descriptors(home_descriptors_segment, seg_folder)
    #loader.log.verbose = False
    #loader.log.verbose=False
#     loader.calculate_segment_features(home_descriptors_segment)
    
    #label Images
    #loader.label_images(home_label_sets[1], home_sp_label_folder[1])
    
    #setting test/trainset
    #loader.load_evaluation_set(test_set_file_path)
    
    #evaluating results
    #not implemented ... use Lazebnik's matlab code instead
    
if __name__ == '__main__':
    '''
    '''
    #profile.run('super_parsing()')
    
    import sys
    
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    
    
    super_parsing(sys.argv[1:])
        

    