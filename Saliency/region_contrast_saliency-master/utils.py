'''
Created on 25 Apr 2014

@author: johann
'''

import numpy as np
import os
import time
import scipy.io.matlab as ml
import pylab
from os.path import basename
import scipy.ndimage as ndimage

#Global Parameters

base_ubuntu = "/home/johann/SiftFlow/"
base_gnome = "/home/jstrassb/im_parser/SiftFlow/"
base_ubuntu_external = "/media/johann/Patrec_external5/SuperParsing/SiftFlow/"
base_gnome_external = "media/jstrassb/Patrec_external5/SuperParsing/SiftFlow/"
sub_experiments = "Experiments/"
sub_label_sets = ("/GeoLabels", 
                  "/SemanticLabels")
sub_data = "/Data"
sub_descriptors = sub_data + "/Descriptors"
sub_descriptors_segment = sub_descriptors +"/SP_Desc_k200"

object_labels = (np.array(['sky', 'horizontal', 'vertical']),
        np.array(['awning','balcony', 'bird', 'boat', 'bridge',
                                   'building', 'car', 'cow', 'crosswalk', 
                                   'desert', 'door', 'fence', 
                                   'field', 'grass', 'moon', 
                                   'mountain', 'person', 'plant', 
                                   'pole', 'river', 'road', 'rock', 
                                   'sand', 'sea', 'sidewalk', 'sign', 
                                   'sky', 'staircase', 'streetlight', 
                                   'sun', 'tree', 'window']))

def read_arr_from_file(path, filename):
    '''
    reads file from filename and returns the numpy array
    '''
    
    if path.endswith('.mat'):
        return read_arr_from_matfile(path, filename)
    if path.endswith('.txt'):
        return read_arr_from_txt(path)
    
def read_arr_from_txt(path, dtype=float):
    '''
    reads txt file from given path and returns the numpy array
    '''
    return np.loadtxt(path, dtype=dtype)

def read_arr_from_matfile(path,filename):
    '''
    reads mat file from filename and returns the numpy array
    '''

    m = np.array(ml.loadmat(path)[filename])
    return m

def write_dict_to_matfile(dict, filepath):
    '''
    stores the given dict into file in filepath
    '''
    ml.savemat(filepath, dict)

def read_dict_from_matfile(path):
    '''
    reads mat file from filename and returns the dict
    '''
    return ml.loadmat(path)



def write_arr_to_matfile(arr, filepath, filename):
    '''
    stores the given numpy array into file in filepath
    '''
    output_dict = {}
    output_dict[filename] = arr
    ml.savemat(filepath, output_dict)


def path_to_pathlist(path, filter=".jpg"):
    '''
    search files in given path and returns a pathlist, given by the filter
    '''
    #return [filepath for filepat in [os.path.join(dirname,subdirname) for subdirname in dirnames]]
    return [path + "/" + f for f in os.listdir(path) if f.endswith(filter)]

def path_to_subfolder_pathlist(path, filter=".jpg"):
    '''
    search files in given path and subfolders and returns a pathlist, given by
    the filter
    '''
    
    p = np.array([f for f in [os.path.join(dirname, filename) 
                     for (dirname, _, filenames) in os.walk(path) 
                     for filename in filenames] 
         if f.endswith(filter) ])
    return np.sort(p)


def path_to_file_name_list(path, filter=".jpg"):
    '''
    search files in given path and returns a pathlist, given by the filter
    '''
    
    return [f for f in os.listdir(path) if f.endswith(filter)]


def write_arr_to_file(arr, filepath):
    '''
    writes the given array to a file, located at filepath
    @param arr: array to be written
    @param filepath: path of the file
    '''
        
    np.savetxt(filepath, arr)
        
   
# def read_arr_from_file(filepath):
#         '''
#         returns the array, stored at filepath
#         @param filepath: path to the file to read from
#         '''
#         
#         return np.loadtxt(filepath)
 
 
def show_image(image):
    '''
    plots the given image
    '''
        
    pylab.imshow(image)
    pylab.axis('off')
    pylab.show()
        
def show_overlayed_superpixels(image, super_pixels, im_alpha = 0.9, sp_alpha = 0.1):
    '''
    plots the given image with a superpixel overlay
    '''
    
    #generating color array for different clusters
    color_int_arr = np.array(
                        range(34))*(360.0/33)
         
                             
    color_array = np.array([hsvtorgb(color_int_arr[i]) 
                            for i in range(34)])
     
    background = image
    overlay = np.zeros(background.shape)
    
    for i in range(super_pixels.shape[0]):
        for j in range(super_pixels.shape[1]):
            overlay[i,j] = color_array[super_pixels[i,j]%34]
            
     
    pylab.imshow(background,alpha=im_alpha)
    pylab.imshow(overlay, alpha=sp_alpha)
    pylab.axis('off')
    pylab.show()
    

def show_superpixel_boundary(image, super_pixels, im_alpha = 0.9, sp_alpha = 0.1):
    '''
    plots the given image with a superpixel overlay
    '''
    
     
    background = image
    overlay = np.zeros(background.shape)
    
    mask = np.zeros(super_pixels.shape)
    unique = np.unique(super_pixels)
    
    for sp in unique:
        tmp_mask = (super_pixels==sp)*1
        dil_mask = ndimage.binary_dilation(tmp_mask).astype(tmp_mask.dtype)
        mask[(dil_mask-tmp_mask)==1]=1
    
    overlay[mask==1] = [0,0,0]
    overlay[mask==0] = [1,1,1]
        
#     for i in range(super_pixels.shape[0]):
#         for j in range(super_pixels.shape[1]):
#             overlay[i,j] = color_array[super_pixels[i,j]%34]
            
    f = pylab.figure()
    f.add_subplot(2,2,1) 
    pylab.imshow(background)#, alpha=im_alpha)
    f.add_subplot(2,2,2) 
    pylab.imshow(overlay)#, alpha=sp_alpha)
    pylab.axis('off')
    
    
    color_int_arr = np.array(
                        range(34))*(360.0/33)
         
                             
    color_array = np.array([hsvtorgb(color_int_arr[i]) 
                            for i in range(34)])
     
    background = image
    overlay = np.zeros(background.shape)
    
    for i in range(super_pixels.shape[0]):
        for j in range(super_pixels.shape[1]):
            overlay[i,j] = color_array[super_pixels[i,j]%34]
            
    
    f.add_subplot(2,2,3) 
    pylab.imshow(background,alpha=im_alpha)
    pylab.imshow(overlay, alpha=sp_alpha)
    
    f.add_subplot(2,2,4) 
    pylab.imshow(overlay)
    
    pylab.show()


def get_border_bounding_box(mask):
    '''
    computes the border and bounding box to the given mask
    '''
    #check if mask contains only ones
    if (sum(mask.flatten()==1) == mask.shape[0]*mask.shape[1]):
        mask[0]=0
        mask[-1]=0
        mask[:,0]=0
        mask[:,-1]=0
    
    #strEl = strel('square',5);
    #int_mask = imerode(mask,strEl,'same');
    full_border = np.zeros(mask.shape)
    struct = np.ones((5,5))
    tmp_mask = (mask==1)*1
    dil_mask = ndimage.binary_dilation(tmp_mask,struct).astype(tmp_mask.dtype)
    er_mask = ndimage.binary_erosion(tmp_mask,struct).astype(tmp_mask.dtype)
    full_border[(dil_mask-er_mask)==1]=1
    
        
#     int_mask = ndimage.binary_erosion(mask)
#     full_border = double(imdilate(mask,strEl,'same')-int_mask);

    (y,x) = full_border.nonzero()
    (r,c) = full_border.shape
    
    top = min(y)
    bottom = max(y)
    left = min(x)
    right = max(x)
    
    yVals = np.array([[i for j in range(c)] for i in range(r)])
    xVals = np.array([[i for i in range(c)] for j in range(r)])
#     yVals = repmat((1:r)',1,c);
#     xVals = repmat(1:c,r,1);
    
    border = np.zeros((r,c,5))
    border[:,:,0] = abs(xVals-left)
    border[:,:,1] = abs(xVals-right)
    border[:,:,2] = abs(yVals-top)
    border[:,:,3] = abs(yVals-bottom)
    index = np.argmin(border[:,:,0:4],axis=2)
    border[:,:,0] = (index==0)*full_border
    border[:,:,1] = (index==1)*full_border
    border[:,:,2] = (index==2)*full_border
    border[:,:,3] = (index==3)*full_border

    struct = np.ones((20,20))
    border[:,:,4] = ndimage.binary_dilation(mask,struct)
    (y,x) = border[:,:,4].nonzero()
    border = border>0
    top = min(y)
    bottom = max(y)
    left = min(x)
    right = max(x)
    bb = [top, bottom, left, right]
    
    return (border, bb)
       

class Logger(object):
    '''
    simple Logger to print current process
    '''
    

    
    def __init__(self, verbose = False, export=False, 
                 log_file = "log_file.txt"):
        
        self.description = "Doing nothing"
        
        self.total_process = 0
        self.current_process = 0
        self.update_num = 1
        self.start_time = 0
        self.verbose = verbose
        self.export = export
        if export:
            self.log_file_name = log_file
            #self.log_file = open(log_file, 'a') 
        
    def set_log_file(self, export, log_file = "log_file.txt"):
        '''
        @param export: set the export function to True/False
        @param log_file: set the log_file name
        '''
        
        self.export = export
        if export:
            self.log_file_name = log_file
            #self.log_file = open(log_file,'a')
    
    def start(self, description, total_process, update_num = 1):
        '''
        start Logging
        @param description: Description of the Current Process
        '''
        
        if self.export:
            self.log_file=open(self.log_file_name,'a')
        self.description = description
        self.total_process = total_process
        self.current_process = 0.0
        self.update_num = update_num
        self.process_percentage = 0
        
        str = "#########################################################\n" +\
                "Starting Process: " + self.description + "\n\n" + "0%\n"
                
#         print "#########################################################"
#         print "Starting Process: " + self.description
#         print ""
#         print "0%"
        print str
        if self.export:
            self.log_file.write(str)
        self.start_time = time.time()
        
        
    def update(self, description=""):
        '''
        updates parameters of current process
        '''
        self.current_process = self.current_process + 100*self.update_num
        if self.verbose:
            str = "{0}/{1}".format(self.current_process/100, self.total_process)
            print str
            if self.export:
                self.log_file.write(str+"\n")
        else:
            if (self.current_process/100)%500==0:
                str = "{0}/{1}".format(self.current_process/100, 
                                       self.total_process)
                print str
                if self.export:
                    self.log_file.write(str+"\n")
                    
        if ((int)(self.current_process/self.total_process)-self.process_percentage)>=5:
            self.process_percentage = min(
                            (int)(self.current_process/self.total_process),100)
            
            str = "{0}%".format(self.process_percentage)
            print str
            if self.export:
                self.log_file.write(str+"\n")
                
            if len(description)>0:
                
                str = description
                if self.export:
                    self.log_file.write(str+"\n")
            if self.process_percentage >= 100:
                self.end()
#         if (self.current_process/self.total_process)%5<0.1:
#             if (self.current_process/self.total_process - self.process_percentage)>4.9:
#                 self.process_percentage = self.process_percentage + 5
#                 print "{0}%".format(self.process_percentage)
#                 if self.process_percentage >= 100:
#                     self.end()
                
    
        
    def end(self):
        '''
        ends the logging
        '''
        
        total_time = (time.time()-self.start_time)/60.0
        
        str = self.description + " ... DONE!\n\n" +\
            "Calculation time: {0} Minutes".format(total_time)+"\n"+\
            "#########################################################\n"
        print str
        if self.export:
            self.log_file.write(str) 
            self.log_file.close()
            
#         print self.description + " ... DONE!"
#         print ""
#         print "Calculation time: {0} Minutes".format(total_time)
#         print "#########################################################"
    

def hsvtorgb(h):
        '''
        converts hsv value to rgb values
        returns rgb
        @param h: hsv value
        '''
        h_=h/60.0
        c = 1
        x = c*(1-np.linalg.norm(h_%2-1))

        if (h_ >= 0) & (h_< 1):
            return np.array([c,x,0])
        elif (h_ >= 0) & (h_ < 2):
            return np.array([x,c,0])
        elif (h_ >= 0) & (h_ < 3):
            return np.array([0,c,x])
        elif (h_ >= 0) & (h_ < 4):
            return np.array([0,x,c])
        elif (h_ >= 0) & (h_ < 5):
            return np.array([x,0,c])
        elif (h_ >= 0) & (h_ < 6):
            return np.array([c,0,x])
        else:
            return np.array([0,0,0])