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
import pylab as plt
import csv
import Image

#Global Parameters
#base_ubuntu = "/home/johann/SiftFlow/"
base_ubuntu = "/home/johann/Barcelona/"
#base_gnome = "/home/jstrassb/im_parser/SiftFlow/"
base_gnome = "/data/jstrassb/SuperParsing/SiftFlow/" #local hdd on jackson
#base_gnome = "/data/jstrassb/SuperParsing/Barcelona/"
base_ubuntu_external = "/media/johann/Patrec_external5/SuperParsing/SiftFlow/"
# base_gnome_external = "/media/Patrec_external5/SuperParsing/SiftFlow/"
#base_ubuntu_external = "/media/johann/Patrec_external5/SuperParsing/Barcelona/"
base_gnome_external = "/media/Patrec_external5/SuperParsing/Barcelona/"
sub_experiments = "Experiments/"
sub_label_sets = ("/GeoLabels", 
                  "/SemanticLabels")
sub_data = "/Data"
sub_descriptors = sub_data + "/Descriptors"
sub_descriptors_segment = sub_descriptors +"/SP_Desc_k200"

object_labels = (np.array(['sky', 'horizontal', 'vertical']),
        np.array(['awning','balcony', 'bird', 'boat', 'bridge',
                                   'building', 'bus', 'car', 'cow', 'crosswalk', 
                                   'desert', 'door', 'fence', 
                                   'field', 'grass', 'moon', 
                                   'mountain', 'person', 'plant', 
                                   'pole', 'river', 'road', 'rock', 
                                   'sand', 'sea', 'sidewalk', 'sign', 
                                   'sky', 'staircase', 'streetlight', 
                                   'sun', 'tree', 'window']))

object_labels_barcelona = (np.array(['sky', 'horizontal', 'vertical']),
                np.array(['air conditioning', 'airplane', 'animal', 'apple', 
                          'awning', 'bag', 'balcony', 'basket', 'bed', 'bench', 
                          'bicycle', 'bird', 'blind', 'boat', 'book', 
                          'bookshelf', 'bottle', 'bowl', 'box', 'branch', 
                          'brand name', 'bridge', 'brushes', 'building', 'bus', 
                          'cabinet', 'candle', 'car', 'carpet', 'cat', 
                          'ceiling', 'central reservation', 'chair', 'cheetah', 
                          'chimney', 'clock', 'closet', 'cloud', 
                          'coffeemachine', 'column', 'cone', 'counter top', 
                          'cpu', 'crocodile', 'crosswalk', 'cup', 'curb', 
                          'curtain', 'cushion', 'deer', 'dishwasher', 'dog', 
                          'door', 'drawer', 'duck', 'elephant', 'eye', 'face', 
                          'faucet', 'fence', 'field', 'firehydrant', 'fish', 
                          'flag', 'floor', 'flower', 'foliage', 'fork', 
                          'fridge', 'frog', 'furniture', 'glass', 'goat', 
                          'grass', 'ground', 'hand', 'handrail', 'head', 
                          'headlight', 'hippo', 'jar', 'keyboard', 'knife', 
                          'knob', 'lamp', 'land', 'landscape', 'laptop', 'leaf', 
                          'leopard', 'license plate', 'light', 'lion', 'lizard', 
                          'magazine', 'manhole', 'mirror', 'motorbike', 
                          'mountain', 'mouse', 'mousepad', 'mug', 'napkin', 
                          'object', 'orange', 'outlet', 'painting', 'paper', 
                          'parkingmeter', 'path', 'pen', 'person', 'phone', 
                          'picture', 'pillow', 'pipe', 'plant', 'plate', 'pole', 
                          'poster', 'pot', 'pumpkin', 'river', 'road', 'rock', 
                          'roof', 'sand', 'screen', 'sculpture', 'sea', 'shelf', 
                          'sidewalk', 'sign', 'sink', 'sky', 'snake', 'snow', 
                          'socket', 'sofa', 'speaker', 'spoon', 'stair', 
                          'stand', 'stove', 'streetlight', 'sun', 'switch', 
                          'table', 'tail light', 'teapot', 'television', 'text', 
                          'tiger', 'towel', 'tower', 'traffic light', 'trash', 
                          'tree', 'truck', 'umbrella', 'van', 'vase', 'wall', 
                          'water', 'wheel', 'window', 'windshield', 'wire', 
                          'worktop', 'zebra']))

def read_arr_from_file(path, filename):
    '''
    reads file from path (mat or txt file) and returns the numpy array
    @param path: path to file
    @param filename: index-name used in mat-files
    '''
    
    if path.endswith('.mat'):
        return read_arr_from_matfile(path, filename)
    if path.endswith('.txt'):
        return read_arr_from_txt(path)
    
def read_arr_from_txt(path, dtype=float):
    '''
    reads txt file from given path and returns the numpy array
    data read as floats
    @param path: path to file
    '''
    return np.loadtxt(path, dtype=dtype)

def read_arr_from_matfile(path,filename):
    '''
    reads mat file from path and returns the numpy array
    @param path: path to file
    @param filename: index-name used in mat-files
    '''
    m = np.array(ml.loadmat(path)[filename])
    return m

def write_dict_to_matfile(d, filepath):
    '''
    stores the given dict into file in filepath
    @param d: dict to store
    @param filepath: path to store dict at
    '''
    ml.savemat(filepath, d)

def read_dict_from_matfile(path):
    '''
    reads mat file from path and returns the dict
    '''
    return ml.loadmat(path)



def write_arr_to_matfile(arr, filepath, filename):
    '''
    stores the given numpy array into file in filepath
    @param arr: numpy array
    @param filepath: path for storing the mat-file
    @param filename: index-name for mat-file
    '''
    output_dict = {}
    output_dict[filename] = arr
    ml.savemat(filepath, output_dict)


def path_to_pathlist(path, filt=".jpg"):
    '''
    search files in given path and returns a pathlist to files, given by the filter
    @param path: path to read in (folder)
    @param filt: file-filter to consider specific file types (eg. .jpg, .mat, .png)
    '''
    return [path + "/" + f for f in os.listdir(path) if f.endswith(filt)]

def path_to_subfolder_pathlist(path, filt=".jpg"):
    '''
    search files in given path and subfolders and returns a pathlist, given by
    the filter
    @param path: path to read in (folder)
    @param filt: file-filter to consider specific file types (eg. .jpg, .mat, .png)
    '''
    
    p = np.array([f for f in [os.path.join(dirname, filename) 
                     for (dirname, _, filenames) in os.walk(path) 
                     for filename in filenames] 
         if f.endswith(filt) ])
    return np.sort(p)


def path_to_file_name_list(path, filt=".jpg"):
    '''
    search files in given path and returns a file list, given by the filter
    @param path: path to read in (folder)
    @param filt: file-filter to consider specific file types (eg. .jpg, .mat, .png)
    '''
    return [f for f in os.listdir(path) if f.endswith(filt)]


def write_arr_to_file(arr, filepath):
    '''
    writes the given array to a txt-file, located at filepath
    @param arr: numpy array to be written
    @param filepath: path of the file
    '''    
    np.savetxt(filepath, arr)
        
         
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
    @param image: image to be visualized
    @param super_pixels: superpixels-array
    @param im_alpha: alpha for image
    @param sp_alpha: alpha for superpixel
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
    plots the given image and a superpixel boundaries
    @param image: image to visualize
    @param super_pixels: superpixels used for boundaries
    @param im_alpha: Deprecated - ignore
    @param sp_alpha: Deprecated - ignore    
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
    
    
    full_border = np.zeros(mask.shape)
    struct = np.ones((5,5))
    tmp_mask = (mask==1)*1
    dil_mask = ndimage.binary_dilation(tmp_mask,struct).astype(tmp_mask.dtype)
    er_mask = ndimage.binary_erosion(tmp_mask,struct).astype(tmp_mask.dtype)
    full_border[(dil_mask-er_mask)==1]=1
    

    (y,x) = full_border.nonzero()
    (r,c) = full_border.shape
    
    top = min(y)
    bottom = max(y)
    left = min(x)
    right = max(x)
    
    yVals = np.array([[i for _ in range(c)] for i in range(r)])
    xVals = np.array([[i for i in range(c)] for _ in range(r)])
    
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
                 log_file = "log_file.txt", verbose_count = 500):
        '''
        @param verbose: set verbose to True to get a log every (verbose_count-th) update call
        @param export: set export to True to export log file
        @param log_file: name of log file for export
        @param verbose_count: regulates verbose update logs
        '''
        self.description = "Doing nothing"
        
        self.verbose_count = verbose_count
        self.total_process = 0
        self.current_process = 0
        self.update_num = 1
        self.start_time = 0
        self.verbose = verbose
        self.export = export
        if export:
            self.log_file_name = log_file

        
    def set_log_file(self, export, log_file = "log_file.txt"):
        '''
        @param export: set the export function to True/False
        @param log_file: set the log_file name
        '''
        
        self.export = export
        if export:
            self.log_file_name = log_file

    
    def start(self, description, total_process, update_num = 1):
        '''
        start Logging
        @param description: Description of the Current Process
        @param total_process: number of total update call of current logging process
        @param update_num: update steps (of total_process) per update call
        '''
        
        if self.export:
            self.log_file=open(self.log_file_name,'a')
        self.description = description
        self.total_process = total_process
        self.current_process = 0.0
        self.update_num = update_num
        self.process_percentage = 0
        
        s = "#########################################################\n" +\
                "Starting Process: " + self.description + "\n\n" + "0%\n"
                
        print s
        if self.export:
            self.log_file.write(s)
        self.start_time = time.time()
        
        
    def update(self, description=""):
        '''
        updates parameters of current process
        @param description: additional description to print
        '''
        self.current_process = self.current_process + 100*self.update_num
        if self.verbose:
            s = "{0}/{1}".format(self.current_process/100, self.total_process)
            print s
            if self.export:
                self.log_file.write(s+"\n")
        else:
            if (self.current_process/100)%self.verbose_count==0:
                s = "{0}/{1}".format(self.current_process/100, 
                                       self.total_process)
                print s
                if self.export:
                    self.log_file.write(s+"\n")
                    
        if ((int)(self.current_process/self.total_process)-self.process_percentage)>=5:
            self.process_percentage = min(
                            (int)(self.current_process/self.total_process),100)
            
            s = "{0}%".format(self.process_percentage)
            print s
            if self.export:
                self.log_file.write(s+"\n")
                
            if len(description)>0:
                
                s = description
                if self.export:
                    self.log_file.write(s+"\n")
            if self.process_percentage >= 100:
                self.end()

                
    def end(self):
        '''
        ends the logging
        '''
        
        total_time = (time.time()-self.start_time)/60.0
        
        outstr = self.description + " ... DONE!\n\n" +\
            "Calculation time: {0} Minutes".format(total_time)+"\n"+\
            "#########################################################\n"
        print outstr
        if self.export:
            self.log_file.write(outstr) 
            self.log_file.close()
            

    

class CSVWriter(object):
    '''
    writes Data to CSV Files
    '''
    
    def __init__(self, save_path, labels):
        '''
        @param save_path: path to save csv-file
        @param labels: labels of csv-file
        '''
        csv.register_dialect("tab", delimiter="\t", 
                                         quoting=csv.QUOTE_ALL)
        
        self.save_path = save_path
        self.labels = labels
            # (num_sp, total size, min_size, max_size, mean_size)
        
        self.init(save_path, labels)
            
    def init(self, save_path,labels):
        '''
        @param save_path: path to save csv-file
        @param labels: labels of csv-file
        '''
        
        self.save_path = save_path
        self.labels = labels
        self.data = [{}]
        
        self.writer = csv.DictWriter(open(self.save_path, "wb"),
                            self.labels, dialect = 'excel-tab')
        
        data = {}
        for l in self.labels:
            data[l] = l    
        self.writer.writerow(data)
        
    def add(self,content):
        '''
        adds content to data
        '''
        if len(content)!=len(self.labels):
            print "Lenght of Content and Labels does not match"
            raise ValueError
        
        else:
            data = {}
            for i in range(len(content)):
                data[self.labels[i]] = content[i]
                
            self.data.append(data)
            
    def write_csv(self):
        '''
        writes csv file
        '''
        
        self.writer.writerows(self.data)
        
            
        
        

def save_sp_boundary(sp_path, save_path, experiment):
    '''
    stores superpixel-image as superpixel-boundary-image
    @param sp_path: file-path to superpixel-file
    @param save_path: file-path for storage
    @param experiment: experiment name 
    '''
    p_list = path_to_subfolder_pathlist(sp_path ,filter=".mat")
    path_list = [path for path in p_list 
                 if not os.path.isfile(save_path + '/'+ \
                          os.path.splitext(basename(path))[0]+'.jpg')]
    
    sp_arr_list = [read_arr_from_matfile(path_list[i],"superPixels") 
         for i in range(len(path_list))]    
    u_list = [np.unique(sp_arr) for sp_arr in sp_arr_list]
    
    if len(path_list)>0:
        overlay = np.zeros((sp_arr_list[0].shape[0], sp_arr_list[0].shape[1], 3))
        log = Logger(verbose=False)
        log.start("Saving SP Boundaries for {0}".format(experiment),len(path_list),1)
        for i in range(len(sp_arr_list)):
            
            if not os.path.isfile(save_path + '/'+ \
                              os.path.splitext(basename(path_list[i]))[0]+'.jpg'):
                                
            
                mask = np.zeros(sp_arr_list[i].shape)
            
                for sp in u_list[i]:
                    tmp_mask = (sp_arr_list[i]==sp)*1
                    dil_mask = ndimage.binary_dilation(tmp_mask).astype(tmp_mask.dtype)
                    mask[(dil_mask-tmp_mask)==1]=1
            
                overlay[mask==1] = [0,0,0]
                overlay[mask==0] = [1,1,1]

                        
                fig = plt.figure(frameon=False)
                fig.set_size_inches(2.56,2.56)
                ax = plt.Axes(fig, [0.,0.,1.,1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(overlay, aspect='normal')
                fig.savefig(save_path + '/'+ os.path.splitext(basename(path_list[i]))[0]+'.jpg')
                plt.close('all')
                
            log.update()  
        
    else:
        print "Skipping SP Boundaries for {0}".format(experiment)
            
        

def save_sp_boundary_colored(sp_path, save_path, experiment):
    '''
    stores superpixel-image as colored superpixel-boundary-image
    @param sp_path: file-path to superpixel-file
    @param save_path: file-path for storage
    @param experiment: experiment name  
    '''
    
    p_list = path_to_subfolder_pathlist(sp_path ,filter=".mat")
    path_list = [path for path in p_list 
                 if not os.path.isfile(save_path + '/'+ \
                          os.path.splitext(basename(path))[0]+'.jpg')]
    
    sp_arr_list = [read_arr_from_matfile(path_list[i],"superPixels") 
         for i in range(len(path_list))]
    u_list = [np.unique(sp_arr) for sp_arr in sp_arr_list]
    len_u = [len(u) for u in u_list]    
    
    if len(path_list)>0:
        overlay = np.zeros((sp_arr_list[0].shape[0], sp_arr_list[0].shape[1], 3))
        log = Logger(verbose=False)
        log.start("Saving SP Color Boundaries for {0}".format(experiment),len(path_list),1)
        for i in range(len(sp_arr_list)):
            
            if not os.path.isfile(save_path + '/'+ \
                              os.path.splitext(basename(path_list[i]))[0]+'.jpg'):
                
                num_cluster = len_u[i]#len(unique)
  
                
                
                color_int_arr = np.array(
                            range(num_cluster))*(360.0/num_cluster)
             
                                 
                color_array = np.array([hsvtorgb(color_int_arr[k]) 
                                for k in range(num_cluster)])
         

                
                for u in u_list[i]:#unique:
                    overlay[sp_arr_list[i] == u] = color_array[u%num_cluster]

            
                        
                fig = plt.figure(frameon=False)
                fig.set_size_inches(2.56,2.56)
                ax = plt.Axes(fig, [0.,0.,1.,1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(overlay, aspect='normal')
                fig.savefig(save_path + '/'+ os.path.splitext(basename(path_list[i]))[0]+'.jpg')
                plt.close('all')
                
            log.update() 
            
    else:
        print "Skipping SP Color Boundaries for {0}".format(experiment)    


def save_sp_boundaries(system = 'ubuntu', hdd = 'e', mult ='a'):
    '''
    stores superpixel-images as colored/simple superpixel-boundary-images
    @param system: ubuntu/gnome defines the storage folder (see global parameters)
    @param hdd: 'e' sets storage path to external hard drive (see global parameters)
    @param mult: 'a' sets reading path to all available experiments (see global parameters)
                - single experiments-path not implemented: leave mult as 'a'
    '''
    
    #Checking on which system program is running - setting base Folder
    if system == 'ubuntu':
        base = base_ubuntu
        if hdd == 'e':
            base = base_ubuntu_external
                
    elif system == 'gnome':
        base = base_gnome
        if hdd == 'e':
            base = base_gnome_external
            
    multiple = False
    if mult == 'a':
        multiple = True
        
                             
    #setting path where super_pixels are in
    if multiple:
        path_list = path_to_file_name_list(base+sub_experiments, "")
            
        log = Logger(verbose=True)
        log.start("Multiple Superpixels Boundary Image calculation", 
                      len(path_list), 1)
        for experiment in path_list:
            try:
                sp_path = base + sub_experiments + experiment +\
                    sub_descriptors_segment + "/super_pixels"
                
                if not os.path.exists(sp_path):
                    print "Folder {0} doesn't have SP Data".format(experiment)
                    continue
                
                if "GRID" in experiment:
                    print "Skipping {0}, not relevant".format(experiment)
                    continue 
                    
                save_global = base + "SP_Boundary"
                save_global_colored = base + "SP_Boundary_Colored"
                if not os.path.exists(save_global):
                    os.makedirs(save_global)
                if not os.path.exists(save_global_colored):
                    os.makedirs(save_global_colored)
                    
                save_path = save_global + '/' + experiment
                save_path_colored = save_global_colored + '/' + experiment
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    
                if not os.path.exists(save_path_colored):
                    os.makedirs(save_path_colored)
                    
                save_sp_boundary(sp_path,
                                    save_path,experiment)
                
                save_sp_boundary_colored(sp_path, save_path_colored,experiment)

            except:
                print "Error: " + experiment + " not a folder."
                raise 
                
                log.update("Calculated: "+experiment)
                

def _save_sp_overlay(image, save_path, colored_path, boundary_path, image_path):
    '''
    stores image with superpixel overlay
    @param image: given image name
    @param save_path: folder path to store new overlay image
    @param colored_path: path to colored superpixel-image folder
    @param boundary_path: path to boundary superpixel image folder
    @param image_path: path to image folder
    '''
    
    background = np.array(Image.open(image_path + '/' + image + '.jpg'))
    overlay = np.array(Image.open(colored_path + '/' + image + '.jpg'))
    overlay2 = np.array(Image.open(boundary_path + '/' + image + '.jpg'))
    
    overlay[overlay2==0] = 0
    background[overlay2==0] = 0
    
   
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2.56,2.56)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(background, aspect='normal', alpha = 0.7)
    ax.imshow(overlay, aspect='normal', alpha = 0.3)

    fig.savefig(save_path + '/'+ image+'.jpg')
    plt.close('all')

def save_sp_overlay(image = "coast_arnat59", system = 'ubuntu', hdd = 'e', mult ='a'):
    '''
    stores images with superpixel overlay
    @param image: given image name
    @param system: ubuntu/gnome defines the storage folder (see global parameters)
    @param hdd: 'e' sets storage path to external hard drive (see global parameters)
    @param mult: 'a' sets reading path to all available experiments (see global parameters)
                - single experiments-path not implemented: leave mult as 'a'
    '''
    
    #Checking on which system program is running - setting base Folder
    if system == 'ubuntu':
        base = base_ubuntu
        if hdd == 'e':
            base = base_ubuntu_external
                
    elif system == 'gnome':
        base = base_gnome
        if hdd == 'e':
            base = base_gnome_external
            
    multiple = False
    if mult == 'a':
        multiple = True
        
                
    #setting path where super_pixels are in
    if multiple:
        path_list = path_to_file_name_list(base+sub_experiments, "")
            
        log = Logger(verbose=True)
        log.start("Multiple Superpixels Boundary Image calculation", 
                      len(path_list), 1)
        for experiment in path_list:
            try:
                
                    
                save_global = base + "SP_Boundary"
                save_global_colored = base + "SP_Boundary_Colored"
                    
                save_path = save_global + '/' + experiment
                save_path_colored = save_global_colored + '/' + experiment
                
                image_path = base+ 'Images'
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    continue
                    
                if not os.path.exists(save_path_colored):
                    os.makedirs(save_path_colored)
                    continue
                
                save_global_mixed = base + "SP_Overlay"
                if not os.path.exists(save_global_mixed):
                    os.makedirs(save_global_mixed)
                
                save_path_overlay = save_global_mixed + '/' + experiment
                
                if not os.path.exists(save_path_overlay):
                    os.makedirs(save_path_overlay)
                try:    
                    _save_sp_overlay(image, save_path_overlay, save_path_colored, save_path, image_path)
                except:
                    print "Failure in {0}...skipping".format(experiment)
                
                
                
            except:
                print "Error: " + experiment + " not a folder."
                raise 
                
                log.update("Calculated: "+experiment)
                
                
def summarize_results(path):
    '''
    summarize ResultMRF from given experiments folder (path)
    to path/../sumResults.csv
    @param path: experiments folder containing experiments (e.g. SLIC_.., Quick_Shift_..)
    '''
    
    folder_list = [f for f in os.listdir(path)]
    
    files = {}
    experiments = {}
    i = 0
    for folder in folder_list:
        if folder != 'copy_me_to_experiment':
            try:
                file_path = path + '/' + folder + '/Data/Base/ResultsMRF.txt'
                
                lines = [line.strip() for line in open(file_path)]
                          
                files[i] = lines
                experiments[i] = folder
                i = i+1
            except:
                print "Skipping {0}".format(folder)
    
    writer = CSVWriter(os.path.abspath(os.path.join(path, '..', 
                                                    'sumResults.csv')), 
                       ['Experiment'] + files[0][0].split("\t"))
    for j in range(len(files[0])):
        for k in range(len(files.keys())):
            if j > 0:
                values = files[k][j].split("\t")
                if len(values)>1: 
                    if (j<=6) & (values[0] == 'sky'):
                        values[0] = 'sky_geo'
                    
                    writer.add([experiments[k]] + values)
    
    writer.write_csv()
                
                
def _validate(path, experiment, sum_true, sum_total):
    '''
    cross validation of one experiment
    @param path: path to experiments folder
    @param experiment: experiment name (folder name)
    @param sum_true: true labeled pixels
    @param sum_total: all labeled pixels
    '''
    
    
    if not os.path.exists(path + '/CrossValidation'):
                    os.makedirs(path + '/CrossValidation')
    if not os.path.exists(path + '/CrossValidation/total'):
                    os.makedirs(path + '/CrossValidation/total')
    
    k_fold = sum_total.shape[0]
    index = np.array(range(k_fold))
    
    writer_all = CSVWriter(path + '/CrossValidation/total/' + experiment + '.csv', 
                        ['Class','Result','k'])
    
    for k in (np.array(range(k_fold))+1):
        


        cluster = k_fold/k + 1**(k_fold%k)-0**(k_fold%k)
        
        k_true = np.zeros((cluster,sum_true.shape[1]))
        k_total = np.zeros((cluster,sum_total.shape[1]))
        k_ratio = np.zeros((cluster,sum_total.shape[1]))
        k_per_pixel = np.zeros((cluster,1))
        k_mean_class = np.zeros((cluster,1))
        
        for i in range(k_fold/k + 1**(k_fold%k)-0**(k_fold%k)):
             
            k_true[i] = np.sum(sum_true[((index>=k*i) & (index<(k*i+k)))==False], axis = 0)
            k_total[i] = np.sum(sum_total[((index>=k*i) & (index<(k*i+k)))==False], axis = 0) 
             
            k_ratio[i] = k_true[i]*1.0/k_total[i]
             
            no_class = np.isnan(k_ratio[i])
            k_ratio[i][no_class] = 0
             
            k_per_pixel[i] = sum(k_true[i])/sum(k_total[i])
            k_mean_class[i] = sum(k_ratio[i])/(len(k_ratio[i])-len(no_class[no_class]))
         
        ratio = np.sum(k_ratio, axis=0)/cluster
        per_pixel = sum(k_per_pixel)/cluster
        mean_class = sum(k_mean_class)/cluster
         

        
        
        if not os.path.exists(path + '/CrossValidation/' + str(k)):
                    os.makedirs(path + '/CrossValidation/' + str(k))
                    
        writer = CSVWriter(path + '/CrossValidation/' + str(k) + '/'+ 
                                                experiment + '.csv', 
                        ['Class','Result'])
        
        writer.add(['PerPixel'] + [per_pixel[0]])
        writer_all.add(['PerPixel'] + [per_pixel[0]] + [k])
        writer.add(['MeanClass'] + [mean_class[0]])
        writer_all.add(['MeanClass'] + [mean_class[0]] + [k])
        
        for j in range(len(ratio)):
            
            writer.add([object_labels[1][j]] + [ratio[j]])
            
            writer_all.add([object_labels[1][j]] + [ratio[j]] + [k])
            
        
        writer.write_csv()
        
    writer_all.write_csv()
            
            
    


def cross_validation(path):
    '''
    cross_validation of computed superParsing results
    stores results in path/CrossValidation/total/
    @param path: path to experiments-folder
    '''
    
    folder_list = [f for f in os.listdir(path)]
    
    i = 0
    for folder in folder_list:
        if folder != 'copy_me_to_experiment':
            try:

                prob_path = path + '/' + folder + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
                l_path = path + '/' + folder + '/SemanticLabels'

                prob_paths = [p for p in os.listdir(prob_path) if 'cache' not in p]
                try:


                    labels = np.array([read_arr_from_matfile(prob_path + '/' + p, 'L') for p in prob_paths])
                    or_labs = np.array([read_arr_from_matfile(l_path + '/' + p, 'S') for p in prob_paths])
                                    
                    #Evaluation
                    
                    label_true = np.zeros((len(prob_paths),len(object_labels[1])))
                    label_num = np.zeros((len(prob_paths),len(object_labels[1])))
                    
                    label_pix_true = np.zeros(object_labels[1].shape)
                    label_pix_num = np.zeros(object_labels[1].shape)

                                
                    for i in range(len(labels)):
                        u = np.unique(or_labs[i])
                        for l in u:
                            if l >0:
                                mask = or_labs[i]==l

                                label_pix_true[l-1] += len(labels[i][mask][(labels[i][mask]==l)].flatten())
                                label_pix_num[l-1] += len(mask[mask].flatten())
                                
                                label_true[i][l-1] += len(labels[i][mask][(labels[i][mask]==l)].flatten())
                                label_num[i][l-1] += len(mask[mask].flatten())
                    
                            
                    
                    print folder
                    _validate(path, folder, label_true, label_num)
                except:
                    print "Skipping {0}".format(folder)
                    continue; 
                
                    
            except:
                continue
            

    print 'done'


def _validate_eaw(path, export_path, experiment, sum_true, sum_total):
    '''
    cross validation of one eaw experiment
    @param path: path to experiments folder
    @param export_path: folder name for export
    @param experiment: experiment name (folder name)
    @param sum_true: true labeled pixels
    @param sum_total: all labeled pixels
    '''
    
    
    if not os.path.exists(path + '/EAW_Validation'):
                    os.makedirs(path + '/EAW_Validation')
    if not os.path.exists(path + '/EAW_Validation/total'):
                    os.makedirs(path + '/EAW_Validation/total')
    
    k_fold = sum_total.shape[0]
    index = np.array(range(k_fold))
    
    writer_all = CSVWriter(path + '/EAW_Validation/total/' + experiment + '.csv', 
                        ['Class','Result','k'])
    
    for k in [1]:
        


        cluster = k_fold/k + 1**(k_fold%k)-0**(k_fold%k)
        
        k_true = np.zeros((cluster,sum_true.shape[1]))
        k_total = np.zeros((cluster,sum_total.shape[1]))
        k_ratio = np.zeros((cluster,sum_total.shape[1]))
        k_per_pixel = np.zeros((cluster,1))
        k_mean_class = np.zeros((cluster,1))
        
        for i in range(k_fold/k + 1**(k_fold%k)-0**(k_fold%k)):
             
            k_true[i] = np.sum(sum_true[((index>=k*i) & (index<(k*i+k)))==False], axis = 0)
            k_total[i] = np.sum(sum_total[((index>=k*i) & (index<(k*i+k)))==False], axis = 0) 
             
            k_ratio[i] = k_true[i]*1.0/k_total[i]
             
            no_class = np.isnan(k_ratio[i])
            k_ratio[i][no_class] = 0
             
            k_per_pixel[i] = sum(k_true[i])/sum(k_total[i])
            k_mean_class[i] = sum(k_ratio[i])/(len(k_ratio[i])-len(no_class[no_class]))
         
        ratio = np.sum(k_ratio, axis=0)/cluster
        per_pixel = sum(k_per_pixel)/cluster
        mean_class = sum(k_mean_class)/cluster
         

        
        
        if not os.path.exists(path + '/EAW_Validation/' + export_path):
                    os.makedirs(path + '/EAW_Validation/' + export_path)
                    
        writer = CSVWriter(path + '/EAW_Validation/' + export_path + '/'+ 
                                                experiment + '.csv', 
                        ['Class','Result'])
        
        writer.add(['PerPixel'] + [per_pixel[0]])
        writer_all.add(['PerPixel'] + [per_pixel[0]] + [k])
        writer.add(['MeanClass'] + [mean_class[0]])
        writer_all.add(['MeanClass'] + [mean_class[0]] + [k])
        
        for j in range(len(ratio)):
            
            writer.add([object_labels[1][j]] + [ratio[j]])
            
            writer_all.add([object_labels[1][j]] + [ratio[j]] + [k])
            
        
        writer.write_csv()
        
    writer_all.write_csv()
            
            


def cross_validation_eaw(path, eaw_path):#!DEPRECATED
    '''
    cross_validation of computed superParsing results for edge avoiding wavelets
    stores results in path/EAW_Validation/total/
    @param path: path to experiments-folder
    @param eaw_path: path to edge avoiding wavelets
    '''
    
    folder_list = ['eaw_4','eaw_5', 'eaw_6', 'eaw_7']
    eaw_folder = ['level_4','level_5','level_6','level_7']
    sp_folder = ['index_4','index_5','index_6','index_7']
    eaw_sp = [256,64,16,4]
    i = 0
    labels = {}
    or_labs = {}
    eaw = {}
    sp_list = {} 
    
    log = Logger(verbose=False)
    log.start('Reading EAW Matrices', len(folder_list)*4, 1)
    for j in range(len(folder_list)):
        try:
 
            prob_path = path + '/' + folder_list[j] + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'

            l_path = path + '/' + folder_list[j] + '/SemanticLabels'
             
            eaw_p = eaw_path + '/' + eaw_folder[j]
             
            sp_path = eaw_path + '/relabeled/' + sp_folder[j]

 
            prob_paths = [p for p in os.listdir(prob_path) if 'cache' not in p]
            try:
 
                #read calculated labels
                labels[j] = np.array([read_arr_from_matfile(prob_path + '/' + p, 'L') for p in prob_paths])
                log.update()
                #read original labels
                or_labs[j] = np.array([read_arr_from_matfile(l_path + '/' + p, 'S') for p in prob_paths])
                log.update() 
                 
                eaw[j] = np.array([[read_arr_from_matfile(eaw_p + '/' + os.path.splitext(p)[0] + '_k_' + str(i)+'.mat','im') for i in (np.array(range(eaw_sp[j]))+1)]
                                   for p in prob_paths])
                log.update()
                sp_list[j] = np.array([read_arr_from_matfile(sp_path + '/' + p,'superPixels') for p in prob_paths])
                log.update()

 
 
            except:
                print 'Error in {0}'.format(folder_list[j])
                raise
         
        except:
            print 'Main:Error in {0}'.format(folder_list[j])
            raise 
     
     
    final_labels = {}
    k = 1
    
    log.start("Labeling EAW-Results", len(labels[0]),1) 
    for j in range(len(labels[0])):
         
        weight_map = {}
         
        for i in range(len(folder_list)):
            weight_map[i] = np.zeros(sp_list[i][j].shape)
             
            for u in np.unique(sp_list[i][j]):
                weight_map[i][sp_list[i][j] == u] = \
                    eaw[i][j][u-1][sp_list[i][j] == u]
             
            weight_map[i] = weight_map[i]**k
         
        weights = np.zeros((weight_map[0].shape[0],weight_map[0].shape[1],
                            len(folder_list)))
         
        for i in weight_map.keys():
            weights[:,:,i] = weight_map[i]
             
        ind = np.argmax(weights,axis = 2)
        final_labels[j] = np.zeros(weight_map[0].shape)
         
        for i in range(len(folder_list)):
            final_labels[j][ind == i] = labels[i][j][ind == i]
        
        log.update()
    

         
    label_true = np.zeros((len(final_labels.keys()),len(object_labels[1])))
    label_num = np.zeros((len(final_labels.keys()),len(object_labels[1])))
    

                
    log.start("Generating final labels",len(final_labels.keys()),1)                           
    for i in final_labels.keys():
        u = np.unique(or_labs[i])
        for l in u:
            if l >0:
                mask = or_labs[0][i]==l
                 
                 
                #correct labeled pixels
                label_true[i][l-1] += len(final_labels[i][mask][(final_labels[i][mask]==l)].flatten())
                #original pixel number
                label_num[i][l-1] += len(mask[mask].flatten())
        log.update()
    
            
    
    
    log.start("Validating",1,1)
    _validate_eaw(path, 'EAW_k_'+str(k), label_true, label_num)
    log.update()
            

    
    print 'done'

def eaw_validation2(path, eaw_path):#!DEPRECATED
    '''
    summarize ResultMRF from given experiments folder (path)
    '''
    folder_list = ['eaw_4','eaw_5', 'eaw_6']
    eaw_folder = ['level_4','level_5','level_6']
    sp_folder = ['index_4','index_5','index_6']
    eaw_sp = [256,64,16]

    
    log = Logger(verbose=True)    
    

    
    prob_path = [path + '/' + folder_list[j] + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
                  for j in range(len(folder_list))]
     
    prob_paths = [[p for p in os.listdir(prob_path[j]) if 'cache' not in p] 
                     for j in range(len(folder_list))]
    
    eaw_p = [eaw_path + '/' + eaw_folder[j] for j in range(len(folder_list))]


    final_labels = {}

    
    log.start("Labeling EAW-Results", len(prob_paths[0])*len(folder_list)+len(prob_paths[0]),1)
    for j in range(len(prob_paths[0])):    
         
        weight_map = {}
         
        for i in range(len(folder_list)):
            sp_path = eaw_path + '/relabeled/' + sp_folder[i]
            sp = read_arr_from_matfile(sp_path + '/' + prob_paths[i][j],'superPixels') 
            
            weight_map[i] = np.zeros(sp.shape)
             
            eaw = [read_arr_from_matfile(eaw_p[i] + '/' + \
                    os.path.splitext(prob_paths[i][j])[0] + '_k_' + str(l) + \
                    '.mat','im') for l in (np.array(range(eaw_sp[i]))+1)]

            for u in np.unique(sp):
                
                weight_map[i][sp == u] = \
                    eaw[u-1][sp == u]
             
            weight_map[i] = weight_map[i]**(i+4)
            log.update()
         
        weights = np.zeros((weight_map[0].shape[0],weight_map[0].shape[1],
                            len(folder_list)))
         
        for i in weight_map.keys():
            weights[:,:,i] = weight_map[i]
             
        ind = np.argmax(weights,axis = 2)
        final_labels[j] = np.zeros(weight_map[0].shape)
         
        for i in range(len(folder_list)):
            final_labels[j][ind == i] = read_arr_from_matfile(prob_path[i] + '/' + prob_paths[i][j], 'L')[ind == i]
        log.update()
         
    label_true = np.zeros((len(final_labels.keys()),len(object_labels[1])))
    label_num = np.zeros((len(final_labels.keys()),len(object_labels[1])))
    


    l_path = [path + '/' + folder_list[j] + '/SemanticLabels' 
              for j in range(len(folder_list))]
    
    log.start("Generating final labels",len(final_labels.keys()),1) 
    for i in final_labels.keys():
        or_labs = read_arr_from_matfile(l_path[0] + '/' + prob_paths[0][i], 'S')
        u = np.unique(or_labs)
        for l in u:
            if l >0:
                mask = or_labs==l
                 
                 
                #correct labeled pixels
                label_true[i][l-1] += len(final_labels[i][mask][(final_labels[i][mask]==l)].flatten())
                #original pixel number
                label_num[i][l-1] += len(mask[mask].flatten()) 
        
        log.update()
        

        
def eaw_validation(path, eaw_path, sm, ks, fs, method, export_path):#!DEPRECATED use eaw_val1->eaw_val2 instead for SIFT Flow Database
    '''
    validate Edge-Avoiding-Wavelets results by weighting scale into decision
    weight is weighted by m1(weights,m2(smooth,(k+f*i))
    @param path: Path to experiments folder
    @param eaw_path: path to eaw-scaling-functions (folder) (see eaw_folder for subfolder)
    @param sm: array of smooth-parameter to test
    @param ks: array of k-parameter to test
    @param fs: array of f-parameter to test
    @param method: method m to use ([0,1,2,3]->0:m1=power, m2=multiplication,
                   2:m1=m2=multiplication
    @param export_path: path to export results
    summarize ResultMRF from given experiments folder (path)
    '''

    folder_list = ['eaw_4','eaw_5', 'eaw_6']
    eaw_folder = ['level_4','level_5','level_6']
    sp_folder = ['index_4','index_5','index_6']
    eaw_sp = [256,64,16]

    
    log = Logger(verbose=True)     
    

    
    prob_path = [path + '/' + folder_list[j] + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
                  for j in range(len(folder_list))]
     
    prob_paths = [[p for p in os.listdir(prob_path[j]) if 'cache' not in p] 
                     for j in range(len(folder_list))]
    
    eaw_p = [eaw_path + '/' + eaw_folder[j] for j in range(len(folder_list))]


    final_labels = {}
    k = 1


    weight_map = {}
    for i in range(len(folder_list)):
        weight_map[i] = {}
    log.start("Labeling EAW-Results", len(prob_paths[0]),1)
    for j in range(len(prob_paths[0])):    
         
        
         
        for i in range(len(folder_list)):
            sp_path = eaw_path + '/relabeled/' + sp_folder[i]
            sp = read_arr_from_matfile(sp_path + '/' + prob_paths[i][j],'superPixels') 
            
            weight_map[i][j] = np.zeros(sp.shape)
             
            eaw = [read_arr_from_matfile(eaw_p[i] + '/' + \
                    os.path.splitext(prob_paths[i][j])[0] + '_k_' + str(l) + \
                    '.mat','im') for l in (np.array(range(eaw_sp[i]))+1)]

            for u in np.unique(sp):
                
                weight_map[i][j][sp == u] = \
                    eaw[u-1][sp == u] ### Das muss ich speichern, um EAWs kleiner zu machen
             
        log.update()
    
    weights = {}
        
    
    for j in range(len(prob_paths[0])):
         
        weights[j] = np.zeros((weight_map[0][j].shape[0],weight_map[0][j].shape[1],
                                len(folder_list)))
    
    
    
    method_list = ['expmult','expexp','multmult','multexp']
    log.start('Validation', len(np.array(range(1,71,1)))*11*2,1)
    log.start('Validation', len(sm)*len(ks)*len(fs)*len(method),1)
    for m in method:
        for smooth in sm: #np.array(range(1,71,1))*0.1:
            for k in ks: #np.array([0,1,2,3,4,5,6,7,8,9,10]):
                for f in fs: #np.array([-1.5,-2,1.5,2]):
                    
                    for j in range(len(prob_paths[0])):
             
                        for i in weight_map.keys():
                            weights[j][:,:,i] = weight_map[i][j]
            
                    for j in weights.keys():
                        for i in range(weights[j].shape[2]):
                            
                            if m == 0:
                                weights[j][:,:,i] = weights[j][:,:,i]**(smooth*(k+f*i))
   
                            elif m == 2:
                                weights[j][:,:,i] = weights[j][:,:,i]*(smooth*(k+f*i))

                    for j in range(len(prob_paths[0])):
                             
                        ind = np.argmax(weights[j],axis = 2)
                        final_labels[j] = np.zeros((weights[j].shape[0],weights[j].shape[1]))
                         
                        for i in range(len(folder_list)):
                            final_labels[j][ind == i] = read_arr_from_matfile(prob_path[i] + '/' + prob_paths[i][j], 'L')[ind == i]
                         
                    label_true = np.zeros((len(final_labels.keys()),len(object_labels[1])))
                    label_num = np.zeros((len(final_labels.keys()),len(object_labels[1])))
                    
                
                
                
                    
                    l_path = [path + '/' + folder_list[j] + '/SemanticLabels' 
                              for j in range(len(folder_list))]
                    
                    for i in final_labels.keys():
                        or_labs = read_arr_from_matfile(l_path[0] + '/' + prob_paths[0][i], 'S')
                        u = np.unique(or_labs)
                        for l in u:
                            if l >0:
                                mask = or_labs==l
                                 
                                 
                                #correct labeled pixels
                                label_true[i][l-1] += len(final_labels[i][mask][(final_labels[i][mask]==l)].flatten())
                                #original pixel number
                                label_num[i][l-1] += len(mask[mask].flatten()) 
                        
                                
                                    
                    _validate_eaw(path, export_path, 'EAW_'+method_list[m]+'_sm_'+str(smooth)+'_k_'+str(k)+'_f_'+str(f), label_true, label_num)
                    log.update()

            

    
    print 'done'
    
def val_split(path, sum_true, sum_total):
    '''
    validates results from sum_true and sum_total
    returns per_pixel results and per_class results
    @param param: !Deprecated
    @param sum_true: true labeled pixels
    @param sum_total: all labeled pixels
    '''
    
    
        
    k_fold = sum_total.shape[0]
    index = np.array(range(k_fold))
    
    for k in [1]:
        


        cluster = k_fold/k + 1**(k_fold%k)-0**(k_fold%k)
        
        k_true = np.zeros((cluster,sum_true.shape[1]))
        k_total = np.zeros((cluster,sum_total.shape[1]))
        k_ratio = np.zeros((cluster,sum_total.shape[1]))
        k_per_pixel = np.zeros((cluster,1))
        k_mean_class = np.zeros((cluster,1))
        
        for i in range(k_fold/k + 1**(k_fold%k)-0**(k_fold%k)):
             
            k_true[i] = np.sum(sum_true[((index>=k*i) & (index<(k*i+k)))==False], axis = 0)
            k_total[i] = np.sum(sum_total[((index>=k*i) & (index<(k*i+k)))==False], axis = 0) 
             
            k_ratio[i] = k_true[i]*1.0/k_total[i]
             
            no_class = np.isnan(k_ratio[i])
            k_ratio[i][no_class] = 0
             
            k_per_pixel[i] = sum(k_true[i])/sum(k_total[i])
            k_mean_class[i] = sum(k_ratio[i])/(len(k_ratio[i])-len(no_class[no_class]))
         
        per_pixel = sum(k_per_pixel)/cluster
        mean_class = sum(k_mean_class)/cluster
         

        return per_pixel[0], mean_class[0]
        


def eaw_val1(path, eaw_path, fl = [0,1,2,4]):
    '''
    before using eaw_val2, use this method
    use output of eaw_val1 as input for eaw_val2
    returns path to experiments folder, eaw_path to scaling functions, 
    folder_list of used scaling_functions, prob_path to labeling probabilities, 
    prob_paths to labeling probabilities, 
    weight_map which indicates scaling functions defined by superpixels
    @param path: path to experiments folder
    @param eaw_path: path to scaling functions
    @param fl: folder list (number array) to used scaling functions
    '''
    
    #folder_list = [f for f in os.listdir(path)]
    folder_list = np.array(['eaw_1','eaw_2', 'eaw_3', 'eaw_4'])
    eaw_folder = np.array(['level_1','level_2','level_3','level_4'])
    sp_folder = np.array(['index_1','index_2','index_3','index_4'])
    eaw_sp = np.array([4,16,64,256])
    
    folder_list = folder_list[fl]
    eaw_folder = eaw_folder[fl]
    sp_folder = sp_folder[fl]
    eaw_sp = eaw_sp[fl]
    
    

    
    log = Logger(verbose=True)
    
    prob_path = [path + '/' + folder_list[j] + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
                  for j in range(len(folder_list))]
     
    prob_paths = [[p for p in os.listdir(prob_path[j]) if 'cache' not in p] 
                     for j in range(len(folder_list))]
    
    eaw_p = [eaw_path + '/' + eaw_folder[j] for j in range(len(folder_list))]




    weight_map = {}
    for i in range(len(folder_list)):
        weight_map[i] = {}
    log.start("Labeling EAW-Results", len(prob_paths[0]),1)
    for j in range(len(prob_paths[0])):    
         
        
         
        for i in range(len(folder_list)):
            sp_path = eaw_path + '/relabeled/' + sp_folder[i]
            sp = read_arr_from_matfile(sp_path + '/' + prob_paths[i][j],'superPixels') 
            
            weight_map[i][j] = np.zeros(sp.shape)
             
            eaw = [read_arr_from_matfile(eaw_p[i] + '/' + \
                    os.path.splitext(prob_paths[i][j])[0] + '_k_' + str(l) + \
                    '.mat','im') for l in (np.array(range(eaw_sp[i]))+1)]

            for u in np.unique(sp):
                
                weight_map[i][j][sp == u] = \
                    eaw[u-1][sp == u]
             
        log.update()
    
    return path, eaw_path, folder_list, prob_path, prob_paths, weight_map

def eaw_val2(inp, method, f, bias):
    '''
    use output from eaw_val1 as input for eaw_val2
    validates weighting of labeling results by scaling functions
    weighting: method(weights,(f[i]*(bias[i]+i)) ... i indicates level of scaling function
    method can be 0 for exp, 1 for mult
    @param inp: input from eaw_val1: path, eaw_path, folder_list, prob_path, prob_paths, weight_map
    @param method: 0 or 1. 0:exponential function, 1:multiplication
    @param f: array of weighting values
    @param bias: array of bias values to normalize scaling indices i (e.g. to 1)
    '''
    path = inp[0]
    folder_list = inp[2]
    prob_path = inp[3]
    prob_paths = inp[4]
    weight_map = inp[5]
    try:
        f[1]
    except:
        l = len(weight_map.keys())
        val = f
        f = np.zeros((l,1))
        for i in range(l):
            f[i] = val
            
            
    try:
        bias[1]
    except:
        l = len(weight_map.keys())
        val = bias
        bias = np.zeros((l,1))
        for i in range(l):
            bias[i] = val
    
    weights = {}

        
    
    for j in range(len(prob_paths[0])):
         
        weights[j] = np.zeros((weight_map[0][j].shape[0],weight_map[0][j].shape[1],
                                len(folder_list)))
    
    

                    
    for j in range(len(prob_paths[0])):

        for i in weight_map.keys():
            weights[j][:,:,i] = weight_map[i][j]

    for j in weights.keys():
        for i in range(weights[j].shape[2]):
            if method == 0:
                weights[j][:,:,i] = weights[j][:,:,i]**(f[i]*(bias[i]+i))
            elif method == 1:
                weights[j][:,:,i] = weights[j][:,:,i]*(f[i]*(bias[i]+i))

    final_labels = {}
    for j in range(len(prob_paths[0])):
             
        ind = np.argmax(weights[j],axis = 2)
        final_labels[j] = np.zeros((weights[j].shape[0],weights[j].shape[1]))
        
        #reading the Labels calculated by SuperParsing 
        for i in range(len(folder_list)):
            final_labels[j][ind == i] = read_arr_from_matfile(prob_path[i] + '/' + prob_paths[i][j], 'L')[ind == i]
         
    label_true = np.zeros((len(final_labels.keys()),len(object_labels[1])))
    label_num = np.zeros((len(final_labels.keys()),len(object_labels[1])))
    



    
    l_path = [path + '/' + folder_list[j] + '/SemanticLabels' 
              for j in range(len(folder_list))]
    
    #log.start("Generating final labels",len(final_labels.keys()),1) 
    for i in final_labels.keys():
        or_labs = read_arr_from_matfile(l_path[0] + '/' + prob_paths[0][i], 'S')
        u = np.unique(or_labs)
        for l in u:
            if l >0:
                mask = or_labs==l
                 
                 
                #correct labeled pixels
                label_true[i][l-1] += len(final_labels[i][mask][(final_labels[i][mask]==l)].flatten())
                #original pixel number
                label_num[i][l-1] += len(mask[mask].flatten()) 
        
                

    
    return val_split(path, label_true, label_num)

            
            
def relabel_arr_slow(arr, big_cluster = True): #DEPRECATED because slow
    '''
    relabels the given array in the way, 
    such that the array has #unique complete clusters
    smaller unconnected areas are merged with biggest neighboring region
    used for superpixels, created by edge avoiding wavelets
    @param arr: array to be relabeled
    '''
    arr = arr.astype(int)
    u = np.unique(arr)
    
    iterate = True
    
    m = max(u)
    new_label = m+1
        
    a = {}
    
    for i in u:
        a[i] = {}
        
        a[i]['size'] = []
        
        a[i]['pos'] = []
        
        a[i]['border'] = []
        
        a[i]['label'] = []
    
    
    #relabel labels 1..n to n+1..m
    while iterate:
        
        old_value = u[0]
        if old_value > m:
            iterate = False
            continue
        
        index = np.where(arr==old_value)
        x = index[0][0]
        y = index[1][0]
        
        border = []

        flood_fill_it(x,y,arr,old_value, new_label, border, [])

        
        a[old_value]['size'] += [len(arr[arr==new_label].flatten())]
        
        a[old_value]['pos'] += [[x,y]]
        
        a[old_value]['border'] += [np.unique(border)]
        
        a[old_value]['label'] += [new_label]
                
        new_label += 1
        
        u = np.unique(arr)
    
    keys = np.array(a.keys())
    max_cluster = [np.argmax(a[k]['size']) for k in keys]
    
    #memorize current labeling
    values = {}
    for i in u:
        values[i] = i    
    
    b = {}    
    #relabel biggest areas from n+1...m back to 1..n
    for i in range(len(keys)):
        k = keys[i]
        b[k] = {}
        x = a[k]['pos'][max_cluster[i]][0]
        y = a[k]['pos'][max_cluster[i]][1]
        old_value = a[k]['label'][max_cluster[i]]
        new_label = k
        
        border = []
        points = []
        flood_fill_it(x,y,arr,old_value, new_label, border, points)
        b[k]['border'] = border
        b[k]['points'] = np.array(points)
        b[k]['size'] = a[k]['size'][max_cluster[i]]
        values[k] = k
        values[old_value] = k
        
     
    #relabel small areas from n+1..m to bigger neighbourhood areas 1..n
    
    keys = keys[np.argsort(np.array([b[k]['size'] for k in keys])*-1)]
    
    done = len(keys)
    
    for i in range(len(keys)):
        if(done == len(u)):
            continue
        
        k = keys[i]
        q_unique = []
        q_points = []
        
        ub_1 = np.unique(b[k]['border'])
        p_1 = [b[k]['points'][np.where(b[k]['border']==j)[0][0]] for j in ub_1]
        
        q_unique.append(ub_1)
        q_points.append(p_1)
        
        while q_unique:
            
            ub = q_unique.pop()
            p = q_points.pop()

            for l in range(len(ub)):
                if values[ub[l]]<=m:
                    continue
                elif done == len(u):
                    continue
                else:
                    border = []
                    points = []
                    flood_fill_it(p[l][0],
                               p[l][1],
                               arr,
                               ub[l],
                               k,
                               border,points)
                    q_unique.append(np.unique(border))
                    q_points.append([points[np.where(border==j)[0][0]] for j in q_unique[-1]])
                    values[ub[l]] = k
                    done += 1

        
def relabel_arr(arr): #USE for EAW superpixels
    '''
    relabels the given array in the way, 
    such that the array has #unique complete clusters
    smaller unconnected areas are merged with biggest neighboring region
    used for superpixels, created by edge avoiding wavelets
    @param arr: array to be relabeled
    '''
    arr = arr.astype(int)
    u = np.unique(arr)
    
    iterate = True
    
    m = max(u)
    new_label = m+1
    
    
    a = {}
    
    for i in u:
        a[i] = {}
        
        a[i]['size'] = []
        
        a[i]['pos'] = []
        
        a[i]['border'] = []
        
        a[i]['label'] = []
    
    
    #relabel labels 1..n to n+1..m
    while iterate:
        
        old_value = u[0]
        if old_value > m:
            iterate = False
            continue
        
        index = np.where(arr==old_value)
        x = index[0][0]
        y = index[1][0]
        
        border = []

        flood_fill_it(x,y,arr,old_value, new_label, border, [])

        
        a[old_value]['size'] += [len(arr[arr==new_label].flatten())]
        
        a[old_value]['pos'] += [[x,y]]
        
        a[old_value]['border'] += [np.unique(border)]
        
        a[old_value]['label'] += [new_label]
        
        
        new_label += 1
        
        u = np.unique(arr)
    
    keys = np.sort(np.array(a.keys()))[::-1]
    max_cluster = [np.argmax(a[k]['size']) for k in keys]
    sizes = np.array([a[keys[i]]['size'][max_cluster[i]] for i in range(len(max_cluster))])
    biggest_keys = np.argsort(sizes[::-1])
    max_cluster_values = [a[keys[i]]['label'][max_cluster[i]] for i in range(len(max_cluster))]
    #memorize current labeling
    values = {}
    for i in u:
        values[i] = i
    
    
    b = {}
    
    #relabel biggest areas from n+1...m back to 1..n
    for i in range(len(keys)):
        k = keys[i]
        b[k] = {}
        x = a[k]['pos'][max_cluster[i]][0]
        y = a[k]['pos'][max_cluster[i]][1]
        old_value = a[k]['label'][max_cluster[i]]
        new_label = k
        
        border = []
        arr[arr==old_value] = new_label
        
        
        b[k]['border'] = a[k]['border'][max_cluster[i]]
        bor = b[k]['border']
        border = []
        for j in range(len(b[k]['border'])):
            if (bor[j] > m) & (bor[j] in max_cluster_values):
                border +=\
                     keys[np.argwhere(max_cluster_values == bor[j])[0]]
            elif bor[j] <= m:
                for l in range(len(a[bor[j]]['border'])):
                    if bor[j] in a[bor[j]]['border'][l]:
                        val = a[bor[j]]['label'][l]
                        if val in max_cluster_values:
                            border += [bor[j]]
                        else:
                            border += [val]
                    
            else:
                border += [bor[j]]            
                
                 
        
        
        b[k]['border'] = border
        b[k]['size'] = a[k]['size'][max_cluster[i]]
        values[k] = k
        values[old_value] = k
        
    #relabel small areas from n+1..m to bigger neighbourhood areas 1..n
    
    keys = keys[np.argsort(np.array([b[k]['size'] for k in keys])*-1)]
        
    for k in keys[biggest_keys]:
        for border in b[k]['border']:
            if border > m:
                arr[arr==border] = k
    
 
                

def flood_fill(x, y, arr, old_value, new_value, border, points):
    '''
    implements the Floodfill algorithm recursively
    @param x: x-position of starting point
    @param y: y-position of starting point
    @param arr: numpy array to use flood fill on
    @param old_value: value to override
    @param new_value: new color to override old_value
    @param border: stores colors of current position x',y' if color other than new/old_value
    @param points: stores positions x',y' if color other than new/old_value
    '''

    if (x>=0) & (x<arr.shape[0]) & (y>=0) & (y<arr.shape[1]):
        if arr[x,y] == old_value:
            
            arr[x,y] = new_value
            
            flood_fill(x-1, y, arr, old_value, new_value, border, points)
            flood_fill(x, y-1, arr, old_value, new_value, border, points)
            flood_fill(x+1, y, arr, old_value, new_value, border, points)
            flood_fill(x, y+1, arr, old_value, new_value, border, points)
        
        else:
            if arr[x,y] != new_value:
                border += [arr[x,y]]
                points += [[x,y]]
            return
    else:
        return
    

def flood_fill_it(x, y, arr, old_value, new_value, border, points):
    '''
    implements the Floodfill algorithm iteratively
    @param x: x-position of starting point
    @param y: y-position of starting point
    @param arr: numpy array to use flood fill on
    @param old_value: value to override
    @param new_value: new color to override old_value
    @param border: stores colors of current position x',y' if color other than new/old_value
    @param points: stores positions x',y' if color other than new/old_value
    '''
    
    q = [[x,y]]
    
    while q:
        p = q.pop()
        x = p[0]
        y = p[1]
        
    
        if (x>=0) & (x<arr.shape[0]) & (y>=0) & (y<arr.shape[1]):
            if arr[x,y] == old_value:
            
                arr[x,y] = new_value
            
                q.append([x-1,y])
                q.append([x,y-1])
                q.append([x+1,y])
                q.append([x,y+1])
        
            else:
                if arr[x,y] != new_value:
                    border += [arr[x,y]]
                    points += [[x,y]]
        else:
            continue
        
        
    

def draw_line(arr, x0, x1, y0, y1, color):
    '''
    draw a line into given array arr between points (x0,y0) and (x1,y1) 
    in given color
    @param arr: numpy array to draw line in
    @param color: color of line 
    '''
    arr[y0,x0] = color
    arr[y1,x1] = color
    
    if (x0==x1) & (y0==y1):
        return
    
    deltax = x1 - x0
    deltay = y1 - y0
    
    if deltax == 0:
        xstep = np.array([0])
    elif deltax<0:
        xstep = np.array(range(abs(deltax)+1))*-1
    else:
        xstep = np.array(range(abs(deltax)+1))
        
    if deltay == 0:
        ystep = np.array([0])
    elif deltay<0:
        ystep = np.array(range(abs(deltay)+1))*-1
    else:
        ystep = np.array(range(abs(deltay)+1))
        
    
    x = x0
    y = y0
    if abs(deltax) >= abs(deltay):
        delta = abs(deltay*1.0/deltax)
        err = 0
        j = 0
        for i in range(len(xstep)):
            x = x0+xstep[i]

            y = y0 + ystep[j]
            err = err + delta
            if err >= 0.5:
                j+= 1
                err -= 1
            
            arr[y,x] = color
                
    elif abs(deltay) > abs(deltax):
        delta = abs(deltax*1.0/deltay)
        err = 0
        j = 0
        for i in range(len(ystep)):
            x = x0+xstep[j]
            y = y0 + ystep[i]
            err = err + delta
            if err >= 0.5:
                j+= 1
                err -= 1
            arr[y,x] = color
            
            
    
def annotations2labels(annotation_path,output_path_sem, output_path_geo, image_home):
    '''
    converts Annotations in a given folder from xml to mat files
    
    @param annotation_path: path to annotations (given as xml files)
    @param output_path_sem: 
    @param output_path_geo: 
    @param image_home: 
    '''
    import xml.etree.ElementTree as ET
    import re
        
    
    sem_names = np.array(['air conditioning', 'airplane', 'animal', 'apple', 
                          'awning', 'bag', 'balcony', 'basket', 'bed', 'bench', 
                          'bicycle', 'bird', 'blind', 'boat', 'book', 
                          'bookshelf', 'bottle', 'bowl', 'box', 'branch', 
                          'brand name', 'bridge', 'brushes', 'building', 'bus', 
                          'cabinet', 'candle', 'car', 'carpet', 'cat', 
                          'ceiling', 'central reservation', 'chair', 'cheetah', 
                          'chimney', 'clock', 'closet', 'cloud', 
                          'coffeemachine', 'column', 'cone', 'counter top', 
                          'cpu', 'crocodile', 'crosswalk', 'cup', 'curb', 
                          'curtain', 'cushion', 'deer', 'dishwasher', 'dog', 
                          'door', 'drawer', 'duck', 'elephant', 'eye', 'face', 
                          'faucet', 'fence', 'field', 'firehydrant', 'fish', 
                          'flag', 'floor', 'flower', 'foliage', 'fork', 
                          'fridge', 'frog', 'furniture', 'glass', 'goat', 
                          'grass', 'ground', 'hand', 'handrail', 'head', 
                          'headlight', 'hippo', 'jar', 'keyboard', 'knife', 
                          'knob', 'lamp', 'land', 'landscape', 'laptop', 'leaf', 
                          'leopard', 'license plate', 'light', 'lion', 'lizard', 
                          'magazine', 'manhole', 'mirror', 'motorbike', 
                          'mountain', 'mouse', 'mousepad', 'mug', 'napkin', 
                          'object', 'orange', 'outlet', 'painting', 'paper', 
                          'parkingmeter', 'path', 'pen', 'person', 'phone', 
                          'picture', 'pillow', 'pipe', 'plant', 'plate', 'pole', 
                          'poster', 'pot', 'pumpkin', 'river', 'road', 'rock', 
                          'roof', 'sand', 'screen', 'sculpture', 'sea', 'shelf', 
                          'sidewalk', 'sign', 'sink', 'sky', 'snake', 'snow', 
                          'socket', 'sofa', 'speaker', 'spoon', 'stair', 
                          'stand', 'stove', 'streetlight', 'sun', 'switch', 
                          'table', 'tail light', 'teapot', 'television', 'text', 
                          'tiger', 'towel', 'tower', 'traffic light', 'trash', 
                          'tree', 'truck', 'umbrella', 'van', 'vase', 'wall', 
                          'water', 'wheel', 'window', 'windshield', 'wire', 
                          'worktop', 'zebra'])
    
    geo_indices = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 
                          3, 2, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 3, 3, 
                          3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 
                          3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 
                          3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                          3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 
                          1, 3, 3, 1, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 
                          3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                          3, 1, 3, 3, 3, 3, 3, 3])
    
    #geo_names = np.array(['horizontal', 'sky', 'vertical'])
    geo_names = read_arr_from_matfile('/home/johann/test_folder/names_geo.mat','names')
    semantic_label = {}
    #geo_label = [[0,0,0] for i in range(len(sem_names))]
    
    tag_list = [('air conditioning',['air conditioning','air conditionig','air mirroring','air vent']),
        ('airplane',['airplane']),
        ('animal',['animal','animal cow','animal cows','animal figure','animal flamingo','animal horse','animal lion','animal lion cub','animal lion linoness','animal lion lioness','animals','animals flamingos','ant','anteater','Antelope','Antelope, African Antelope']),
        ('apple',['apple','appels']),
        ('awning',['awning','awining']),
        ('bag',['bag']),
        ('balcony',['balcony','balconies','Balconey','bacony']),
        ('basket',['basket']),
        ('bed',['bed','bedpost','bedskirt']),
        ('bench',['bench','bench table']),
        ('bicycle',['bicycle','bicyclist','bicyclists','bicylist','bike','biker','bikes']),
        ('bird',['bird','Bird beak','bird face','bird wing']),
        ('blind',['blind','bllind']),
        ('boat',['boat','boat bumper','boat front','boat hand rail','boat life ring','boat radar','boat ramp','boat side','boat small fishing-boat','boat small pneumatic','bow','bow of boat','bow of boat2']),
        ('book',['book']),
        ('bookshelf',['bookshelf','book stand','boolshelf']),
        ('bottle',['botle','bottl','bottle','bottle cap','bottle warmer']),
        ('bowl',['bowl']),
        ('box',['box']),
        ('branch',['branch','branch no foliage','branches','branchs']),
        ('brand name',['brand name']),
        ('bridge',['bridge']),
        ('brushes',['bruhses','brush','brushes']),
        ('building',['buiding','buidling','buildig','buildigs','buildin','buildin2','building','building distant','building flats','building light','building railing','building shop','buildins','buildling','buildlings','builidng','builing','bulding','buliding','sky-scraper','skyscrape']),
        ('bus',['bus','bus advertisement']),
        ('cabinet',['cabinet']),
        ('candle',['candle','candle light']),
        ('car',['car','car az0deg','car az120deg','car az150deg','car az180deg','car az210deg','car az240deg','car az270deg','car az300deg','car az30deg','car az330deg','car az60deg','car az90deg','car bumper','car ear','car end','car light','Car reflection','car shadow','carRear az90deg az90deg']),
        ('carpet',['carpet']),
        ('cat',['cat','cat face','cat figurine','cat mouth','cat nose','cat paw','cat tail']),
        ('ceiling',['ceiling','ceiling beam','ceiling fan','ceiling light','ceiling vent','CeilingLamp','CeilingLight','ceilling','ceilng light']),
        ('central reservation',['central reservation','centra reservation','centrar reservation']),
        ('chair',['chair','chair az0deg','chair az120deg','chair az150deg','chair az180deg','chair az210deg','chair az240deg','chair az270deg','chair az300deg','chair az30deg','chair az330deg','chair az60deg','chair az90deg','chair figurine','chair leg','chair legs','chais lounge']),
        ('cheetah',['cheetah','cheetah cub','cheetah left mid grassland','cheetah left mid savanna','cheetah left near desert','cheetah left near forest','cheetah left near grassland','cheetah other mid savanna','cheetah right mid desert','cheetah right mid grassland','cheetah right mid savanna','Cheetha']),
        ('chimney',['chimney','chimny']),
        ('clock',['clock']),
        ('closet',['closet']),
        ('cloud',['cloud']),
        ('coffeemachine',['coffeemachine','coffe machine','coffe maker','coffe marker','coffee filter','coffee grinder','coffee pack','coffee pot','coffee shop','coffee stirrer','coffee tablwe','coffeePot','coffeeTable','coffemachine']),
        ('column',['column','collumn','collumn base','colon']),
        ('cone',['cone']),
        ('counter top',['counter top','counertop','counter','counter bottom','counter ledge','countertop','coutnertop']),
        ('cpu',['cpu']),
        ('crocodile',['crocodile','crocodile front far grassland','crocodile front mid water_','crocodile front near forest','crocodile left mid forest','crocodile left mid grassland','crocodile left mid water','crocodile left near forest','crocodile left near grassland','crocodile other mid water','crocodile rear mid water','crocodile right mid forest','crocodile right mid grassland','crocodile right mid water','crocodile right near grassland']),
        ('crosswalk',['crosswalk','crosswalk light','crosswalk marker']),
        ('cup',['cup','cup and saucer']),
        ('curb',['curb','curb2']),
        ('curtain',['curtain','cutain','curtainrod']),
        ('cushion',['cushion','cushio']),
        ('deer',['deer']),
        ('dishwasher',['dishwasher','dish washer','dishwashing soap']),
        ('dog',['dog','dog dish','dog face','dog nose']),
        ('door',['door','Door, entrance','dor','Double dorr']),
        ('drawer',['drawer','drawer pull','drawers','drawers empty']),
        ('duck',['duck']),
        ('elephant',['elephant','elephant calf','Elephant Leg','elephant tail','Elephant Trunk']),
        ('eye',['eye']),
        ('face',['face','face (side view)','face ce','face here is thhe face of the person wallking along the road','Face of a Baby','face profile','face side','face, grandma','facemask','faceProfile']),
        ('faucet',['faucet','fawcett']),
        ('fence',['fence','fencing','fense']),
        ('field',['field','field of grass','fields']),
        ('firehydrant',['firehydrant']),
        ('fish',['fish','fish head']),
        ('flag',['flag']),
        ('floor',['floor','Floor Tile']),
        ('flower',['flower','flower bush','flower field']),
        ('foliage',['foliage','foliageRegion']),
        ('fork',['fork']),
        ('fridge',['fridge']),
        ('frog',['frog','frog front mid forest','frog front mid water','frog front near desert','frog front near forest','frog left mid desert','frog left mid forest','frog left mid grassland','frog left near desert','frog left near grassland','frog other mid grassland','frog rear mid forest','frog right mid desert','frog right mid forest','frog right mid grassland','frog top mid grassland','frog top mid water']),
        ('furniture',['furniture']),
        ('glass',['glass','Glas','glassware']),
        ('goat',['goat','goat front far mountain','goat front mid mountain','goat kid','goat left mid mountain','goat left near forest','goat left near mountain','goat right mid forest','goat right mid mountain','goat right near mountain']),
        ('grass',['grass','gras','grass and leaves','grassy hill']),
        ('ground',['ground','groound','ground leaves','ground leaves and rocks','ground rocks']),
        ('hand',['hand']),
        ('handrail',['handrail','hand rail']),
        ('head',['head','head az0deg','head az120deg','head az150deg','head az180deg','head az210deg','head az240deg','head az270deg','head az300deg','head az30deg','head az330deg','head az60deg','head az90deg']),
        ('headlight',['headlight','head light','Headlamp (White Hatchback)']),
        ('hippo',['hippo','hippo front mid water','hippo left mid desert','hippo other mid desert','hippo right mid water','hippopotamus','hipppo']),
        ('jar',['jar','jar *X','jar filled with pasta','jar of beans','jar of cooking utensils','jar of cotton balls','jar of pens','jar of pinecones','jar of tea','jar of utensils','jar of utincils','jar soap','jars']),
        ('keyboard',['keyboard','key board','key board shelf']),
        ('knife',['knife','knives']),
        ('knob',['knob']),
        ('lamp',['lamp','Lamp cord','Lamp, Table','lam_p']),
        ('land',['land']),
        ('landscape',['landscape']),
        ('laptop',['laptop','lap top','laptop az0deg','laptop az240deg','laptop az270deg','laptop az300deg','laptop az330deg']),
        ('leaf',['leaf','leaf f','leafs','leave','leaves']),
        ('leopard',['leopard','leopard cub','leopard front mid forest','leopard front near water','leopard left mid forest','leopard left mid grassland','leopard left mid water','leopard left near forest','leopard left near water','leopard other mid forest','leopard right mid forest','leopard right mid mountain','leopard right mid savanna','leopard right mid water','leopard right near mountain','leopard right near water','Leopard Tongue','leppard']),
        ('license plate',['license plate','licence plate','licenst plate']),
        ('light',['light','Light bulb','light fixture','lightbulb','lights','ligiht']),
        ('lion',['lion','lion face','lion front far savanna','lion front mid grassland','lion front mid mountain','lion front mid savanna','lion front near mountain','lion left far grassland','lion left mid mountain','lion left mid savanna','lion other mid savanna','lion other near forest','lion plate','lion rear mid savanna','lion right mid grassland','lion right mid mountain','lion right mid savanna','lion right mid water','lion right near grassland','lion right near mountain','lioness','lions','Lios']),
        ('lizard',['lizard','lizard front near forest','lizard left near desert','lizard left near forest','lizard left near mountain','lizard right mid desert','lizard right mid forest','lizard right near desert','lizard right near forest','lizard right near mountain','lizard top mid desert','lizard top near forest','lizard top near mountain','lizards']),
        ('magazine',['magazine','magizine']),
        ('manhole',['manhole','manhol']),
        ('mirror',['miror','mirror','mirrow']),
        ('motorbike',['motorbike','motobikes side','motocycle','motor','motor bike','motor cycle','motorclyclist','Motorcyclist','motorcyclist occluderd','motortbike side','motrorbike']),
        ('mountain',['mountain','mountain fog','mountainRegion','mountan','mountanious region','mountanious region covered by grass','moutain','muntain']),
        ('mouse',['mouse']),
        ('mousepad',['mousepad']),
        ('mug',['mug','mug handle interior']),
        ('napkin',['napkin','napkin holder','napkin ring']),
        ('object',['object','objecct','object_1','object_3','objectds','objectr_2','objects']),
        ('orange',['orange']),
        ('outlet',['outlet']),
        ('painting',['painting','paitings']),
        ('paper',['paper']),
        ('parkingmeter',['parkingmeter']),
        ('path',['path']),
        ('pen',['pen','pen holde','pen holder','pen lid','pen set']),
        ('person',['person','perosn','perdestrian','perso standing','person az0deg','person az120deg','person az150deg','person az180deg','person az210deg','person az240deg','person az270deg','person az300deg','person az30deg','person az330deg','person az60deg','person az90deg','peson standing','peson walking','pesron','pesrson walking','people','people az180deg az180deg','people body','people mover','People near railing','people riding','people side sitting','people sitting','people walking','people wanking','people wise','people_','peoples','2 women','girl','girl walking','girls','boy','boy and girl','a girl','kid','kid standing','kid walking','kid, boy','sitting people','people']),
        ('phone',['phone']),
        ('picture',['picture','pictue','pic','photo','picture label']),
        ('pillow',['pillow','pills','pilow']),
        ('pipe',['pipe']),
        ('plant',['plant','plant bush','plants flowers','plants shrubs']),
        ('plate',['plate']),
        ('pole',['pole']),
        ('poster',['poster','poste']),
        ('pot',['pot','pot hanger','pot holder','pot hole','pot lid','pot of herbs','pot on a stand','pot rack']),
        ('pumpkin',['pumpkin','pumkin']),
        ('river',['river','river bank','rives water']),
        ('road',['road','road marking','road paint','road traffic']),
        ('rock',['rock','rock boulder','rock cliff','rock in water','rock island','rock pile','rock stone','rocks moss','rocky ledge','rocky moutain','rocky plain','roks']),
        ('roof',['roof','roof feature','roof ornament','roof rack','roof tile']),
        ('sand',['sand','sand dune','sand dunes']),
        ('screen',['screen','screen az0deg','screen az210deg','screen az240deg','screen az270deg','screen az300deg','screen az330deg','screen az90deg']),
        ('sculpture',['sculpture','sculpyure']),
        ('sea',['sea','sea shore rocks','seashore','ocean']),
        ('shelf',['shelf','shelf label','shell','shell *X','shelves','shelves empty','shelving','sheslves']),
        ('sidewalk',['sidewalk','side walk','sidealk','sidewalk cafe','sideWalkSign','sideway','sidewsalk','sidwalk','sidwwalk','siewalk','sildewalk']),
        ('sign',['sign','sig','sigh','Sign (pharmacy)','singn','sings','obscuredSign']),
        ('sink',['sink']),
        ('sky',['skly','sky','siky','sky fog','sky light']),
        ('snake',['snake','snake front mid desert','snake left mid grassland','snake left mid water','snake right near grassland','snake top mid desert','snake top mid grassland','snake top mid mountain','snake top near desert','snake top near forest','snake top near grassland','snake top near mountain','snake top near water']),
        ('snow',['snow','snow covered','snow covered branch','snow covered plain','snow covered railing','snow covered valley','snow land','snowflake','snowflake decoration in profile','snowy plain']),
        ('socket',['socket']),
        ('sofa',['sofa','soffa']),
        ('speaker',['speaker']),
        ('spoon',['spoon','spoom','spoon handle']),
        ('stair',['stair','staicase','stair board','stair case','stair rail','staircase railing','stairwell']),
        ('stand',['stand']),
        ('stove',['stove']),
        ('streetlight',['streetlight','Street Lights','street light shadow','street lantern','streetligt','streetlilght']),
        ('sun',['sun']),
        ('switch',['switch']),
        ('table',['table','table &amp; chairs','table and chairs','table and chairs *X','table leg','table legs','Table of Books','table salt','tablecloth','Tables and Chaires']),
        ('tail light',['tail light','tail light (left) (VW Golf)','Tail Light (Right) (VW Golf)']),
        ('teapot',['teapot']),
        ('television',['television','television stand','televison']),
        ('text',['text']),
        ('tiger',['tiger','tiger baby','tiger front mid grassland','tiger front near grassland','tiger in the snow','tiger left mid forest','tiger left mid grassland','tiger left mid mountain','tiger left near forest','tiger left near grassland','tiger right mid forest','tiger right mid grassland','tiger right mid water','tiger right near forest','tiger right near water']),
        ('towel',['towel']),
        ('tower',['tower']),
        ('traffic light',['traffic light','traffic  lights','traffic-light','traffic-lights','traffice light','trafic light']),
        ('trash',['trash']),
        ('tree',['tree','tre','tree knot','tree shadow','trres']),
        ('truck',['truck','truck az180deg','truk','trunk']),
        ('umbrella',['uberella','umbrela','umbrella']),
        ('van',['van','van az0deg','van az240deg','van az30deg']),
        ('vase',['vase']),
        ('wall',['wall','wall art','wall decoration','wall decoration *x','wall fixture','wall hanging','wall light','Wall Ligth','wall tapestry','wall vent *?']),
        ('water',['water','water drop','water fall','water lake','water mist','water ocean spray','water ocean wave','water pond','water region','water spurt','water surf','Water surface','water waterfall','water waterfall frozen','water waterfalls','water wave','water, rocks']),
        ('wheel',['wheel','Wheel excavator','Wheel spinning','winding']),
        ('window',['window','pane','window ledge','window shudder','Window Sill','window side']),
        ('windshield',['windshield','windshield wiper','windwo']),
        ('wire',['wire']),
        ('worktop',['worktop']),
        ('zebra',['zebra'])]
    
    
    tag_dict = {}
    for i in range(len(tag_list)):
        for word in tag_list[i][1]:
            tag_dict[word]=i
            
    
    for i in range(len(sem_names)):
        semantic_label[sem_names[i]] = i+1
        
        
        
        

    sem_names = read_arr_from_matfile('/home/johann/test_folder/names_sem.mat','names')
    a = read_arr_from_matfile('/home/johann/test_folder/coast_natu912.mat', 'ind')
    
    path_list = path_to_subfolder_pathlist(annotation_path, filter=".xml")
    
    
    log = Logger(verbose=False)
    log.start('Relabeling Annotations', len(path_list), 1)

    for path in path_list:
    #### Reading XML-File
        tree = ET.parse(path)
        #tree = ET.parse(im_name+'.xml')
        root = tree.getroot()
        
        image_name = os.path.basename(os.path.splitext(path)[0])
        dir_name = os.path.basename(os.path.dirname(path))
        image_path = image_home + '/' +dir_name + '/' + image_name + '.jpg'
        a = np.array(Image.open(image_path))
        if os.path.exists(output_path_sem + '/'+dir_name + '/' + image_name + '.mat'):
            if os.path.exists(output_path_geo + '/'+dir_name + '/' + image_name + '.mat'):
                log.update()
                continue
        
        b = np.zeros((a.shape[0],a.shape[1]))

        folder =''
        
        c = 2
        
        labs = {}
        for child in root:
            if child.tag == 'folder':
                folder = ''.join(child.text.split('\n'))
            if child.tag == 'object':
                for gchild in child:
                    if gchild.tag == 'finaltag':
                        label = ''.join(gchild.text.split('\n'))
                    
                    if gchild.tag == 'polygon':
                        points = []
                        for pt in gchild:
                            for p in pt:
                                if p.tag == 'x':
                                    x = int(int(re.split(r'\n',p.text)[1])-2)
                                    if x>= b.shape[1]:
                                        x = b.shape[1]-1
                                    elif x<0:
                                        x = 0

                                if p.tag == 'y':
                                    y = int(int(re.split(r'\n',p.text)[1])-2)
                                    if y>= b.shape[0]:
                                        y = b.shape[0]-1
                                    elif y<0:
                                        y = 0

                            if pt.tag == 'pt':
                                points += [[x,y]]
                        
                        points += [points[0]]
                        d = np.zeros(b.shape)
                        for i in range(len(points)-1):
                            draw_line(d, points[i][0], points[i+1][0], points[i][1], points[i+1][1], 1)
                            
                        ind = np.argwhere(d==1)
                        (ystart, xstart),(ystop, xstop) = ind.min(0), ind.max(0)+1
                        mask = d[ystart:ystop,xstart:xstop]
                        it = True
                        tmp = mask.copy()
                        ind = np.argwhere(tmp!=1)

                        while it:

                            try:
                                x = ind[len(ind)/2][0]
                                y = ind[len(ind)/2][1]
                                flood_fill_it(x, y, tmp, 0, c, [], [])
                                ind2 = np.argwhere(tmp==c)
                                (ystart2, xstart2),(ystop2, xstop2) = ind2.min(0), ind2.max(0)+1
                            
                            
                                #check if filled area reached and of image and is not within polygon
                                if (ystart2 == 0) | (xstart2 == 0) | (ystop2 == tmp.shape[0]) | (xstop2 == tmp.shape[1]):
                                    tmp[tmp==c] = -1
                                    ind = np.argwhere(tmp==0)
                            
                                else:
                                    it = False
                            except:
                                print "Skipping one Label Filling for image {0}".format(image_name)
                                it = False
                        
     
                        for i in range(len(points)-1):
                            draw_line(d, points[i][0], points[i+1][0], points[i][1], points[i+1][1], c)
                        
                        d[ystart:ystop,xstart:xstop][tmp == c]=c
                        
                        

                        b[ystart:ystop,xstart:xstop][tmp == c] = d[ystart:ystop,xstart:xstop][tmp == c]
                              
                
                labs[c] = label                
                c+=1
                

                        
                    
        if not os.path.exists(output_path_sem + '/'+folder):
            os.makedirs(output_path_sem + '/'+folder)
        
        if not os.path.exists(output_path_geo + '/'+folder):
            os.makedirs(output_path_geo + '/'+folder)
            
        final_labels = np.zeros(b.shape,dtype = np.uint8)
        final_labels_geo = np.zeros(b.shape, dtype = np.uint8)
        for key in labs.keys():
            print labs[key]
            if labs[key] in tag_dict.keys():
                final_labels[b==key] = tag_dict[labs[key]]+1
                final_labels_geo[b==key] = geo_indices[tag_dict[labs[key]]]
        
        out_file = {}
        out_file['S'] = final_labels
        out_file['names'] = sem_names
        unique = np.unique(final_labels)
        ll = [tag_list[l-1][0] for l in unique if l!=0]
        print "Image: {0}, Labels: {1}, Names: {2}".format(image_name, np.unique(final_labels), ll)
        write_dict_to_matfile(out_file, 
                          output_path_sem + '/'+folder + '/' + image_name + '.mat')
        out_file['S'] = final_labels_geo
        out_file['names'] = geo_names
        write_dict_to_matfile(out_file, 
                          output_path_geo + '/'+folder + '/' + image_name + '.mat')

        log.update()
  
  

def eaw_concatenate(sp_path, scaling_path, out_path):
    '''
    concatenates scaling files of one image into one scaling file based on superpixel indices
    @param sp_path: path to superpixels
    @param scaling_path: path to scaling paths
    @param out_path: output path to store scaling files 
    '''
     
    sp_list = path_to_subfolder_pathlist(sp_path, filter=".mat")
    
    log = Logger()
    
    log.start("Concatenating EAW", len(sp_list), 1) 
    for sp_path in sp_list:
         
        sp = read_arr_from_matfile(sp_path, "superPixels")
       
        out_arr = np.zeros(sp.shape)
        indices = np.unique(sp)
        #read scaling functions and store relevant data into out_arr
        for i in indices:
           
            scaling_arr = read_arr_from_matfile(scaling_path+"/" +\
                         os.path.basename(os.path.dirname(sp_path)) + "/" +\
                        os.path.splitext(os.path.basename(sp_path))[0]+\
                        "_k_" + str(i)+".mat","im")
             
            out_arr[sp==i] = scaling_arr[sp==i]
         
        if not os.path.isdir(out_path + "/" + os.path.basename(os.path.dirname(sp_path))):
                os.makedirs(out_path + "/" + os.path.basename(os.path.dirname(sp_path)))
         
        write_arr_to_matfile(out_arr, 
            out_path + "/" + os.path.basename(os.path.dirname(sp_path)) + "/" + os.path.basename(sp_path), "im")       
       
        log.update() 
        
        

def hsvtorgb(h):
        '''
        converts hsv value to rgb values
        returns rgb as numpy array
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
