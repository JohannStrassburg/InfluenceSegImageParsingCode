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
# base_ubuntu_external = "/media/johann/Patrec_external5/SuperParsing/SiftFlow/"
# base_gnome_external = "/media/Patrec_external5/SuperParsing/SiftFlow/"
base_ubuntu_external = "/media/johann/Patrec_external5/SuperParsing/Barcelona/"
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
                 log_file = "log_file.txt", verbose_count = 500):
        
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
            if (self.current_process/100)%self.verbose_count==0:
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
    


class CSVWriter(object):
    '''
    writes Data to CSV Files
    '''
    
    def __init__(self, save_path, labels):
        '''
        '''
        csv.register_dialect("tab", delimiter="\t", 
                                         quoting=csv.QUOTE_ALL)
        
        self.save_path = save_path
        self.labels = labels
            # (num_sp, total size, min_size, max_size, mean_size)
        
        self.init(save_path, labels)
            
    def init(self, save_path,labels):
        '''
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
        log.start("Saving SP Boundaries for {0}".format(experiment),len(path_list),1)
        for i in range(len(sp_arr_list)):
            
            if not os.path.isfile(save_path + '/'+ \
                              os.path.splitext(basename(path_list[i]))[0]+'.jpg'):
                
                #super_pixels = sp_arr_list[i]
                
            
                mask = np.zeros(sp_arr_list[i].shape)
                #unique = np.unique(super_pixels)
            
                for sp in u_list[i]:
                    tmp_mask = (sp_arr_list[i]==sp)*1
                    dil_mask = ndimage.binary_dilation(tmp_mask).astype(tmp_mask.dtype)
                    mask[(dil_mask-tmp_mask)==1]=1
            
                overlay[mask==1] = [0,0,0]
                overlay[mask==0] = [1,1,1]
        #         
        #                 b = b*255
        #                 b[:,:,0] = b[:,:,0]*s[i]
        #                 b[:,:,1] = b[:,:,1]*s[i]
        #                 b[:,:,2] = b[:,:,2]*s[i]
                        
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
    '''
    
    p_list = path_to_subfolder_pathlist(sp_path ,filter=".mat")
    path_list = [path for path in p_list 
                 if not os.path.isfile(save_path + '/'+ \
                          os.path.splitext(basename(path))[0]+'.jpg')]
    #path_list = path_to_subfolder_pathlist(sp_path ,filter=".mat")
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
                
                #super_pixels = sp_arr_list[i]
                #unique = u_list[i]#np.unique(super_pixels)
                num_cluster = len_u[i]#len(unique)
                #overlay = np.zeros((super_pixels.shape[0], super_pixels.shape[1], 3))
                
                
                color_int_arr = np.array(
                            range(num_cluster))*(360.0/num_cluster)
             
                                 
                color_array = np.array([hsvtorgb(color_int_arr[k]) 
                                for k in range(num_cluster)])
         
                #background = image
                #overlay = np.zeros(background.shape)
                
                for u in u_list[i]:#unique:
                    overlay[sp_arr_list[i] == u] = color_array[u%num_cluster]
    #                overlay[super_pixels == u] = color_array[u%num_cluster]
    #             for k in range(super_pixels.shape[0]):
    #                 for l in range(super_pixels.shape[1]):
    #                     overlay[k,l] = color_array[super_pixels[k,l]%num_cluster]
            
                        
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
    @param system: ubuntu/gnome
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
    sp_analysis = False
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
#                     
#                 if sp_analysis:
#                         sp_class_path = (base + sub_experiments + experiment +\
#                                          sub_label_sets[0],
#                                          base + sub_experiments + experiment +\
#                                          sub_label_sets[1])
#                 
#                 
#                         self.analyze_sp_class_size(sp_class_path, safe_path, 
#                                                    experiment)
            except:
                print "Error: " + experiment + " not a folder."
                raise 
                
                log.update("Calculated: "+experiment)
                

def _save_sp_overlay(image, save_path, colored_path, boundary_path, image_path):
    '''
    '''
    
    background = np.array(Image.open(image_path + '/' + image + '.jpg'))
    overlay = np.array(Image.open(colored_path + '/' + image + '.jpg'))
    overlay2 = np.array(Image.open(boundary_path + '/' + image + '.jpg'))
    
    overlay[overlay2==0] = 0
    background[overlay2==0] = 0
    
    #mask = np.zeros(super_pixels.shape)
    #unique = np.unique(super_pixels)

            
#     f = pylab.figure()
#     f.add_subplot(2,2,1) 
#     pylab.imshow(background)#, alpha=im_alpha)
#     f.add_subplot(2,2,2) 
#     pylab.imshow(overlay)#, alpha=sp_alpha)
#     pylab.axis('off')
    
    
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2.56,2.56)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(background, aspect='normal', alpha = 0.7)
    ax.imshow(overlay, aspect='normal', alpha = 0.3)
    #ax.imshow(overlay2, aspect='normal', alpha = 0.3)
    fig.savefig(save_path + '/'+ image+'.jpg')
    plt.close('all')

def save_sp_overlay(image = "coast_arnat59", system = 'ubuntu', hdd = 'e', mult ='a'):
    '''
    @param system: ubuntu/gnome
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
    sp_analysis = False
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
                
                #colored_boundary_path = base+sub_experiments+"/"
                sp_path = base + sub_experiments + experiment +\
                    sub_descriptors_segment + "/super_pixels"
                
#                 if not os.path.exists(sp_path):
#                     print "Folder {0} doesn't have SP Data".format(experiment)
#                     continue
#                 
#                 if "GRID" in experiment:
#                     print "Skipping {0}, not relevant".format(experiment)
#                     continue 
                    
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
    '''
    
    folder_list = [f for f in os.listdir(path)]
    
    files = {}
    experiments = {}
    i = 0
    num_geo = 0
    for folder in folder_list:
        if folder != 'copy_me_to_experiment':
            try:
                file_path = path + '/' + folder + '/Data/Base/ResultsMRF.txt'
                #target_path = os.path.abspath(os.path.join(path, '..', 'ResultsMRF')) +\
                #    '/'+folder+'_ResultsMRF.txt'
                #print "Copying Results of " + folder + ' to ' + target_path
                #shutil.copyfile(file_path, target_path)
                
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
                
                
def validate(path, experiment, sum_true, sum_total):
    '''
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
        
#         k_true = np.zeros((k,sum_true.shape[1]))
#         k_total = np.zeros((k,sum_total.shape[1]))
#         k_ratio = np.zeros((k,sum_total.shape[1]))
#         k_per_pixel = np.zeros((k,1))
#         k_mean_class = np.zeros((k,1))
        
#         for i in range(k):
#             
#             k_true[i] = np.sum(sum_true[index%k==i], axis = 0)
#             k_total[i] = np.sum(sum_total[index%k==i], axis = 0) 
#             
#             k_ratio[i] = k_true[i]*1.0/k_total[i]
#             
#             no_class = np.isnan(k_ratio[i])
#             k_ratio[i][no_class] = 0
#             
#             k_per_pixel[i] = sum(k_true[i])/sum(k_total[i])
#             k_mean_class[i] = sum(k_ratio[i])/(len(k_ratio[i])-len(no_class[no_class]))
#         
#         ratio = np.sum(k_ratio, axis=0)/k
#         per_pixel = sum(k_per_pixel)/k
#         mean_class = sum(k_mean_class)/k

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
         
#         ratio = np.sum(k_ratio, axis=0)/k
#         per_pixel = sum(k_per_pixel)/k
#         mean_class = sum(k_mean_class)/k
        
        
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
            
            
    
#    writer = CSVWriter(os.path.abspath(os.path.join(path, '/', 
#                                                     experiment'sumResults.csv')), 
#                        ['Experiment'] + files[0][0].split("\t"))
#     for j in range(len(files[0])):
#         for k in range(len(files.keys())):
#             if j > 0:
#                 values = files[k][j].split("\t")
#                 if len(values)>1: 
#                     if (j<=6) & (values[0] == 'sky'):
#                         values[0] = 'sky_geo'
#                     
#                     writer.add([experiments[k]] + values)
#     
#     writer.write_csv()
    


def cross_validation(path):
    '''
    summarize ResultMRF from given experiments folder (path)
    '''
    
    folder_list = [f for f in os.listdir(path)]
    
    files = {}
    experiments = {}
    i = 0
    num_geo = 0
    class_arr = np.array([])
    for folder in folder_list:
        if folder != 'copy_me_to_experiment':
            try:
                #file_path = path + '/' + folder + '/Data/Base/SemanticLabels/rNNSearchR200K200TNN80-SPscGistCoHist'#ResultsMRF.txt'
                #prob_path = path + '/' + folder + '/Data/Base/SemanticLabels/probPerLabelR200K200TNN80-SPscGistCoHist-sc01ratio'
                prob_path = path + '/' + folder + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
                sp_path = path + '/' + folder + '/Data/Descriptors/SP_Desc_k200/super_pixels'
                lab_path = path + '/' + folder + '/SemanticLabels/SP_Desc_k200'#/segIndex.mat'
                l_path = path + '/' + folder + '/SemanticLabels'
                f_list = path_to_file_name_list(lab_path, filter=".mat")
                label = np.array([])
                sp = np.array([])
                sp_size = np.array([])
                #file_paths = [p for p in os.listdir(file_path)]
                prob_paths = [p for p in os.listdir(prob_path) if 'cache' not in p]
                try:
                    #a = read_dict_from_matfile(file_path + '/' + file_paths[0])
                    #b = read_dict_from_matfile(file_path2 + '/' + file_paths2[0])
                    #prob_arr = np.array([read_arr_from_matfile(prob_path + '/' + p, 'probPerLabel') for p in prob_paths])
         #           labels = np.array([read_dict_from_matfile(prob_path + '/' + p) for p in prob_paths])

                    labels = np.array([read_arr_from_matfile(prob_path + '/' + p, 'L') for p in prob_paths])
#                    lab_prob = np.array([np.argmax(prob_arr[i], axis=1)+1 for i in range(len(prob_arr))])
                    or_labs = np.array([read_arr_from_matfile(l_path + '/' + p, 'S') for p in prob_paths])
                    #sp_arr = np.array([read_arr_from_matfile(sp_path + '/' + p, 'superPixels') for p in prob_paths])
                    lab_arr = np.array([read_dict_from_matfile(lab_path + '/' + p) for p in prob_paths])
                    
                    #label = np.array([l['index']['label'][0][0][0] for l in lab_arr] )
                    sp = np.array([l['index']['sp'][0][0][0] for l in lab_arr] )

                    #sp_size = np.array([l['index']['spSize'][0][0][0] for l in lab_arr] )
                    
                    #file_name_list = np.array([f for f in f_list if f!='segIndex.mat'])
#                    seg_lab_index =read_dict_from_matfile(lab_path + '/segIndex.mat')
#                    file_index_list = np.array([i for i in range(len(file_name_list)) if file_name_list[i] in prob_paths])
                    
                    
                    #Evaluation
                    
                    label_true = np.zeros((len(prob_paths),len(object_labels[1])))
                    label_num = np.zeros((len(prob_paths),len(object_labels[1])))
                    
                    #label_num = np.zeros(object_labels[1].shape)
                    label_pix_true = np.zeros(object_labels[1].shape)
                    label_pix_num = np.zeros(object_labels[1].shape)
#                     
#                     
#                     for i in range(len(prob_arr)):
#                         true_clsf = lab_prob[i][sp[i]-1]==label[i]
#                         for j in range(len(true_clsf)):
#                             
#                             label_num[label[i][j]-1] += 1
#                             label_pix_num[label[i][j]-1] += sp_size[i][j]
#                             
#                             if true_clsf[j]:
#                                 label_true[label[i][j]-1] += 1
#                                 label_pix_true[label[i][j]-1] += sp_size[i][j]
                                
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
                    validate(path, folder, label_true, label_num)
                    #print "True/Total: {0}  /  {1}".format(sum(label_true), sum(label_num))
                    #print 'a'
                except:
                    #raise
                    print "Skipping {0}".format(folder)
                    continue; 
                
                #for i in len(prob_arr):
                    
            except:
                #raise
                continue
            
            #print 'a'
#             
#             
#                 
#                 #target_path = os.path.abspath(os.path.join(path, '..', 'ResultsMRF')) +\
#                 #    '/'+folder+'_ResultsMRF.txt'
#                 #print "Copying Results of " + folder + ' to ' + target_path
#                 #shutil.copyfile(file_path, target_path)
#                 
#                 lines = [line.strip() for line in open(file_path)]
#                           
#                 files[i] = lines
#                 experiments[i] = folder
#                 i = i+1
#             except:
#                 print "Skipping {0}".format(folder)
    
    print 'done'
#    writer = CSVWriter(os.path.abspath(os.path.join(path, '..', 
#                                                     'sumResults.csv')), 
#                        ['Experiment'] + files[0][0].split("\t"))
#     for j in range(len(files[0])):
#         for k in range(len(files.keys())):
#             if j > 0:
#                 values = files[k][j].split("\t")
#                 if len(values)>1: 
#                     if (j<=6) & (values[0] == 'sky'):
#                         values[0] = 'sky_geo'
#                     
#                     writer.add([experiments[k]] + values)
#     
#     writer.write_csv()

def validate2(path, export_path, experiment, sum_true, sum_total):
    '''
    '''
    
    
    if not os.path.exists(path + '/EAW_Validation'):
                    os.makedirs(path + '/EAW_Validation')
    if not os.path.exists(path + '/EAW_Validation/total'):
                    os.makedirs(path + '/EAW_Validation/total')
    
    k_fold = sum_total.shape[0]
    index = np.array(range(k_fold))
    
    writer_all = CSVWriter(path + '/EAW_Validation/total/' + experiment + '.csv', 
                        ['Class','Result','k'])
    
    for k in [1]:#(np.array(range(k_fold))+1):
        


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
            
            
    

    


def cross_validation2(path, eaw_path):
    '''
    summarize ResultMRF from given experiments folder (path)
    '''
    
    #folder_list = [f for f in os.listdir(path)]
    folder_list = ['eaw_4','eaw_5', 'eaw_6', 'eaw_7']
    eaw_folder = ['level_4','level_5','level_6','level_7']
    sp_folder = ['index_4','index_5','index_6','index_7']
    eaw_sp = [256,64,16,4]
    files = {}
    experiments = {}
    i = 0
    num_geo = 0
    class_arr = np.array([])
    labels = {}
    or_labs = {}
    eaw = {}
    sp_list = {} 
    
    log = Logger(verbose=False)
    log.start('Reading EAW Matrices', len(folder_list)*4, 1)
    for j in range(len(folder_list)):
        #if folder in ['eaw_4','eaw_5', 'eaw_6', 'eaw_7']:
        try:
 
            prob_path = path + '/' + folder_list[j] + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
            #sp_path = path + '/' + folder + '/Data/Descriptors/SP_Desc_k200/super_pixels'
            #lab_path = path + '/' + folder + '/SemanticLabels/SP_Desc_k200'#/segIndex.mat'
            l_path = path + '/' + folder_list[j] + '/SemanticLabels'
             
            eaw_p = eaw_path + '/' + eaw_folder[j]
             
            sp_path = eaw_path + '/relabeled/' + sp_folder[j]
            #f_list = path_to_file_name_list(lab_path, filter=".mat")
            #label = np.array([])
            #sp = np.array([])
            #sp_size = np.array([])
 
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
                #lab_arr = np.array([read_dict_from_matfile(lab_path + '/' + p) for p in prob_paths])
                 
 
                #sp = np.array([l['index']['sp'][0][0][0] for l in lab_arr] )
 
 
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
    
#     prob_path = [path + '/' + folder_list[j] + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
#                  for j in range(len(folder_list))]
#     
#     prob_paths = [[p for p in os.listdir(prob_path[j]) if 'cache' not in p] 
#                     for j in range(len(folder_list))]
#     
#     for j in range(len(folder_list)):
#         #if folder in ['eaw_4','eaw_5', 'eaw_6', 'eaw_7']:
#         try:
# 
#             prob_path = path + '/' + folder_list[j] + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
#             #sp_path = path + '/' + folder + '/Data/Descriptors/SP_Desc_k200/super_pixels'
#             #lab_path = path + '/' + folder + '/SemanticLabels/SP_Desc_k200'#/segIndex.mat'
#             l_path = path + '/' + folder_list[j] + '/SemanticLabels'
#             
#             eaw_p = eaw_path + '/' + eaw_folder[j]
#             
#             sp_path = eaw_path + '/relabeled/' + sp_folder[j]
#             #f_list = path_to_file_name_list(lab_path, filter=".mat")
#             #label = np.array([])
#             #sp = np.array([])
#             #sp_size = np.array([])
# 
#             prob_paths = [p for p in os.listdir(prob_path) if 'cache' not in p]
#             try:
# 
#                 #read calculated labels
#                 labels[j] = np.array([read_arr_from_matfile(prob_path + '/' + p, 'L') for p in prob_paths])
#                 
#                 #read original labels
#                 or_labs[j] = np.array([read_arr_from_matfile(l_path + '/' + p, 'S') for p in prob_paths])
#                 
#                 
#                 eaw[j] = np.array([[read_arr_from_matfile(eaw_p + '/' + os.path.splitext(p)[0] + '_k_' + str(i)+'.mat','im') for i in (np.array(range(eaw_sp[j]))+1)]
#                                    for p in prob_paths])
#                 
#                 sp_list[j] = np.array([read_arr_from_matfile(sp_path + '/' + p,'superPixels') for p in prob_paths])
#                 
#                 #lab_arr = np.array([read_dict_from_matfile(lab_path + '/' + p) for p in prob_paths])
#                 
# 
#                 #sp = np.array([l['index']['sp'][0][0][0] for l in lab_arr] )
# 
# 
#             except:
#                 print 'Error in {0}'.format(folder_list[j])
#                 raise
#         
#         except:
#             print 'Main:Error in {0}'.format(folder_list[j])
#             raise 
#     
#     
#     final_labels = {}
#     k = 1
     #read calculated labels
#                 labels[j] = np.array([read_arr_from_matfile(prob_path + '/' + p, 'L') for p in prob_paths])
#                 
#                 #read original labels
#                 or_labs[j] = np.array([read_arr_from_matfile(l_path + '/' + p, 'S') for p in prob_paths])
#                 
#                 
#                 eaw[j] = np.array([[read_arr_from_matfile(eaw_p + '/' + os.path.splitext(p)[0] + '_k_' + str(i)+'.mat','im') for i in (np.array(range(eaw_sp[j]))+1)]
#                                    for p in prob_paths])
#                 
#                 sp_list[j] = np.array([read_arr_from_matfile(sp_path + '/' + p,'superPixels') for p in prob_paths])
    #for j in range(len(labels[0])):
#     for j in range(len(prob_paths[0])):    
#         
#         weight_map = {}
#         
#         for i in range(len(folder_list)):
#             
#             weight_map[i] = np.zeros(sp_list[i][j].shape)
#             
#             for u in np.unique(sp_list[i][j]):
#                 weight_map[i][sp_list[i][j] == u] = \
#                     eaw[i][j][u-1][sp_list[i][j] == u]
#             
#             weight_map[i] = weight_map[i]**k
#         
#         weights = np.zeros((weight_map[0].shape[0],weight_map[0].shape[1],
#                             len(folder_list)))
#         
#         for i in weight_map.keys():
#             weights[:,:,i] = weight_map[i]
#             
#         ind = np.argmax(weights,axis = 2)
#         final_labels[j] = np.zeros(weight_map[0].shape)
#         
#         for i in range(len(folder_list)):
#             final_labels[j][ind == i] = read_arr_from_matfile(prob_path + '/' + prob_paths[i][j], 'L')[ind == i]
         
    label_true = np.zeros((len(final_labels.keys()),len(object_labels[1])))
    label_num = np.zeros((len(final_labels.keys()),len(object_labels[1])))
    

    #label_pix_true = np.zeros(object_labels[1].shape)
    #label_pix_num = np.zeros(object_labels[1].shape)

    
    
#     for i in final_labels.keys():
#         or_labs = read_arr_from_matfile(l_path + '/' + prob_paths[0][i], 'S')
#         u = np.unique(or_labs[i])
#         for l in u:
#             if l >0:
#                 mask = or_labs[0][i]==l
#                 
#                 #correct labeled pixels
#                 #label_pix_true[l-1] += len(labels[i][mask][(labels[i][mask]==l)].flatten())
#                 
#                 #original pixel number
#                 #label_pix_num[l-1] += len(mask[mask].flatten())
#                 
#                 #correct labeled pixels
#                 label_true[i][l-1] += len(final_labels[i][mask][(final_labels[i][mask]==l)].flatten())
#                 #original pixel number
#                 label_num[i][l-1] += len(mask[mask].flatten()) 
                
    log.start("Generating final labels",len(final_labels.keys()),1)                           
    for i in final_labels.keys():
        u = np.unique(or_labs[i])
        for l in u:
            if l >0:
                mask = or_labs[0][i]==l
                 
                #correct labeled pixels
                #label_pix_true[l-1] += len(labels[i][mask][(labels[i][mask]==l)].flatten())
                 
                #original pixel number
                #label_pix_num[l-1] += len(mask[mask].flatten())
                 
                #correct labeled pixels
                label_true[i][l-1] += len(final_labels[i][mask][(final_labels[i][mask]==l)].flatten())
                #original pixel number
                label_num[i][l-1] += len(mask[mask].flatten())
        log.update()
    
            
    
    #print folder
    
    log.start("Validating",1,1)
    validate2(path, 'EAW_k_'+str(k), label_true, label_num)
    log.update()
#                     
#                 except:
#                     
#                     print "Skipping {0}".format(folder)
#                     continue; 
#                 
#                 
#                     
#             except:
#                 
#                 continue
            

    
    print 'done'


def eaw_validation2(path, eaw_path):
    '''
    summarize ResultMRF from given experiments folder (path)
    '''
    
    #folder_list = [f for f in os.listdir(path)]
    folder_list = ['eaw_4','eaw_5', 'eaw_6', 'eaw_7']
    eaw_folder = ['level_4','level_5','level_6','level_7']
    sp_folder = ['index_4','index_5','index_6','index_7']
    eaw_sp = [256,64,16,4]
    folder_list = ['eaw_4','eaw_5', 'eaw_6']
    eaw_folder = ['level_4','level_5','level_6']
    sp_folder = ['index_4','index_5','index_6']
    eaw_sp = [256,64,16]

    
    log = Logger(verbose=True)
#    log.start('Reading EAW Matrices', len(folder_list)*4, 1)

     
    

    
    prob_path = [path + '/' + folder_list[j] + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
                  for j in range(len(folder_list))]
     
    prob_paths = [[p for p in os.listdir(prob_path[j]) if 'cache' not in p] 
                     for j in range(len(folder_list))]
    
    eaw_p = [eaw_path + '/' + eaw_folder[j] for j in range(len(folder_list))]


    final_labels = {}
    k = 1


    
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
        
        
def eaw_validation(path, eaw_path, sm, ks, fs, method, export_path):
    '''
    summarize ResultMRF from given experiments folder (path)
    '''
    
    #folder_list = [f for f in os.listdir(path)]
    folder_list = ['eaw_4','eaw_5', 'eaw_6', 'eaw_7']
    eaw_folder = ['level_4','level_5','level_6','level_7']
    sp_folder = ['index_4','index_5','index_6','index_7']
    eaw_sp = [256,64,16,4]
    folder_list = ['eaw_4','eaw_5', 'eaw_6']
    eaw_folder = ['level_4','level_5','level_6']
    sp_folder = ['index_4','index_5','index_6']
    eaw_sp = [256,64,16]

    
    log = Logger(verbose=True)
#    log.start('Reading EAW Matrices', len(folder_list)*4, 1)

     
    

    
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
             
            #weight_map[i][j] = weight_map[i][j]**(i+4)
        log.update()
    
    weights = {}
    #for i in range(len(folder_list)):
    #    #weights[i] = {}
        
    
    for j in range(len(prob_paths[0])):
         
        weights[j] = np.zeros((weight_map[0][j].shape[0],weight_map[0][j].shape[1],
                                len(folder_list)))
    
    
    #weight_map = [] #clear memory occupied by weight_map
    #log2 = Logger()
    
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
                            #elif m == 1:
                            #    weights[j][:,:,i] = weights[j][:,:,i]**(smooth**(k+f*i))
                            elif m == 2:
                                weights[j][:,:,i] = weights[j][:,:,i]*(smooth*(k+f*i))
                            #elif m == 3:
                            #    weights[j][:,:,i] = weights[j][:,:,i]*(smooth**(k+f*i))
                    for j in range(len(prob_paths[0])):
                             
                        ind = np.argmax(weights[j],axis = 2)
                        final_labels[j] = np.zeros((weights[j].shape[0],weights[j].shape[1]))
                         
                        for i in range(len(folder_list)):
                            final_labels[j][ind == i] = read_arr_from_matfile(prob_path[i] + '/' + prob_paths[i][j], 'L')[ind == i]
                        #log.update()
                         
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
                        
                        #log.update()
                                
                
                    
                    #log.start("Validating",1,1)
                    validate2(path, export_path, 'EAW_'+method_list[m]+'_sm_'+str(smooth)+'_k_'+str(k)+'_f_'+str(f), label_true, label_num)
                    log.update()

            

    
    print 'done'
    
def val_split(path, sum_true, sum_total):
    '''
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
         
        ratio = np.sum(k_ratio, axis=0)/cluster
        per_pixel = sum(k_per_pixel)/cluster
        mean_class = sum(k_mean_class)/cluster
         

        return per_pixel[0], mean_class[0]
        
#         writer.add(['PerPixel'] + [per_pixel[0]])
#         writer_all.add(['PerPixel'] + [per_pixel[0]] + [k])
#         writer.add(['MeanClass'] + [mean_class[0]])
#         writer_all.add(['MeanClass'] + [mean_class[0]] + [k])
#         
#         for j in range(len(ratio)):
#             
#             writer.add([object_labels[1][j]] + [ratio[j]])
#             
#             writer_all.add([object_labels[1][j]] + [ratio[j]] + [k])
#             
#         
#         writer.write_csv()
#         
#     writer_all.write_csv()


def eaw_val1(path, eaw_path, fl = [0,1,2,4]):
    '''
    summarize ResultMRF from given experiments folder (path)
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
    
    
#     folder_list = ['eaw_4','eaw_5', 'eaw_6']
#     eaw_folder = ['level_4','level_5','level_6']
#     sp_folder = ['index_4','index_5','index_6']
#     eaw_sp = [256,64,16]

    
    log = Logger(verbose=True)
#    log.start('Reading EAW Matrices', len(folder_list)*4, 1)

     
    

    
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
                    eaw[u-1][sp == u]
             
            #weight_map[i][j] = weight_map[i][j]**(i+4)
        log.update()
    
    return path, eaw_path, folder_list, prob_path, prob_paths, weight_map

def eaw_val2(input, method, f, bias):
    '''
    method can be 0 for exp, 1 for mult
    '''
    path = input[0]
    eaw_path = input[1]
    folder_list = input[2]
    prob_path = input[3]
    prob_paths = input[4]
    weight_map = input[5]
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
    #for i in range(len(folder_list)):
    #    #weights[i] = {}
        
    
    for j in range(len(prob_paths[0])):
         
        weights[j] = np.zeros((weight_map[0][j].shape[0],weight_map[0][j].shape[1],
                                len(folder_list)))
    
    
    #method_list = ['expmult','expexp','multmult','multexp']

                    
    for j in range(len(prob_paths[0])):

        for i in weight_map.keys():
            weights[j][:,:,i] = weight_map[i][j]

    for j in weights.keys():
        for i in range(weights[j].shape[2]):
            if method == 0:
                weights[j][:,:,i] = weights[j][:,:,i]**(f[i]*(bias[i]+i))
            elif method == 1:
                weights[j][:,:,i] = weights[j][:,:,i]*(f[i]*(bias[i]+i))
#                             if m == 0:
#                                 weights[j][:,:,i] = weights[j][:,:,i]**(smooth*(k+f*i))
#                             #elif m == 1:
#                             #    weights[j][:,:,i] = weights[j][:,:,i]**(smooth**(k+f*i))
#                             elif m == 2:
#                                 weights[j][:,:,i] = weights[j][:,:,i]*(smooth*(k+f*i))
#                             #elif m == 3:
#                             #    weights[j][:,:,i] = weights[j][:,:,i]*(smooth**(k+f*i))
    final_labels = {}
    for j in range(len(prob_paths[0])):
             
        ind = np.argmax(weights[j],axis = 2)
        final_labels[j] = np.zeros((weights[j].shape[0],weights[j].shape[1]))
        
        #reading the Labels calculated by SuperParsing 
        for i in range(len(folder_list)):
            final_labels[j][ind == i] = read_arr_from_matfile(prob_path[i] + '/' + prob_paths[i][j], 'L')[ind == i]
        #log.update()
         
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
        
        #log.update()
                

    
    #log.start("Validating",1,1)
    return val_split(path, label_true, label_num)
    #log.update()

            

    
    #print 'done'

def relabel_arr_slow(arr, big_cluster = True):
    '''
    relabels the given array in the way, 
    such that the array has #unique complete clusters (if big_cluster is True)
    '''
    arr = arr.astype(int)
    u = np.unique(arr)
    num_cluster = len(u)
    
    iterate = True
    
    m = max(u)
    new_label = m+1
    
    #a = np.array([{} for i in range(num_cluster)])
    
    a = {}
    
    for i in u:
        a[i] = {}
        
        a[i]['size'] = []
        
        a[i]['pos'] = []
        
        a[i]['border'] = []
        
        a[i]['label'] = []
    
        #a[i]['old_label'] = []
    
    #cluster = [[u[i]] for i in range(num_cluster)]
    #size_dict = {}
    #pos_dict = {}
    
    
    #size array
    
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
        
        #a[old_value]['old_label'] += [old_value]
        
        new_label += 1
        
        u = np.unique(arr)
    
    keys = np.array(a.keys())
    max_cluster = [np.argmax(a[k]['size']) for k in keys]
    
    #memorize current labeling
    values = {}
    for i in u:
        values[i] = i
        #values[u[i]]=u[i]
    
    
    b = {}
    old_labels = []
    
    #relabel biggest areas from n+1...m back to 1..n
    for i in range(len(keys)):
        #print k
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
        
        #a[k]['pos'].remove([x,y])
        #a[k]['label'].remove(old_value)
        #a[k]['size'].remove(a[k]['size'][max_cluster[i]])
     
    #relabel small areas from n+1..m to bigger neighbourhood areas 1..n
    
    keys = keys[np.argsort(np.array([b[k]['size'] for k in keys])*-1)]
    
    done = len(keys)
    
    #print arr
    for i in range(len(keys)):
        if(done == len(u)):
            continue
        
        k = keys[i]
        #print "k:{0}".format(k)
        q_unique = []
        q_points = []
        
        ub_1 = np.unique(b[k]['border'])
        p_1 = [b[k]['points'][np.where(b[k]['border']==j)[0][0]] for j in ub_1]
        
        q_unique.append(ub_1)
        q_points.append(p_1)
        
        while q_unique:
            
            ub = q_unique.pop()
            p = q_points.pop()
            #print "\t ub:{0}".format(ub)
            #print "\t p:{0}".format(p)
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
                    #print arr
                    #print "\n\n"
                    
        
        #print '==============================================================================================='
        #print arr
        
def relabel_arr(arr, big_cluster = True):
    '''
    relabels the given array in the way, 
    such that the array has #unique complete clusters (if big_cluster is True)
    '''
    arr = arr.astype(int)
    u = np.unique(arr)
    num_cluster = len(u)
    
    iterate = True
    
    m = max(u)
    new_label = m+1
    
    #a = np.array([{} for i in range(num_cluster)])
    
    a = {}
    
    for i in u:
        a[i] = {}
        
        a[i]['size'] = []
        
        a[i]['pos'] = []
        
        a[i]['border'] = []
        
        a[i]['label'] = []
    
        #a[i]['old_label'] = []
    
    #cluster = [[u[i]] for i in range(num_cluster)]
    #size_dict = {}
    #pos_dict = {}
    
    
    #size array
    
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
        
        #a[old_value]['old_label'] += [old_value]
        
        new_label += 1
        
        u = np.unique(arr)
    
    #keys = np.array(a.keys())
    keys = np.sort(np.array(a.keys()))[::-1]
    max_cluster = [np.argmax(a[k]['size']) for k in keys]
    sizes = np.array([a[keys[i]]['size'][max_cluster[i]] for i in range(len(max_cluster))])
    biggest_keys = np.argsort(sizes[::-1])
    max_cluster_values = [a[keys[i]]['label'][max_cluster[i]] for i in range(len(max_cluster))]
    #memorize current labeling
    values = {}
    for i in u:
        values[i] = i
        #values[u[i]]=u[i]
    
    
    b = {}
    old_labels = []
    
    #relabel biggest areas from n+1...m back to 1..n
    for i in range(len(keys)):
        #print k
        k = keys[i]
        b[k] = {}
        x = a[k]['pos'][max_cluster[i]][0]
        y = a[k]['pos'][max_cluster[i]][1]
        old_value = a[k]['label'][max_cluster[i]]
        new_label = k
        
        border = []
        points = []
        arr[arr==old_value] = new_label
        #flood_fill_it(x,y,arr,old_value, new_label, border, points)
        
        
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
                
                 
        
        #b[k]['points'] = np.array(points)
        
        b[k]['border'] = border
        #b[k]['points'] = np.array(points)
        b[k]['size'] = a[k]['size'][max_cluster[i]]
        values[k] = k
        values[old_value] = k
        
        #a[k]['pos'].remove([x,y])
        #a[k]['label'].remove(old_value)
        #a[k]['size'].remove(a[k]['size'][max_cluster[i]])
     
    #relabel small areas from n+1..m to bigger neighbourhood areas 1..n
    
    keys = keys[np.argsort(np.array([b[k]['size'] for k in keys])*-1)]
    
    done = len(keys)
    
    for k in keys[biggest_keys]:
        for border in b[k]['border']:
            if border > m:
                arr[arr==border] = k
    
    #print arr
#     for i in range(len(keys)):
#         if(done == len(u)):
#             continue
#         
#         k = keys[i]
#         #print "k:{0}".format(k)
#         q_unique = []
#         q_points = []
#         
#         ub_1 = np.unique(b[k]['border'])
#         p_1 = [b[k]['points'][np.where(b[k]['border']==j)[0][0]] for j in ub_1]
#         
#         q_unique.append(ub_1)
#         q_points.append(p_1)
#         
#         while q_unique:
#             
#             ub = q_unique.pop()
#             p = q_points.pop()
#             #print "\t ub:{0}".format(ub)
#             #print "\t p:{0}".format(p)
#             for l in range(len(ub)):
#                 if values[ub[l]]<=m:
#                     continue
#                 elif done == len(u):
#                     continue
#                 else:
#                     border = []
#                     points = []
#                     flood_fill_it(p[l][0],
#                                p[l][1],
#                                arr,
#                                ub[l],
#                                k,
#                                border,points)
#                     q_unique.append(np.unique(border))
#                     q_points.append([points[np.where(border==j)[0][0]] for j in q_unique[-1]])
#                     values[ub[l]] = k
#                     done += 1        
##########################################################        
        
#     max_cluster = [np.argmax(a['size'][a['old_label'] == u[i]]) for i in range(num_cluster)]
#         
#     for i in range(len('old_label')):
#         
#         
#         b = a['border']
#         
#         
#         old_value = a['label'][i]
#         x = a['pos'][i][0]
#         y = a['pos'][i][1]
#         flood_fill(x,y,arr,old_value, new_label,[])
                
   
    


def flood_fill(x, y, arr, old_value, new_value, border, points):
    '''
    implements the Floodfill algorithm
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
    implements the Floodfill algorithm
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
                #flood_fill(x-1, y, arr, old_value, new_value, border, points)
                #flood_fill(x, y-1, arr, old_value, new_value, border, points)
                #flood_fill(x+1, y, arr, old_value, new_value, border, points)
                #flood_fill(x, y+1, arr, old_value, new_value, border, points)
        
            else:
                if arr[x,y] != new_value:
                    border += [arr[x,y]]
                    points += [[x,y]]
                #return
        else:
            continue#return
        
        
     
def annotate(xml_file, arr):
    '''
    '''
    
    print 'x'
    

def draw_line(arr, x0, x1, y0, y1, color):
    '''
    draw a line into given array arr between points (x0,y0) and (x1,y1) 
    in given color
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
    '''
    import xml.etree.ElementTree as ET
    import re
    
    im_name = '/home/johann/annotate/img_0923'
    im = np.array(Image.open(im_name+'.jpg'))
    
    
    
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
    
#     for i in range(len(sem_names)):
#         print '{0} <--> {1}'.format(sem_names[i], tag_list[i][0])
    
    #tag_list = read_arr_from_txt('/home/johann/Downloads/barcelona/tags.txt', dtype=np.char)
    
    tag_dict = {}
    for i in range(len(tag_list)):
        for word in tag_list[i][1]:
            tag_dict[word]=i
            
    
    for i in range(len(sem_names)):
        semantic_label[sem_names[i]] = i+1
        
        
        
        
#     sem_list = path_to_subfolder_pathlist('/home/johann/Downloads/barcelona/LabelsSemantic', filter='.mat')
#     geo_list = path_to_subfolder_pathlist('/home/johann/Downloads/barcelona/LabelsGeo', filter='.mat')
#     
#     log = Logger()
#     log.start('converting sem to geo', len(sem_list), 1)
#     for i in range(len(sem_list)):
#         sem = read_arr_from_matfile(sem_list[i],'S')
#         geo = read_arr_from_matfile(geo_list[i],'S')
#         
#         for j in np.unique(sem):
#             if j != 0:
#                 g = geo[sem==j].flatten()
#                 geo_label[j-1][0] += len(g[g==1])
#                 geo_label[j-1][1] += len(g[g==2])
#                 geo_label[j-1][2] += len(g[g==3])
#                 #geo_label[j-1] += geo[sem==j].flatten().tolist()
#                 #geo_label[j-1] += [k for k in geo[sem==j].flatten()]
#         
#         log.update()
#         
#     
#     finale_geo_lab = [np.argmax(g)+1 for g in geo_label]
#     geo    
#         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
# , 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
# 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1,
#  1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
# , 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 3, 3, 3, 
# 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
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
        #a = read_arr_from_matfile(im_name+'.mat', 'S')
        if os.path.exists(output_path_sem + '/'+dir_name + '/' + image_name + '.mat'):
            if os.path.exists(output_path_geo + '/'+dir_name + '/' + image_name + '.mat'):
                log.update()
                continue
        
#         write_dict_to_matfile(out_file, 
#                           output_path_sem + '/'+folder + '/' + image_name + '.mat')
#         out_file['S'] = final_labels_geo
#         out_file['names'] = geo_names
#         write_dict_to_matfile(out_file, 
#                           output_path_geo + '/'+folder + '/' + image_name + '.mat')
#         
        
        b = np.zeros((a.shape[0],a.shape[1]))

        folder =''
        name = ''
        
        c = 2
        
        labs = {}
        for child in root:
            if child.tag == 'folder':
                folder = ''.join(child.text.split('\n'))
            if child.tag == 'object':
                for gchild in child:
                    if gchild.tag == 'finaltag':
                        label = ''.join(gchild.text.split('\n'))
                    
                    #print '{0}  --  {1}'.format(gchild.tag,gchild.text)    
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
                        #print points
                        d = np.zeros(b.shape)
                        for i in range(len(points)-1):
                            #draw_line(b, points[i][0], points[i+1][0], points[i][1], points[i+1][1], 1)
                            draw_line(d, points[i][0], points[i+1][0], points[i][1], points[i+1][1], 1)
                            
                        ind = np.argwhere(d==1)
                        (ystart, xstart),(ystop, xstop) = ind.min(0), ind.max(0)+1
                        mask = d[ystart:ystop,xstart:xstop]
                        it = True
                        tmp = mask.copy()
                        ind = np.argwhere(tmp!=1)
                        #print "{0},{1} - {2},{3}".format(ystart,ystop,xstart,xstop)
#                         if xstop == 630:
#                             print 'check that out'
                        skipping = False #skipping value to skip if label area are just lines
                        while it:
                            #print x
                            #print y
                            #print label
                            try:
                                x = ind[len(ind)/2][0]
                                y = ind[len(ind)/2][1]
                                flood_fill_it(x, y, tmp, 0, c, [], [])
                                #print tmp
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
                                #skipping = True
                        
                        #if skipping:
                            #continue        
                        for i in range(len(points)-1):
                            #draw_line(b, points[i][0], points[i+1][0], points[i][1], points[i+1][1], c)
                            draw_line(d, points[i][0], points[i+1][0], points[i][1], points[i+1][1], c)
                        #for point in points:
    #                        tmp[point]
                        #    flood_fill_it(point[1],point[0],tmp,1,c,[],[])
                        #flood_fill_it(points[0][1], points[0][0], tmp, 1, c, [], [])
                        
                        d[ystart:ystop,xstart:xstop][tmp == c]=c
                        
                        
    ################################################################################
    ###################
    ####    b = labeled image so far
    ####    ystart:ystop, xstart:xstop = AREA of current label
    ####    tmp = template of AREA with current label c
    ####
    ####
    ################################################################################
                        #check if another label intersects with the current label
                        #and decide which label goes in front
                        label_list = [c]
    #                     for u in np.unique(b[ystart:ystop,xstart:xstop][tmp == c]):
    #                         
    #                         if u == 0: # not labeled
    #                             continue
    #                         
    #                         elif u != c:
    #                             #tmp2 = tmp.copy()
    #                             #tmp2 = b[ystart:ystop,xstart:xstop][tmp == c]
    #                             #tmp2[tmp2 == u]
    #                             
    #                             #get color of intersection between c in tmp and u in b[AREA]
    #                             tmp2 = (b[ystart:ystop,xstart:xstop]+tmp) 
    #                             color_old = im[ystart:ystop,xstart:xstop][tmp2 == (c+u)]
    #                             
    #                             #tmp2[tmp2 == c+u] = 0
    #                             tmp3 = tmp.copy()
    #                             tmp3[tmp2 == c+u] = 0
    #                             
    #                             co = np.sum(color_old,axis = 0)*1.0/color_old.shape[0]/256
    #                             color_new = im[ystart:ystop,xstart:xstop][tmp3 == c]
    #                             
    #                             if len(color_new)==0:
    #                                 label_list += [u]
    #                                 continue
    #                              
    #                             
    #                             cn = np.sum(color_new,axis=0)*1.0/color_new.shape[0]/256    
    #                             #rgb to (y) uv 
    #                             old_y = 0.299*co[0]+0.587*co[1]+0.114*co[2]
    #                             old_u = -0.14713*co[0]-0.28886*co[1]+0.436*co[2]
    #                             old_v = 0.615*co[0]-0.51499*co[1]-0.10001*co[2]
    #                             
    #                             new_y = 0.299*cn[0]+0.587*cn[1]+0.114*cn[2]
    #                             new_u = -0.14713*cn[0]-0.28886*cn[1]+0.436*cn[2]
    #                             new_v = 0.615*cn[0]-0.51499*cn[1]-0.10001*cn[2]
    #                                
    #                             dist = (old_y-new_y)**2+(old_u-new_u)**2 + (old_v-new_v)**2
    #                             
    #                             if dist > 0.004:
    #                                 label_list += [u]
    #                             
    #                             
    #                             #color_old = im[tmp2 ==]
    #                             #tmp2[!=c]
    #                     
    #                     for u in label_list[1:]:
    #                         d[ystart:ystop,xstart:xstop][(b[ystart:ystop,xstart:xstop]+tmp)==c+u]=u
    #                             
                        b[ystart:ystop,xstart:xstop][tmp == c] = d[ystart:ystop,xstart:xstop][tmp == c]
                              
                
                labs[c] = label                
                c+=1
                
#                 if c>=0:
#                     f = pylab.figure()
#                     f.add_subplot(2,1,1)
#                     overlay = np.zeros(b.shape)
#                     overlay[b>0] = 1
#                     pylab.imshow(b)
#                     f.add_subplot(2,1,2)
#                     overlay2 = np.zeros(b.shape)
#                     overlay2[d>0] = 1
#                     pylab.imshow(overlay2)
#                     pylab.show()
                            
                            
                    
                    #if gchild == 'name':
                        
                    
                
        #[<Element 'name' at 0x7f83d2c6d790>, <Element 'deleted' at 0x7f83d2c6d7d0>, <Element 'verified' at 0x7f83d2c6d810>, <Element 'date' at 0x7f83d2c6d850>, <Element 'polygon' at 0x7f83d2c6d8d0>, <Element 'viewpoint' at 0x7f83d2c73850>, <Element 'id' at 0x7f83d2c73890>, <Element 'tag' at 0x7f83d2c738d0>, <Element 'description' at 0x7f83d2c73910>, <Element 'pointer' at 0x7f83d2c73950>, <Element 'namendx' at 0x7f83d2c73990>, <Element 'finaltag' at 0x7f83d2c739d0
    
#         f = pylab.figure()
#          
#         overlay = np.zeros(b.shape)
#         overlay[b>0] = 1
#         pylab.imshow(b)
#         pylab.show()
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
        #print np.unique(final_labels)
        write_dict_to_matfile(out_file, 
                          output_path_sem + '/'+folder + '/' + image_name + '.mat')
        out_file['S'] = final_labels_geo
        out_file['names'] = geo_names
        write_dict_to_matfile(out_file, 
                          output_path_geo + '/'+folder + '/' + image_name + '.mat')

        log.update()
  
  

def eaw_concatenate(sp_path, scaling_path, out_path):
    '''
    concatenates scaling files into one
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
        

#     
#     
#           
#           #if not os.path.isdir(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i]))):
#         #     os.makedirs(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])))
#     
#     
#     #folder_list = [f for f in os.listdir(path)]
#     folder_list = ['eaw_4','eaw_5', 'eaw_6', 'eaw_7']
#     eaw_folder = ['level_4','level_5','level_6','level_7']
#     sp_folder = ['index_4','index_5','index_6','index_7']
#     eaw_sp = [256,64,16,4]
#     folder_list = ['eaw_4','eaw_5', 'eaw_6']
#     eaw_folder = ['level_4','level_5','level_6']
#     sp_folder = ['index_4','index_5','index_6']
#     eaw_sp = [256,64,16]
# 
#     
#     log = Logger(verbose=True)
# #    log.start('Reading EAW Matrices', len(folder_list)*4, 1)
# 
#      
#     
# 
#     
#     prob_path = [path + '/' + folder_list[j] + '/Data/Base/MRF/SemanticLabels/R200K200TNN80-SPscGistCoHist-sc01ratio C00 B.5.1 S0.000 IS0.000 Pcon IPpot Seg WbS1'
#                   for j in range(len(folder_list))]
#      
#     prob_paths = [[p for p in os.listdir(prob_path[j]) if 'cache' not in p] 
#                      for j in range(len(folder_list))]
#     
#     eaw_p = [eaw_path + '/' + eaw_folder[j] for j in range(len(folder_list))]
# 
# 
#     final_labels = {}
#     k = 1
# 
# 
#     weight_map = {}
#     for i in range(len(folder_list)):
#         weight_map[i] = {}
#     log.start("Labeling EAW-Results", len(prob_paths[0]),1)
#     for j in range(len(prob_paths[0])):    
#          
#         
#          
#         for i in range(len(folder_list)):
#             sp_path = eaw_path + '/relabeled/' + sp_folder[i]
#             sp = read_arr_from_matfile(sp_path + '/' + prob_paths[i][j],'superPixels') 
#             
#             weight_map[i][j] = np.zeros(sp.shape)
#              
#             eaw = [read_arr_from_matfile(eaw_p[i] + '/' + \
#                     os.path.splitext(prob_paths[i][j])[0] + '_k_' + str(l) + \
#                     '.mat','im') for l in (np.array(range(eaw_sp[i]))+1)]
# 
#             for u in np.unique(sp):
#                 
#                 weight_map[i][j][sp == u] = \
#                     eaw[u-1][sp == u] ### Das muss ich speichern, um EAWs kleiner zu machen
#              
#             #weight_map[i][j] = weight_map[i][j]**(i+4)
#         log.update()
#     
#     weights = {}
#     #for i in range(len(folder_list)):
#     #    #weights[i] = {}
#         
#     
#     for j in range(len(prob_paths[0])):
#          
#         weights[j] = np.zeros((weight_map[0][j].shape[0],weight_map[0][j].shape[1],
#                                 len(folder_list)))
#     
#     
#     #weight_map = [] #clear memory occupied by weight_map
#     #log2 = Logger()
#     
#     method_list = ['expmult','expexp','multmult','multexp']
#     log.start('Validation', len(np.array(range(1,71,1)))*11*2,1)
#     log.start('Validation', len(sm)*len(ks)*len(fs)*len(method),1)
#     for m in method:
#         for smooth in sm: #np.array(range(1,71,1))*0.1:
#             for k in ks: #np.array([0,1,2,3,4,5,6,7,8,9,10]):
#                 for f in fs: #np.array([-1.5,-2,1.5,2]):
#                     
#                     for j in range(len(prob_paths[0])):
#              
#                         for i in weight_map.keys():
#                             weights[j][:,:,i] = weight_map[i][j]
#             
#                     for j in weights.keys():
#                         for i in range(weights[j].shape[2]):
#                             
#                             if m == 0:
#                                 weights[j][:,:,i] = weights[j][:,:,i]**(smooth*(k+f*i))
#                             #elif m == 1:
#                             #    weights[j][:,:,i] = weights[j][:,:,i]**(smooth**(k+f*i))
#                             elif m == 2:
#                                 weights[j][:,:,i] = weights[j][:,:,i]*(smooth*(k+f*i))
#                             #elif m == 3:
#                             #    weights[j][:,:,i] = weights[j][:,:,i]*(smooth**(k+f*i))
#                     for j in range(len(prob_paths[0])):
#                              
#                         ind = np.argmax(weights[j],axis = 2)
#                         final_labels[j] = np.zeros((weights[j].shape[0],weights[j].shape[1]))
#                          
#                         for i in range(len(folder_list)):
#                             final_labels[j][ind == i] = read_arr_from_matfile(prob_path[i] + '/' + prob_paths[i][j], 'L')[ind == i]
#                         #log.update()
#                          
#                     label_true = np.zeros((len(final_labels.keys()),len(object_labels[1])))
#                     label_num = np.zeros((len(final_labels.keys()),len(object_labels[1])))
#                     
#                 
#                 
#                 
#                     
#                     l_path = [path + '/' + folder_list[j] + '/SemanticLabels' 
#                               for j in range(len(folder_list))]
#                     
#                     #log.start("Generating final labels",len(final_labels.keys()),1) 
#                     for i in final_labels.keys():
#                         or_labs = read_arr_from_matfile(l_path[0] + '/' + prob_paths[0][i], 'S')
#                         u = np.unique(or_labs)
#                         for l in u:
#                             if l >0:
#                                 mask = or_labs==l
#                                  
#                                  
#                                 #correct labeled pixels
#                                 label_true[i][l-1] += len(final_labels[i][mask][(final_labels[i][mask]==l)].flatten())
#                                 #original pixel number
#                                 label_num[i][l-1] += len(mask[mask].flatten()) 
#                         
#                         #log.update()
#                                 
#                 
#                     
#                     #log.start("Validating",1,1)
#                     validate2(path, export_path, 'EAW_'+method_list[m]+'_sm_'+str(smooth)+'_k_'+str(k)+'_f_'+str(f), label_true, label_num)
#                     log.update()
# 
#             
# 
#     
#     print 'done'
        

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
