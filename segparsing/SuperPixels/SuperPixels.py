'''
Created on May 19, 2014

@author: johann strassburg
SuperPixels.py: implements SPCalculator to calculate superpixels
'''

import vlfeat
from utils.utils import path_to_subfolder_pathlist, write_arr_to_matfile, Logger,\
    read_arr_from_matfile
import os
import numpy as np
import threading
from skimage import data, io, filter, segmentation
from os.path import basename
from Queue import Queue as queue
from scipy.cluster.vq import kmeans, vq
import Image
from multiprocessing import Process, Pool
from multiprocessing import Queue as MP_Queue
import gc

class SPCalculator(object):
    '''
    SPCalculator calculates SuperPixels for given images using a specific 
    SuperPixel Method
    '''


    def __init__(self, loader):
        '''
        @param loader:  Loader instance, used for image retrieval 
                        and further parameters
        '''
        
        self.loader = loader
        
        self.log = Logger(verbose=True)
        
        self.image_array_list = self.loader.image_array_list
        
        self.image_list = self.loader.image_list
        
        self.image_path_list = self.loader.image_path_list
        
        self.sp_thread = np.zeros((len(self.image_list),256,256))
        
        self.sp_thread_lock = threading.Lock()
        
        self.max_thread = threading.Semaphore(1)
        
        self.log_lock = threading.Lock()
        
        self.seq_reading = self.loader.seq_reading
        
        self.sp_array = np.array([])
        self.sp_labeled = np.array([])
        self.sp_labels = []
        
        
    
    def calculate_sp(self, seg_method, seg_folder):
        '''
        calculates SuperPixels using the given segmentation method and params
        @param seg_method: segmentation method to use including its parameters
        @param seg_folder: folder for superpixels
        '''
        
        if seg_method['method'] == "Quick_Shift":
            self.quick_shift(seg_folder, seg_method['ratio'], seg_method['kernelsize'], 
                             seg_method['maxdist'])
        elif seg_method['method'] == "SLIC":
            self.slic(seg_folder, seg_method['region_size'], 
                      seg_method['regularizer'])
        
        elif seg_method['method'] == "Ground_Truth":
            self.ground_truth(seg_folder, seg_method['label_folder'])
            
        elif seg_method['method'] == "Saliency":
            self.saliency(seg_folder, seg_method['saliency_folder'], 
                          seg_method['k'])
        
        elif seg_method['method'] == "GRID":
            self.grid(seg_folder, seg_method['k'])
            
        
    
    def quick_shift(self, seg_folder, ratio = 0.5, kernelsize = 2, maxdist = 200):
        '''
        calculates SuperPixels using QuickShift
        @param seg_folder: folder for superpixels
        @param ratio: ratio of importance between color and spatial space
        @param kernelsize: Kernelsize for Quick Shift method
        @param maxdist: set maximum distance to influence size of superpixels
        '''
        
        self.log.start("Calculating Quick Shift SuperPixels", 
                       len(self.image_list), 1)
                

        self.sp_thread_lock.acquire()
        for i in range(len(self.image_list)):
            
            if not os.path.isdir(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i]))):
                os.makedirs(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])))
            
            self.max_thread.acquire()
            try:
                
                self._qs_thread(i, seg_folder, ratio, kernelsize, maxdist)
                
            except:
                print "Upps, Fehler bei {0}".format(i)
            
        
            
        self.sp_thread_lock.acquire()
        self.sp_thread_lock.release()

            
    def _qs_thread(self,i, seg_folder, ratio, kernelsize, maxdist):
        '''
        quick shift thread
        '''

        if not os.path.isfile(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])) + '/' + self.image_list[i] +".mat"):
            
            if self.seq_reading:
                image = np.array(Image.open(self.image_path_list[i]))   
            else:  
                image = self.image_array_list[i]
                

            if len(image.shape) == 2: #check if a grayscale image is given
                im = np.zeros((image.shape[0],image.shape[1],3))
                for k in [0,1,2]:
                    im[:,:,k]=image
                image = im
            iseg = vlfeat.vl_quickseg(image, ratio, kernelsize, maxdist)

            labels = iseg[1]
            if min(np.unique(labels)) == 0:
                labels = labels + 1

            write_arr_to_matfile(labels,
                            seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])) + '/' + self.image_list[i] +".mat", 
                            "superPixels")
            gc.collect()

        
        self.max_thread.release()
        self.log_lock.acquire()
        try:
            self.log.update()
            if self.log.process_percentage>=100 :
                try:
                    self.sp_thread_lock.release()
                except:
                    print 'here'

        finally:
            self.log_lock.release()
            


    def slic(self, seg_folder, region_size=10, regularizer=0.01):
        '''
        implements the vl_slic segmentation using threads
        @param seg_folder: superpixel folder
        @param region_size: size of a superpixel
        @param regularizer: factor to influence color/space for segments creation
        '''
        
        self.log.start("Calculating SLIC SuperPixels", 
                       len(self.image_list), 1)
                
        
        self.sp_thread_lock.acquire()
        for i in range(len(self.image_path_list)):
            
            if not os.path.isdir(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i]))):
                os.makedirs(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])))

            self.max_thread.acquire()
            try:
                t = threading.Thread(target=self._slic_thread, 
                                     args = (i, seg_folder, region_size, 
                                             regularizer,))
                t.start()
            except:
                print "Upps, Fehler bei {0}".format(i)
            
        
        self.sp_thread_lock.acquire()
        self.sp_thread_lock.release()
        
    
    def _slic_thread(self,i, seg_folder, region_size, regularizer):
        '''
        thread for the vl_slic segmentation
        '''
        

        if not os.path.isfile(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])) +\
             os.path.basename(os.path.splitext(self.image_path_list[i])[0]) +".mat"):
            
            if self.seq_reading:
                image = np.array(Image.open(self.image_path_list[i]))   
            else:  
                image = self.image_array_list[i]
            
            if len(image.shape) == 2: #check if a grayscale image is given
                im = np.zeros((image.shape[0],image.shape[1],3))
                for k in [0,1,2]:
                    im[:,:,k]=image
                image = im
            labels = segmentation.slic(image, region_size, regularizer)
            
            unique = np.unique(labels)
            if min(unique) == 0:
                labels = labels + 1
                unique = np.unique(labels)
                
            if unique[-1]>len(unique):
                lab = labels.copy()
                for j in range(len(unique)):
                    
                    labels[lab==unique[j]] = j+1
                
                    
            write_arr_to_matfile(labels,
                            seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])) + '/' + self.image_list[i] +".mat", 
                            "superPixels")
        
        self.max_thread.release()
        self.log_lock.acquire()
        try:
            self.log.update()
            if self.log.process_percentage>=100 :
                try:
                    self.sp_thread_lock.release()
                except:
                    print 'here'

        finally:
            self.log_lock.release()
    
        
        
    def ground_truth(self, seg_folder, label_folder):
        '''
        calculates GroundTruth Superpixels
        @param seg_folder: superpixel folder
        '''
        
        self.log.start("Calculating GroundTruth SuperPixels", 
                       len(self.image_list), 1)
        self.log.verbose = True
        
        path_list = path_to_subfolder_pathlist(label_folder, filter=".mat")
        
        for i in range(len(path_list)):
            self.label_sp(i, seg_folder, path_list)
            
            write_arr_to_matfile(self.sp_labeled,
                            seg_folder + "/" + \
                            os.path.splitext(basename(path_list[i]))[0] +".mat", 
                            "superPixels")
            self.log.update()
            
            
    def grid(self, seg_folder, k):
        '''
        calculates grid-wise segmentation
        @param seg_folder: superpixel folder
        @param k: segmentation factor, constructs a k x k grid
        '''
        
        self.log.start("Calculating GRID SuperPixels", 
                       len(self.image_list), 1)
                

        for i in range(len(self.image_list)):
            
            if not os.path.isdir(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i]))):
                os.makedirs(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])))

            
            if not os.path.isfile(seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])) +\
             os.path.basename(os.path.splitext(self.image_path_list[i])[0]) +".mat"):
            
                if self.seq_reading:
                    image = np.array(Image.open(self.image_path_list[i]))
                else:
                    image = self.image_array_list[i]
                y_max = image.shape[1]
                x_max = image.shape[0]
                
                y_step = y_max/k
                x_step = x_max/k
                
                y_rest = y_max%k
                x_rest = x_max%k
                
                
                
                if x_rest>= 0.75*x_step:
                    x_range = range(0,x_max,x_step)
                else:
                    x_steps = [x_step for j in range(k-1)]
                    for j in range(x_rest):
                        x_steps[j%len(x_steps)] =\
                            x_steps[j%len(x_steps)]+1
                    x_range = [0]+x_steps
                    for j in range(len(x_range)-1):
                        x_range[j+1] = x_range[j]+x_range[j+1]
                        
                    
                if  y_rest >= 0.75*y_step:
                    y_range = range(0,y_max,y_step)
                else:
                    y_steps = [y_step for j in range(k-1)]
                    for j in range(y_rest):
                        y_steps[j%len(y_steps)] =\
                            y_steps[j%len(y_steps)]+1
                    y_range = [0]+y_steps
                    for j in range(len(y_range)-1):
                        y_range[j+1] = y_range[j]+y_range[j+1]


                x_steps = [(a,min(a+x_step+1,x_max)) for a in x_range]
                y_steps = [(a,min(a+y_step+1,y_max)) for a in y_range]
                
                super_pixels = np.zeros((image.shape[0],
                                         image.shape[1]))
                
                label = 1
                
                for x in x_steps:
                    for y in y_steps:
                        super_pixels[x[0]:x[1],y[0]:y[1]] = label
                        label +=1
                
                write_arr_to_matfile(super_pixels,
                                seg_folder + "/" + os.path.basename(os.path.dirname(self.image_path_list[i])) + '/' + self.image_list[i] +".mat", 
                                "superPixels")
            self.log.update()
            
                    
            


    def saliency(self, seg_folder, saliency_folder, k):
        '''
        clusters and relabels saliency maps of images
        @param seg_folder: superpixel folder
        @param saliency_folder: folder to salience-images
        '''
        
        self.log.start("Recomputing Saliency SuperPixels", 
                       len(self.image_array_list), 1)
        self.log.verbose = True
        
        path_list = path_to_subfolder_pathlist(saliency_folder, filter=".mat")
        
        for i in range(len(path_list)):
            
            
            self.sp_array = read_arr_from_matfile(path_list[i], "S")
            self.sp_labeled = np.zeros(self.sp_array.shape) -1
            
            pixel = np.reshape(self.sp_array, (self.sp_array.shape[0]*\
                                               self.sp_array.shape[1]))
            
            #clustering
            centroids,_ = kmeans(pixel, 60)
            # quantization
            qnt,_ = vq(pixel, centroids)
            
            centers_idx = np.reshape(qnt, (self.sp_array.shape[0],
                                           self.sp_array.shape[1]))
            self.sp_array = centroids[centers_idx]
            
            
            self.label_ffw()
            
            write_arr_to_matfile(self.sp_labeled,
                            seg_folder + "/" + \
                            os.path.splitext(basename(path_list[i]))[0] +".mat", 
                            "superPixels")
            self.log.update()
        
    
    def _gt_thread(self,i, seg_folder, path_list):
        '''
        Ground Truth Thread
        @param i: index to current image
        @param seg_folder: superpixel folder
        @param path_list: path to images 
        '''
        
        if not os.path.isfile(seg_folder + "/" + \
                        os.path.splitext(basename(path_list[i]))[0] +".mat"):
            
            
            segments = read_arr_from_matfile(path_list[i], "S")

            segments_new = segments.copy()
            labels = np.unique(segments)
            label_new = 1
            for l in labels:
                segment = (segments == l)*1
                segment_arg = np.argwhere(segments == l)

                
                
                mask = np.array([[1,1,1,1,1],
                              [1,0,1,0,1],
                              [1,1,1,1,1],
                              [1,0,1,0,1],
                              [1,1,1,1,1]])
                
                cluster = {}
                
                points = range(len(segment_arg))
                c = 0
                
                for p in range(len(segment_arg)):
                    q = queue()
                    if p in points:
                        points.remove(p)
                        cluster[c] = [p]
                        q.put(p)
                        
                        mod_arr = segments_new == -5
                        while(not q.empty()):
                            
                            i = q.get()
                            
                            seg = segment_arg[i]
                            row_up = max(0,seg[0]-1)
                            row_down = min(segment.shape[0],seg[0]+2)
                            col_left = max(0,seg[1]-1)
                            col_right = min(segment.shape[1],seg[1]+2)
                            
                            
                            if seg[0]-1<0:
                                mask_up = 0
                                mask_down = 2
                            elif seg[0]+1 >= segment.shape[0]:
                                mask_up = 3
                                mask_down = 5
                            else:
                                mask_up = 1
                                mask_down = 4
                                
                            if seg[1]-1<0:
                                mask_left = 0
                                mask_right = 2
                            elif seg[1]+1 >= segment.shape[1]:
                                mask_left = 3
                                mask_right = 5
                            else:
                                mask_left = 1
                                mask_right = 4
                            
                            
                            segment[row_up:row_down, col_left:col_right] += \
                                mask[mask_up:mask_down,mask_left:mask_right]
                                
                            [(cluster[c].append(j),q.put(j), points.remove(j))  
                                for j in points
                                if segment[segment_arg[j][0],
                                           segment_arg[j][1]]>1]


                            segments_new[segment>1] = label_new
                            segment[row_up:row_down, col_left:col_right] -= \
                                mask[mask_up:mask_down,mask_left:mask_right]
                        

                        label_new += 1
                        
                        c +=1
                
                
                
                print l
            write_arr_to_matfile(segments_new,
                            seg_folder + "/" + \
                            os.path.splitext(basename(path_list[i]))[0] +".mat", 
                            "superPixels")



    def label_sp(self, i, seg_folder, path_list):
        '''
        labels superpixel image from path_list[i] uniquely
        '''
        
        self.sp_array = read_arr_from_matfile(path_list[i], "S")
        self.sp_labeled = np.zeros(self.sp_array.shape) -1
        self.label_ffw()
        
        
    
    def label_ffw(self):
        '''
        relabels (quantized) superpixel-image(self.sp_array) to labels from 1 to n
        with one segment/label (self.sp_labeled (initialized as -1)) 
        '''
        current_label = 1
        for i in range(self.sp_array.shape[0]):
            for j in range(self.sp_array.shape[1]):
                #### first Row ####
                if i == 0:
                    if j == 0:
                        self.sp_labels.append(current_label)
                        self.sp_labeled[i,j] = current_label
                    else:
                        if self.sp_array[i,j-1] == self.sp_array[i,j]:
                            self.sp_labeled[i,j] = self.sp_labeled[i,j-1]
                        else:
                            current_label += 1
                            self.sp_labels.append(current_label)
                            self.sp_labeled[i,j] = current_label
                #### else ####
                else:
                    if j == 0:
                        if self.sp_array[i-1,j] == self.sp_array[i,j]:
                            self.sp_labeled[i,j] = self.sp_labeled[i-1,j]
                        else:
                            current_label += 1
                            self.sp_labels.append(current_label)
                            self.sp_labeled[i,j] = current_label
                    else:
                        if self.sp_array[i,j-1] == self.sp_array[i,j]:
                            self.sp_labeled[i,j] = self.sp_labeled[i,j-1]
                            # relabel necessary
                            if self.sp_array[i-1,j] == self.sp_array[i,j]:
                                if self.sp_labeled[i-1,j] != self.sp_labeled[i,j]:
                                    label1 = self.sp_labeled[i-1,j]
                                    label2 = self.sp_labeled[i,j]
                                    if label1<label2:
                                        self.sp_labeled[i,j] = label2
                                        self.relabel(i,j,label2,label1)
                                    else:
                                        self.sp_labeled[i,j] = label1
                                        self.relabel(i,j,label1,label2)
                                    
                        elif self.sp_array[i-1,j] == self.sp_array[i,j]:
                            self.sp_labeled[i,j] = self.sp_labeled[i-1,j]
                        
                        else:
                            current_label += 1
                            self.sp_labels.append(current_label)
                            self.sp_labeled[i,j] = current_label
                            
        unique = np.unique(self.sp_labeled)
        relabel = {}
        for i in range(len(unique)):
            relabel[unique[i]]=i+1
        
        sp_labels = self.sp_labeled.copy()
        for i in relabel.keys():
            sp_labels[self.sp_labeled==i]=relabel[i]
        
        self.sp_labeled = sp_labels
                            
        
    def relabel(self, i, j, old_label, new_label):
        '''
        recursive relabeling of self.sp_labeled from old_label to new_label
        starting at position i,j
        '''
        
        if (i<0) | (j<0) | (i==self.sp_labeled.shape[0]) | \
            (j == self.sp_labeled.shape[1]):
            return
        elif self.sp_labeled[i,j] == new_label:
            return          
        if self.sp_labeled[i,j] == old_label:
            self.sp_labeled[i,j] = new_label
            self.relabel(i-1, j, old_label, new_label)
            self.relabel(i, j-1, old_label, new_label)
            self.relabel(i, j+1, old_label, new_label)
            
                
        
        