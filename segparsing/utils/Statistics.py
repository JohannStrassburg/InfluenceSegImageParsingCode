'''
Created on May 20, 2014

@author: johann strassburg

File used to create Statistics
'''

import numpy as np
from utils import Logger
from utils import path_to_subfolder_pathlist
import os.path
from os.path import basename
from utils import read_arr_from_matfile
import getopt
from utils import base_ubuntu, base_gnome, sub_experiments, \
    sub_descriptors_segment, base_ubuntu_external, base_gnome_external

import csv
from utils import path_to_file_name_list
from utils import sub_label_sets, read_dict_from_matfile,\
    read_arr_from_txt
from utils import object_labels


class SP_Statistic(object):
    '''
    class to analyze superpixels
    USAGE: use script for execution, see usage for help
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        #self.loader = loader
        self.log = Logger(verbose_count=10000)
        
    def mean_sp_num(self, loader):
        '''
        Calculates and prints the mean SuperPixel size
        @param loader: loader instance with preloaded superpixel files
        '''
        
        self.loader = loader
        sp_num = 0
        sp_min = 999999
        sp_max = 0
        self.log.start("Compute Mean SP Number", len(self.loader.sp_array_list),
                        1)
        for sp in self.loader.sp_array_list:
            num = len(np.unique(sp))
            
            if num<sp_min:
                sp_min = num
            if num>sp_max:
                sp_max = num
                
            sp_num = sp_num + num
            self.log.update()
        
        sp_total_num = sp_num
        sp_mean_num = sp_total_num*1.0/len(self.loader.sp_array_list)
        
        
        print "Total SuperPixels Number is {0}".format(sp_total_num)
        print "Mean SuperPixels Number per Image is {0}".format(sp_mean_num)
        
    
    def analyze_super_pixels(self, sp_path, safe_path, experiment):
        '''
        analyzes super pixels at given path
        @param sp_path: path to superpixels
        @param safe_path: path to store results
        @param experiment: experiment name
        '''
        
        if os.path.isfile(safe_path+"/Statistics_"+experiment +".csv"):
            print "Skipping {0} Analysis - it already exists".format(experiment)
            return
        
        self.log.start("Read SuperPixels from Experiment: {0}"\
                       .format(experiment), 1, 1)
        path_list = path_to_subfolder_pathlist(sp_path, filter=".mat")
        #self.sp_path_list = path_list
        
        sp_name_list = np.array([os.path.splitext(basename(filepath))[0] 
                                    for filepath in path_list])
        
        
        
        sp_array_list = np.array([read_arr_from_matfile(filepath,
                                                             'superPixels')
                             for filepath in path_list])
        
        
        self.log.update()
        
        self.log.start("Analyze SuperPixels from Experiment: {0}"\
                       .format(experiment), 1, 1)
        
        
        num_sp = np.array([len(np.unique(sp)) for sp in sp_array_list])
        num_sp_arg = np.argsort(num_sp)

        
        stats_sum_file =open(safe_path+'/Stats_Sum_'+experiment+'.txt','a')
        s = "##################### Statistics for Experiment "+\
            experiment + " #####################\n\n"+\
            "Max SP Num: \t{0}\n".format(num_sp[num_sp_arg[-1]])+\
            "Min SP Num: \t{0}\n".format(num_sp[num_sp_arg[0]])+\
            "Mean SP Num: \t{0}\n".format(np.sum(num_sp)*1.0/len(num_sp))
            
        stats_sum_file.write(s)
        stats_sum_file.close()
            
        
        csv.register_dialect("tab", delimiter="\t", 
                                         quoting=csv.QUOTE_ALL)
        writer = csv.DictWriter(open(safe_path+"/Statistics_"+experiment +".csv", "wb"),
                            ["Image", "#SuperPixels"], dialect = 'excel-tab')
 
        writer.writerow({"Image" : "Image", "#SuperPixels" : "#SuperPixels"})
        data = [{}]
        for i in range(len(num_sp)):
            data.append({"Image" : sp_name_list[i], "#SuperPixels" : num_sp[i]})
        
        writer.writerows(data)
        
        

        
        self.log.update()

        
    def analyze_sp_class_size(self, sp_class_path, safe_path, experiment):
        '''
        analyzes super pixels according to their size and labels
        @param sp_class_path: used label paths (e.g. /GeoLabels, /SemanticLabels
        @param safe_path: path to store statistics at
        @param experiment: experiment name
        '''
        
        
        stats = {}

        for i in range(len(sp_class_path)):
            
            
            
            stats[i] = {}
            evals = ['test', 'train', 'total']
            
            skipping = True
            for ev in evals:
                skipping = skipping & os.path.isfile(safe_path+"/SP_Stats_Size_" + \
                                         experiment +'_' + ev + ".csv")
            
            if skipping:
                print "Skipping {0} sp_size_Analysis - files already exist".format(experiment)
                return
            
            stats[i]['test'] = {}
            stats[i]['train'] = {}
            stats[i]['total'] = {}
            path_list = path_to_subfolder_pathlist(sp_class_path[i] + \
                                            '/SP_Desc_k200', filter=".mat")
            
            path_list = [path for path in path_list 
                         if (os.path.splitext(basename(path))[0] != 'segIndex')]
            
            image_name_list = np.array([os.path.splitext(basename(filepath))[0] 
                                    for filepath in path_list])
            
            test_files = read_arr_from_txt(os.path.abspath(os.path.join(
                                    sp_class_path[i], '..', 'TestSet1.txt')), 
                                           dtype = np.str)
            test_file_names = np.array([os.path.splitext(basename(test_file))[0]
                                    for test_file in test_files])
            
            test_file_indices = np.array([j for j in range(len(
                                        image_name_list)) if image_name_list[j]
                                           in test_file_names])
            
            train_file_indices = np.array([j for j in range(len(
                                        image_name_list)) if j not in 
                                            test_file_indices])
            
            
            self.log.start("Analyzing SuperPixel Size per Label in Set {0}".\
                           format(i), 
                       len(object_labels[i]) + \
                           len(test_file_indices)*len(object_labels[i]) + \
                           len(train_file_indices)*len(object_labels[i]) + \
                           len(path_list)*len(object_labels[i]), 1)
            
            for j in range(len(object_labels[i])):
                
                # (num_sp, total size, min_size, max_size, mean_size)
                stats[i]['test'][object_labels[i][j]] = np.array([0,0,9999999,0])
                stats[i]['train'][object_labels[i][j]] = np.array([0,0,9999999,0])
                stats[i]['total'][object_labels[i][j]] = np.array([0,0,9999999,0])
                self.log.update()
            
            #evaluate test files
            for ind in test_file_indices:
                for j in range(len(object_labels[i])):
                    seg_file = read_dict_from_matfile(path_list[ind])['index']
                    
                    if len(seg_file['label'][0][0])>0:
                        label = seg_file['label'][0][0][0]\
                            [seg_file['label'][0][0][0]==j+1]
                        sp_size = seg_file['spSize'][0][0][0]\
                            [seg_file['label'][0][0][0]==j+1]
                    
                        if len(label)>0:
                            stats[i]['test'][object_labels[i][j]]+= \
                                np.array([len(label),
                                      sum(sp_size),
                                      min(min(sp_size) - stats[i]['test'][object_labels[i][j]][2],0),
                                      max(max(sp_size) - stats[i]['test'][object_labels[i][j]][3],0)])
                    self.log.update()

                        
                        
            for ind in train_file_indices:
                for j in range(len(object_labels[i])):
                    seg_file = read_dict_from_matfile(path_list[ind])['index']
                    
                    if len(seg_file['label'][0][0])>0:
                        label = seg_file['label'][0][0][0]\
                            [seg_file['label'][0][0][0]==j+1]
                        sp_size = seg_file['spSize'][0][0][0]\
                            [seg_file['label'][0][0][0]==j+1]
                    
                        if len(label)>0:
                            stats[i]['train'][object_labels[i][j]]+= \
                                np.array([len(label),
                                  sum(sp_size),
                                  min(min(sp_size) - stats[i]['train'][object_labels[i][j]][2],0),
                                  max(max(sp_size) - stats[i]['train'][object_labels[i][j]][3],0)])
                    
                    self.log.update()

                        
                        
            for ind in range(len(path_list)):
                for j in range(len(object_labels[i])):
                    seg_file = read_dict_from_matfile(path_list[ind])['index']
                    
                    if len(seg_file['label'][0][0])>0:
                        label = seg_file['label'][0][0][0]\
                            [seg_file['label'][0][0][0]==j+1]
                        sp_size = seg_file['spSize'][0][0][0]\
                            [seg_file['label'][0][0][0]==j+1]
                        
                        if len(label)>0:
                            stats[i]['total'][object_labels[i][j]]+= \
                                np.array([len(label),
                                  sum(sp_size),
                                  min(min(sp_size) - \
                                      stats[i]['total'][object_labels[i][j]][2],
                                      0),
                                  max(max(sp_size)-stats[i]['total']\
                                      [object_labels[i][j]][3],0)])
                    self.log.update()    


        self.log.start("Writing SuperPixel Size Statistic ", 
                       len(evals)*(sum([len(x) for x in object_labels])), 1)
            
        for ev in evals:
            csv.register_dialect("tab", delimiter="\t", 
                                         quoting=csv.QUOTE_ALL)
            # (num_sp, total size, min_size, max_size, mean_size)
            writer = csv.DictWriter(open(safe_path+"/SP_Stats_Size_" + \
                                         experiment +'_' + ev + ".csv", "wb"),
                            ["Label", "#SuperPixels", "#Pixel", "#min_sp_size",
                             "#max_sp_size", "#mean_sp_size"], dialect = 'excel-tab')
            
            writer.writerow({"Label" : "Label", "#SuperPixels" : "#SuperPixels",
                                 "#Pixel" : "#Pixel", 
                                 "#min_sp_size" : "#min_sp_size", 
                                 "#max_sp_size" : "#max_sp_size",
                                 "#mean_sp_size" : "#mean_sp_size"})
            
            data = [{}]
            for i in range(len(sp_class_path)):
                data.append({"Label" : "###", "#SuperPixels" : "###",
                                 "#Pixel" : "###", 
                                 "#min_sp_size" : "###", 
                                 "#max_sp_size" : "###",
                                 "#mean_sp_size" : "###"})
                for lab in object_labels[i]:
                    
                    data.append({"Label" : lab, 
                                 "#SuperPixels" : stats[i][ev][lab][0],
                                 "#Pixel" : stats[i][ev][lab][1], 
                                 "#min_sp_size" : stats[i][ev][lab][2], 
                                 "#max_sp_size" : stats[i][ev][lab][3],
                                 "#mean_sp_size" : stats[i][ev][lab][1]*1.0/\
                                                    stats[i][ev][lab][0]})
                    self.log.update()
            
            writer.writerows(data)
                

    def usage(self):
        '''
        prints the usage of the script if called wrong or asked for 'help'
        '''
        print "USAGE of the Script:\n"+\
                "-h: help\n"+\
                "-e: external hard drive as source\n"+\
                "-a: all Experiments in Folder\n"+\
                "-s: analyze superPixels in size/class\n"+\
                "first parameter:\n"+\
                "\tExperiment path\n"+\
                "\t\te.g.: 'Quick_Shift_2_48_0.05\n\n"+\
                "second parameter:\n"+\
                "\tsystem of computation:\n"+\
                "\t\t'ubuntu' or 'gnome'\n"+\
                "\t\t'gnome' is taken as standard"

    def script(self,argv):
        '''
        script to execute statistical methods
        see usage for help
        '''
    
        #handling Arguments
        try:
            opts, args = getopt.getopt(argv, "h:eas", ["help"])
            
        except:
            self.usage()
            sys.exit(2)
        
        print opts    
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                self.usage()                     
                sys.exit(1)
        
        if len(args)>0:
            
            experiment = args[0]
            
            if len(args)==2:
                if args[1] == 'ubuntu':
                    system = 'ubuntu'
                else:
                    system = 'gnome'
            else:
                system = 'gnome'
        
        #Checking on which system program is running - setting base Folder
        if system == 'ubuntu':
            base = base_ubuntu
            for opt, arg in opts:
                if opt in ('-e'):
                    base = base_ubuntu_external
                
        elif system == 'gnome':
            base = base_gnome
            for opt, arg in opts:
                if opt in ('-e'):
                    base = base_gnome_external
            
        multiple = False
        sp_analysis = False
        for opt, arg in opts:
            if opt in ('-a'):
                multiple = True
            if opt in ('-s'):
                sp_analysis = True
                
        
                
        print base
        #setting path where super_pixels are in
        if multiple:
            path_list = path_to_file_name_list(base+sub_experiments, "")
            
            log = Logger(verbose=True)
            log.start("Multiple Superpixels Statistic calculation", 
                      len(path_list), 1)
            for experiment in path_list:
                try:
                    sp_path = base + sub_experiments + experiment +\
                    sub_descriptors_segment + "/super_pixels"
                    
                    if not os.path.exists(sp_path):
                        print "Folder {0} doesn't have SP Data".format(experiment)
                        log.update()
                        continue
                    
                    safe_path = base + "Statistics"
                    if not os.path.exists(safe_path):
                        os.makedirs(safe_path)
             
                    self.analyze_super_pixels(sp_path,
                                    safe_path, experiment)
                    
                    if sp_analysis:
                        sp_class_path = (base + sub_experiments + experiment +\
                                         sub_label_sets[0],
                                         base + sub_experiments + experiment +\
                                         sub_label_sets[1])
                
                
                        self.analyze_sp_class_size(sp_class_path, safe_path, 
                                                   experiment)
                except:
                    print "Error: " + experiment + " not a folder."
                    raise 
                
                log.update("Calculated: "+experiment)
        else:
            sp_path = base + sub_experiments + experiment +\
                sub_descriptors_segment + "/super_pixels"
                
            safe_path = base + "Statistics"
            if not os.path.exists(safe_path):
                os.makedirs(safe_path)
            
            self.analyze_super_pixels(sp_path,
                                      safe_path, experiment) 
            
            if sp_analysis:
                sp_class_path = (base + sub_experiments + experiment +\
                    sub_label_sets[0],
                                base + sub_experiments + experiment +\
                    sub_label_sets[1])
                
                
                self.analyze_sp_class_size(sp_class_path, safe_path, experiment)       
        


if __name__ == '__main__':
    '''
    '''
    
    import sys
    
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    
    sp_statistic = SP_Statistic()
    
    #sp_statistic.script(['-a','-e','-s','none','ubuntu'])
    sp_statistic.script(sys.argv[1:])
    #super_parsing(sys.argv[1:])
