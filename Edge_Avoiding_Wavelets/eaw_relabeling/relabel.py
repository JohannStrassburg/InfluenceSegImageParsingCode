import utils
from utils import relabel_arr
from os.path import basename
import os.path

def relabel(global_path):
    '''
    relabels EAW superpixels in the way, 
    such that the array has #unique complete clusters
    @param global_path: path to EAW superpixels
    '''
    
    path_list = utils.path_to_subfolder_pathlist(global_path, filter=".mat")
    
    relabeled_num = 0
    print path_list
    log = utils.Logger(verbose = True)
    log.start("Relabeling EAW_SuperPixels",len(path_list), 1)
    for path in path_list:
        #print "relabeling {0}".format(path)
        folder = 'none'
        if (True):        
            folder = basename(os.path.abspath(os.path.join(path, '..','..')))
            #im_folder = basename(os.path.abspath(os.path.join(path, '..')))
            target_folder = os.path.abspath(os.path.join(path, '..', '..','..','relabeled', folder, basename(os.path.abspath(os.path.join(path, '..')))))
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)
            #print "Target Folder: {0}".format(target_folder)
            target_path = os.path.abspath(os.path.join(path, '..', '..','..','relabeled', folder, target_folder, basename(path)))
            if os.path.isfile(target_path):
                log.update()
                continue
            #print "Target Path: {0}".format(target_path)            
            arr = utils.read_arr_from_matfile(path,"ind")
                            
            #print "Relabeling ... {0}".format(basename(path)) 
            
            relabel_arr(arr)
            
            
            #print "Relabeled ... {0}".format(basename(path)) 
                        
            utils.write_arr_to_matfile(arr, target_path, "superPixels")
            print "Wrote ... {0}".format(basename(path))                    
        else:        
            print 'Failure in {0}'.format(folder)
            print 'Failure in image {0}'.format(basename(target_path))
            raise    
            if folder in ['index_4','index_5','index_6','index_7','index_8']:
                print 'Failure in {0}'.format(folder)
                print 'Failure in image {0}'.format(basename(target_path))
                raise
            
            
        
        log.update()
    print "Relabeled files: {0}/{1}".format(relabeled_num,len(path_list))