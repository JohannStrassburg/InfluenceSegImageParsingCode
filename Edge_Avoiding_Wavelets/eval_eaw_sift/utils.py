'''
Created on 25 Apr 2014

@author: johann
'''

import numpy as np
from utils import Logger, read_arr_from_matfile, object_labels_barcelona, path_to_subfolder_pathlist, read_dict_from_matfile
import os



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

