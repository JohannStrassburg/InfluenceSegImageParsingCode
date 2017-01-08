'''
Created on Sep 21, 2014

@author: johann strassburg
'''

from utils import path_to_subfolder_pathlist, read_arr_from_matfile


def compare(path1, path2):
    '''
    compare shapes of superpixel-files at path1 with label-files at path2
    @param path1: path to superpixel-files
    @param path2: path to label files
    '''
    
    file_path1 = path_to_subfolder_pathlist(path1, '.mat')
    file_path2 = path_to_subfolder_pathlist(path2, '.mat')

    count = 0
    for i in range(len(file_path1)):
        
        file1 = read_arr_from_matfile(file_path1[i], 'superPixels')
        file2 = read_arr_from_matfile(file_path2[i], 'S')
        
        
        (a,b) = file1.shape
        (c,d) = file2.shape
        if (a != c) | (b != d):
            print '({0},{1}) -- ({2},{3})'.format(a,b,c,d)
            count +=1
            print file_path1[i]
            
    
    print str(count)
        

if __name__ == '__main__':
    
    path1 = '/media/johann/Patrec_external5/SuperParsing/Barcelona/Experiments/sal_ldrccb_prod/Data/Descriptors/SP_Desc_k200/super_pixels'
    path2 = '/media/johann/Patrec_external5/SuperParsing/Barcelona/SemanticLabels'
    
    compare(path1,path2)
    pass