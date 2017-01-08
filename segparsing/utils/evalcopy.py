import os.path
from os.path import basename
import shutil

def copyResults(path):
    '''
    copies ResultMRF from given experiments folder (path) to 
    path, '..', 'ResultsMRF'
    @param path: path to experiments folder
    '''
    
    folder_list = [f for f in os.listdir(path)]
    
    for folder in folder_list:
        if folder != 'copy_me_to_experiment':
            try:
                file_path = path + '/' + folder + '/Data/Base/ResultsMRF.txt'
                target_path = os.path.abspath(os.path.join(path, '..', 'ResultsMRF')) +\
                    '/'+folder+'_ResultsMRF.txt'
                print "Copying Results of " + folder + ' to ' + target_path
                shutil.copyfile(file_path, target_path)
            except:
                print "Skipping {0}".format(folder)