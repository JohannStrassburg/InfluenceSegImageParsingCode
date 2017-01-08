import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import numpy as np
import Image
import utils
import matplotlib.pyplot as plt
import os.path
from os.path import basename
from os.path import dirname

def run(path,segments):
    '''
    reads saliency files and converts them into heatmap images to location path/output/
    @param path: path to saliency main folder
    @param segments: folder name to saliency files
    '''
    
    #fh = utils.path_to_subfolder_pathlist("super_pixels",filter=".mat")
    spatial = utils.path_to_subfolder_pathlist(path + '/' + segments,filter=".mat")
    #f = [utils.read_arr_from_matfile(fh[i], "superPixels") for i in range(len(fh))]
    #s = [utils.read_arr_from_matfile(spatial[i],"S") for i in range(len(spatial))]
    if not os.path.exists('output'):
        os.makedirs('output')
    extent = [0,255,0,255]
    for i in range(len(spatial)):
        mat = utils.read_arr_from_matfile(spatial[i],"S")
        if os.path.exists(path + '/' + 'output/'+ basename(dirname(spatial[i])) + '/' + os.path.splitext(basename(spatial[i]))[0]+'.jpg'):
            im = np.array(Image.open(path + '/' + 'output/'+ basename(dirname(spatial[i])) + '/' + os.path.splitext(basename(spatial[i]))[0]+'.jpg'))
            (x,y,_) = im.shape
            (x1,y1) = mat.shape 
            if (x == x1) & (y == y1):
                    

                continue
        
        b = np.zeros(mat.shape)        

        u = np.unique(mat)        
        for j in range(len(u)):
            b[mat==u[j]] = j

        fig = plt.figure(frameon=False)
        
        fig.set_size_inches((mat.shape[1]+0.5)*1.0/100,(mat.shape[0]+0.5)*1.0/100)        
        #fig.set_size_inches(2.56,2.56)
        ax = plt.Axes(fig, [0.,0.,1.,1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(b, aspect='normal')
        folder = basename(dirname(spatial[i]))
        if not os.path.exists(path + '/' + 'output2/'+folder):
            os.makedirs(path + '/' + 'output2/'+folder)
        fig.savefig(path + '/' + 'output2/'+ folder + '/' + os.path.splitext(basename(spatial[i]))[0]+'.jpg')
        im2 = np.array(Image.open(path + '/' + 'output2/'+ basename(dirname(spatial[i])) + '/' + os.path.splitext(basename(spatial[i]))[0]+'.jpg'))
        (x2,y2,_) = im2.shape
        print '({0},{1} -- {2})'.format(x1-x2,y1-y2,basename(spatial[i]))
        plt.close()
        #print "({0},{1}) -- {2}".format(y1,x1,im2.shape)
        #print "{0}/{1}".format(i+1,len(spatial))
        #plt.savefig('output/'+ os.path.splitext(basename(spatial[i]))[0]+'.jpg')
    print 'done'

if __name__ == '__main__':
        '''
        '''
        path = '/media/johann/Patrec_external5/SuperParsing/Barcelona/saliency_image_conversion'
        segments = 'segments'
        run(path, segments)
