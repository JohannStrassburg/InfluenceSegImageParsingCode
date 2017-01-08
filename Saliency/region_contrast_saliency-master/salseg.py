import numpy as np
import Image
import utils
import matplotlib.pyplot as plt
import os.path
from os.path import basename

def segment(in_folder, output_folder):
	'''
	takes salience segments from in_folder and 
	saves black and white images to output_folder
	'''
	
	#fh = utils.path_to_subfolder_pathlist("super_pixels",filter=".mat")
	sal_path_list = utils.path_to_subfolder_pathlist(in_folder,filter=".mat")
	#f = [utils.read_arr_from_matfile(fh[i], "superPixels") for i in range(len(fh))]
	s = [utils.read_arr_from_matfile(sal_path_list[i],"S") 
		for i in range(len(sal_path_list))]
	
	#extent = [0,255,0,255]
	for i in range(len(s)):
		
		b = np.ones([s[i].shape[0],s[i].shape[1],3], dtype=np.uint8)
		#b = np.zeros(s[i].shape)
		#u = np.unique(s[i])
		#for j in range(len(u)):
		#	b[s[i]==u[j]] = j
		#print 'shape is {0}'.format(b.shape)
		print "{0}/{1}".format(i+1,len(s))
		b = b*255
		b[:,:,0] = b[:,:,0]*s[i]
		b[:,:,1] = b[:,:,1]*s[i]
		b[:,:,2] = b[:,:,2]*s[i]
		fig = plt.figure(frameon=False)
		fig.set_size_inches((mat.shape[1]+0.5)*1.0/100,(mat.shape[0]+0.5)*1.0/100)        
		ax = plt.Axes(fig, [0.,0.,1.,1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		ax.imshow(b, aspect='normal')
		fig.savefig(output_folder + '/'+ os.path.splitext(basename(sal_path_list[i]))[0]+'.jpg')
		print "{0}/{1}".format(i+1,len(s))
		#plt.savefig('output/'+ os.path.splitext(basename(spatial[i]))[0]+'.jpg')
