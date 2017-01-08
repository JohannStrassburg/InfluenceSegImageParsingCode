'''
Created on 16.04.2014

@author: Johann Strassburg
Reimplementation of Lazebnik et al.'s MATLAB Code to SuperParsing to calculate segment features
Recommendation: Use original code instead
rounding errors are marked
'#APPROVED' marks methods which were tested to create same results as the given MATLAB code for test data
'''
import numpy as np
import Image
import leargist
from utils.utils import Logger


class segmentFeatureExtractor(object):
    '''
    extracts features for Segments
    '''
    

    def __init__(self):
        '''
        Constructor
        '''
        
        self.feature_list = ['sift_hist_int_', 'sift_hist_left', 
                             'sift_hist_right', 'sift_hist_top', 'top_height',
                             'absolute_mask', 'bb_extent',
                             'centered_mask_sp', 'color_hist',
                             'color_std', 'color_thumb', 'color_thumb_mask',
                             'dial_color_hist', 'dial_text_hist_mr',
                             'gist_int', 'int_text_hist_mr', 
                             'mean_color', 'pixel_area',
                             'sift_hist_bottom', 'sift_hist_dial']
        
        
    
    
    
    
    #===========================================================================
    # #APPROVED
    #===========================================================================
    def calculate_texton_hist(self, textons, dictionarySize ):
        '''
        description needs to be inserted
        '''
        if textons.any:
            count = np.bincount(textons)
            count = count*1.0/len(textons)
            count = count[1:]
            if dictionarySize>len(count):
                descHist = np.concatenate([count,np.zeros(dictionarySize-len(count))])
            else:
                descHist = count
            #descHist = hist(textons.flatten(1), 1:dictionarySize)./length(textons)
        else:
            descHist = np.zeros(dictionarySize,1) 
        
        return descHist
    
    #===========================================================================
    # #APPROVED
    #===========================================================================
    def sift_hist_int(self,mask_crop, centers, textons):
        '''
        description needs to be inserted
        1x100 double
        '''
        mask = mask_crop
        textonIm = textons['sift_textons'].flatten(1)#textonIm = textons.sift_textons[:]
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv]  
        #textonIm = np.array([x for x in (textonIm*(mask==0)).flat if x>0])
        dictionarySize = centers['sift_centers'].shape[0]#dictionarySize = size(centers.sift_centers,1)

        
        #desc = calculate_texton_hist( textonIm, dictionarySize )
        desc = self.calculate_texton_hist( textonIm, dictionarySize )

        return desc
    
    #===========================================================================
    # APPROVED
    #===========================================================================
    def sift_hist_left(self, centers, textons, borders):
        '''
        description needs to be inserted
        1x100 double
        '''
        mask=borders[:,:,0]#mask=borders[:,:,1]
        mask = mask.flatten(1)#mask = mask[:]
        textonIm = textons['sift_textons'].flatten(1)#textonIm = textons.sift_textons[:]
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv] 
   
        dictionarySize = centers['sift_centers'].shape[0]#dictionarySize = size(centers.sift_centers,1)
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        ###desc = calculate_texton_hist( textonIm, dictionarySize )
        
        return desc
    #==============================================================================
    # APPROVED
    #==============================================================================
    def sift_hist_right(self, centers, textons, borders):
        '''
        description needs to be inserted
        1x100 double
        '''
        mask=borders[:,:,1]#mask=borders[:,:,2]
        mask = mask.flatten(1)#mask = mask[:]
        textonIm = textons['sift_textons'].flatten(1)#textonIm = textons.sift_textons[:]
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv] 
          
        dictionarySize = centers['sift_centers'].shape[0]#dictionarySize = size(centers.sift_centers,1)
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        ####desc = calculate_texton_hist( textonIm, dictionarySize )  
        
        return desc
    #===========================================================================
    # APPROVED
    #===========================================================================
    def sift_hist_top(self, centers, textons, borders):
        '''
        description needs to be inserted
        1x100 double
        '''
  
        mask = borders[:,:,2]#mask= borders[:,:,3]
        mask = mask.flatten(1)#mask = mask[:]
        textonIm = textons['sift_textons'].flatten(1)#textonIm = textons.sift_textons[:]
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv] 
          
        dictionarySize = centers['sift_centers'].shape[0]#dictionarySize = size(centers.sift_centers,1)
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        ####desc = calculate_texton_hist( textonIm, dictionarySize )
 
        return desc
    
    #===========================================================================
    # #APPROVED #IndexCountError!!!
    #===========================================================================
    def top_height(self, mask, mask_crop, bb):
        '''
        description needs to be inserted
        1x1 double
        '''
 
        #[y,x] = find(mask_crop>0)
        (y,_) = mask_crop.nonzero()
        #desc = min(y+bb[1])/size(mask,1)
        desc = min(y+bb[0]+1)*1.0/mask.shape[0]
        
        return desc
    
    #======================================================================
    # APPROVED
    #======================================================================
    def top_text_hist_mr(self, centers, textons, borders):
        '''
        description needs to be inserted
        '''
         
        
        mask= borders[:,:,2]#mask= borders[:,:,3]
        
        mask = mask.flatten(1)#mask = mask(:)
        textonIm = textons['mr_filter'].flatten(1)
        ####textonIm = textons.mr_filter[:]
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv] 
         
        dictionarySize = centers['mr_resp_centers'].shape[0]
        ####dictionarySize = size(centers.mr_resp_centers,1)
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        ####desc = calculate_texton_hist( textonIm, dictionarySize )

        return desc
    
    #===========================================================================
    # APPROVED IMSIZE-ROUND-FAILURE
    #===========================================================================
    def absolute_mask(self, im, mask):

        '''
        description needs to be inserted
        return 8x8 matrix
        '''
        mask = mask*1.0
        I = Image.fromarray(mask)
        I = I.resize((8,8), Image.ANTIALIAS)
        im = np.array(I)
        desc = np.maximum(im,np.zeros(im.shape))
        #desc = np.max(scipy.misc.imresize(mask,(8,8)))  # @UndefinedVariable
        #desc = max(imresize(double(mask),[8 8]),0)
        return desc
         
    
    #===========================================================================
    # APPROVED   
    #===========================================================================
    def bb_extent(self, mask, mask_crop, bb):
        '''
        description needs to be inserted
        return 1x2 double matrix
        '''
        #[y x] = find(mask_crop) 
        (y,x) = mask_crop.nonzero()
        #desc = [(max(y+bb(1))-min(y+bb(1)))/size(mask,1) (max(x+bb(3))-min(x+bb(3)))/size(mask,2)]
        desc = np.array([(max(y+bb[0])-min(y+bb[0]))*1.0/mask.shape[0] , 
                         (max(x+bb[2])-min(x+bb[2]))*1.0/mask.shape[1]])
        return desc
    
    #===========================================================================
    # APPROVED   
    #===========================================================================
    def bottom_height(self, mask, mask_crop, bb):
        '''
        description needs to be inserted
        '''
        (y,_) = mask_crop.nonzero()#[y,x] = find(mask_crop>0)
        desc = max(y+bb[0])*1.0/mask.shape[0]#desc = max(y+bb(1))/size(mask,1)
        return desc
    
    #===========================================================================
    # APPROVED
    #===========================================================================
    def bottom_text_hist_mr(self, centers, textons, borders):
        '''
        description needs to be inserted
        '''
        mask = borders[:,:,3]#mask= borders[:,:,4]
        mask = mask.flatten(1)#mask = mask[:]
        textonIm = textons['mr_filter'].flatten(1)#anschauen
        #textonIm(mask) = []
        mask_inv = (mask==0).flatten(1)
        #mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv]
         
        dictionarySize = centers['mr_resp_centers'].shape[0]
        ####################dictionarySize = size(centers.mr_resp_centers,1)
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        #####desc = calculate_texton_hist( textonIm, dictionarySize )
        
        return desc
        
        
    #===========================================================================
    # APPROVED #IMRESIZE-FAILURE   
    #===========================================================================
    def centered_mask(self, im, mask_crop):
        '''
        description needs to be inserted
        '''
        (y,x) = mask_crop.nonzero()#[y x] = find(mask_crop) 
        mask_crop = mask_crop[min(y):max(y)+1, min(x):max(x)+1]
        (h,w) = mask_crop.shape#[h,w] = size(mask_crop)
        padAmount = np.fix((h-w)/2.0)#padAmount = fix((h-w)/2) @UndefinedVariable
        mask_crop = self.pad(mask_crop,(max(-padAmount,0),max(padAmount,0)))
        #mask_crop=padarray(mask_crop,[max(-padAmount,0) max(padAmount,0)],'both')
        mask_crop = mask_crop*1.0
        I = Image.fromarray(mask_crop)
        I = I.resize((32,32), Image.ANTIALIAS)
        im = np.array(I)
        desc = np.maximum(im,np.zeros(im.shape))
        #desc = max(scipy.misc.imresize(map(float,mask_crop),(32,32)))  # @UndefinedVariable
        #desc = max(self.imresize(map(float,mask_crop),(32,32)),0)#desc = max(imresize(double(mask_crop),[32 32]),0)#anschauen
        desc[desc>1] = 1
        desc[desc<0] = 0#desc(desc>1) = 1desc(desc<0) = 0
        return desc
         
    #===========================================================================
    # APPROVED - IMRESIZE - FAILURE ... BIGGER!
    #===========================================================================
    def centered_mask_sp(self, im, mask, mask_crop):
        '''
        description needs to be inserted
        8x8 double
        '''
        #[y x] = find(mask_crop)
        (y,x) = mask_crop.nonzero() 
        #[y,x] = find(mask_crop)
        mask = mask_crop[min(y):max(y)+1, min(x):max(x)+1]
        h,w = mask.shape
        padAmount = np.fix((h-w)/2)
        #print padAmount
        #mask = padarray(mask,[max(-padAmount,0) max(padAmount,0)],'both')
        mask = self.pad(mask,(max(-padAmount,0),max(padAmount,0)))
        #print mask[81,2]
        mask = mask*1.0
        I = Image.fromarray(mask)
        I = I.resize((8,8), Image.ANTIALIAS)
        im = np.array(I)
        
        desc = np.maximum(im,np.zeros(im.shape))
        #print desc
        #desc = max(scipy.misc.imresize(map(float,mask),(8,8)))  # @UndefinedVariable
        #desc = max(imresize(double(mask),[8 8]),0)
        desc[desc>1] = 1
        desc[desc<0] = 0#desc(desc>1) = 1desc(desc<0) = 0
        return desc

    def pad(self,a,(i,j), symmetric = False):
        '''
        simulates the padarray function of matlab
        '''
        log = Logger()
        #log.start('map i',1,1)
        i = int(i)#map(int, i)
        j = int(j)#map(int, j)
        #log.update()
        if symmetric:
            #log.start('range_x',1,1)
            x = np.array(range(a.shape[1]) + range(a.shape[1]-1,-1,-1))
            #log.update()
            #log.start('range_y',1,1)
            y = np.array(range(a.shape[0])+range(a.shape[0]-1,-1,-1))
            #log.update()
            #log.start('arr=mod_xy',1,1)
            len_y = len(y)
            len_x = len(x)
            #print len_x
            #print len_y
#            log.start('new_method',1,1)
            lmod = np.array(range(-j,j+a.shape[1],1))%len_y
            kmod = np.array(range(-i,i+a.shape[0],1))%len_x
            arr = [a[y[k]][x[lmod]] for k in kmod]
#            log.update()
#             log.start('old_method',1,1)
#             arr = [[a[y[k%len_y]][x[l%len_x]] 
#                    for l in xrange(-j,j+a.shape[1],1)] 
#                   for k in xrange(-i,i+a.shape[0],1) ]
#             log.update()
        else:

            #log.start('np.zwros',1,1)
            arr = np.zeros((a.shape[0]+2*i,a.shape[1]+2*j))
            #log.update()
            #log.start('arr[shape',1,1)
            arr[i:i+a.shape[0],j:j+a.shape[1]] = a
            #log.update()
            
        return np.array(arr)
    
    #===========================================================================
    # APPROVED!!!!!       
    #===========================================================================
    def color_hist(self, im, mask_crop):
        '''
        description needs to be inserted
        1x24 double
        '''
         
        desc = []
        numBins = 8
        binSize = 256/numBins
        binCenters=np.array(range(0,255-(binSize-1)/2,binSize))+(binSize-1)/2.0  # @UndefinedVariable
        binCenters = np.array([(binCenters[i]+binCenters[i+1])/2.0 for i in range(len(binCenters)-1)])
        binCenters = np.concatenate([np.array([0]), binCenters, np.array([255])])
        
        #########binCenters = (binSize-1)/2:binSize:255
        for c in range(3):
            r = im[:,:,c]#r = im[:,:,c]
            r = r*1.0

            mask_filter = (mask_crop==1).flatten(1)
            rm = r.flatten(1)[mask_filter]

            desc = np.concatenate([desc, np.histogram(rm,binCenters)[0]])
            #desc = np.concatenate([desc, np.histogram(rm, binCenters)])#*1.0/sum(mask_crop.flatten(1))
        m = mask_crop==1
        desc = desc*1.0/len(m[m])   
        #desc = desc*1.0/np.sum(mask_crop.flatten(1))
        return desc
         

    #===========================================================================
    # APPROVED - bigger difference at "Real-Test"
    #===========================================================================
    def color_std(self, im, mask_crop):
        '''
        description needs to be inserted
        3x1
        ''' 
        desc = np.zeros((3,1))*1.0#desc = zeros(3,1) @UndefinedVariable
        for c in range(3):#for c = 1:3
            r = im[:,:,c]#r = im[:,:,c)
            r = r*1.0

            mask_filter = (mask_crop==1).flatten(1)
            rm = r.flatten(1)[mask_filter]
            
            desc[c] = np.std(rm)#(axis=0)#desc(c) = std(double(r(mask_crop)))
        return desc
         
    #===========================================================================
    # APPROVED - IMRESIZE-FAILURE
    #===========================================================================
    def color_thumb(self, im, mask_crop):
        '''
        description needs to be inserted
        8x8x3 uint8
        '''
        #[y x] = find(mask_crop) 
        (y,x) = mask_crop.nonzero()
        im = im[min(y):max(y)+1, min(x):max(x)+1,:]
        ###########im = im[min(y):max(y), min(x):max(x),:]
        I = Image.fromarray(im)
        I = I.resize((8,8), Image.ANTIALIAS)
        desc = np.array(I)
        
        #desc = scipy.misc.imresize(im,(8,8))  # @UndefinedVariable
        #desc = self.imresize(im,(8,8))#desc = imresize(im,[8 8])
        return desc
         

    #===========================================================================
    # OBSERVE!!!
    #===========================================================================
    def color_thumb_mask(self, im, mask_crop):
        '''
        description needs to be inserted
        8x8x3 uint8
        '''
        #[y x] = find(mask_crop) 
        (y,x) = mask_crop.nonzero()#[y,x] = find(mask_crop)
        mask_crop = mask_crop[min(y):max(y)+1, min(x):max(x)+1]
        ######mask_crop = mask_crop[min(y):max(y), min(x):max(x)]
        im = im[min(y):max(y)+1, min(x):max(x)+1,:]#im = im[min(y):max(y), min(x):max(x),:]
        tile = np.tile(mask_crop, [1, 1, 3])
        #print tile
        tile = np.reshape(tile, (mask_crop.shape[0], mask_crop.shape[1], 3))
        #print '#################################'
        #print tile
        #print tile.shape
        #print mask_crop.shape
        sh = im.shape
        im = np.reshape(im,tile.shape)*map(np.uint8, tile)#im = im.*uint8(repmat(mask_crop,[1 1 3])) @UndefinedVariable
        im = np.reshape(im,sh)
        I = Image.fromarray(im)
        I = I.resize((8,8), Image.ANTIALIAS)
        desc = np.array(I)
        #desc = scipy.misc.imresize(im,(8,8))  # @UndefinedVariable
        #desc = self.imresize(im,(8,8))
        return desc 
         
    #===========================================================================
    # APPROVED
    #===========================================================================
    def dial_color_hist(self, im, borders):
        '''
        description needs to be inserted
        1x24 double
        '''
        mask = borders[:,:,4]#mask=borders[:,:,5]
         
        desc = []
        numBins = 8
        binSize = 256/numBins
        binCenters = np.array(range(0,255-(binSize-1)/2,binSize))+(binSize-1)/2.0#binCenters = (binSize-1)/2:binSize:255 
        binCenters = np.array([(binCenters[i]+binCenters[i+1])/2.0 for i in range(len(binCenters)-1)])
        binCenters = np.concatenate([np.array([0]), binCenters, np.array([255])])
        
        for c in range(3):
            r = im[:,:,c]
            r = r*1.0
            
            mask_filter = (mask==1).flatten(1)
            rm = r.flatten(1)[mask_filter]
            
            desc = np.concatenate([desc, np.histogram(rm,binCenters)[0]])
        m = mask==1
        desc = desc*1.0/len(m[m])    
#        desc = desc*1.0/np.sum(mask.flatten(1))
        return desc

#         for c = 1:3
#             r = im[:,:,c]
#             desc = [desc hist(double(r(mask)),binCenters)/sum(mask[:])]
         
         
         
         
    #===========================================================================
    # APPROVED
    #===========================================================================     
    def dial_text_hist_mr(self, centers, textons, borders):
        '''
        description needs to be inserted
        1x100 double
        '''

        mask = borders[:,:,4]#mask=borders[:,:,5]
        mask = mask.flatten(1)#mask = mask[:]
        textonIm = textons['mr_filter'].flatten(1)#textonIm = textons.mr_filter[:]
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv]  
         
        dictionarySize = centers['mr_resp_centers'].shape[0]#dictionarySize = size(centers.mr_resp_centers,1)
        #desc = calculate_texton_hist( textonIm, dictionarySize )#desc = calculate_texton_hist( textonIm, dictionarySize )
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        
        return desc
    
     
    #===========================================================================
    # APPROVED???!  960x1    
    #===========================================================================
    def gist_int(self, im, mask, centers, imAll):
        '''
        description needs to be inserted
        320x1 double
        '''
        log = Logger()     
#        i = Image.fromarray(im)
        #im2 = self.im2double(im)
        #im2 = Image.fromarray(im2,'L')
        #descriptors = leargist.color_gist(im2, nblocks=4)
        #return descriptors
        
        numberBlocks = 4
        if im.ndim == 3:#if ndims(im) == 3
            #img = Image.fromarray(imAll)
            #img.convert('LA')
             
            #I = np.array(img)
             
#            I = cv2.cvtColor(imAll, cv2.COLOR_BGR2GRAY)
            #log.start('round',1,1)
            I = np.round(self.rgb2gray(imAll)).astype(int)
            #log.update()
        
        
#            I = imAll.convert('LA')
#anschauen            I = rgb2gray(imAll)
#       
        #log.start('find', 1, 1)   
        (y,x) = mask.nonzero()#[y x] = find(mask)
        y1 = min(y) 
        y2 = max(y) 
        x1 = min(x)
        x2 = max(x)
        h = y2-y1+1
        w=x2-x1+1
        #log.update()
        
        #log.start('padAmount',1, 1)
        padAmount = np.around((h-w)/2.0)#padAmount = round((h-w)/2)
        #log.update()
        #log.start('self.pad',1,1)
        I = self.pad(I,(max(-padAmount,0), max(padAmount,0)),symmetric=True)
        #log.update()
        #log.start('y,y max',1,1)
        #I = padarray(I,[max(-padAmount,0) max(padAmount,0)],'symmetric','both')
        mh = (y1+y2)/2+max(-padAmount,0) 
        #log.update()
        #log.start('y,x max',1,1)
        mw = (x1+x2)/2+max(padAmount,0)
        #log.update()
        #log.start('np.floor',1,1)
        s = np.floor(max(h,w)/2)
        #log.update()
        #log.start('I[x,y,x,y,x]',1,1)
        I = I[int(np.fix(mh-s+0.5)):int(np.fix(mh+s))+1,int(np.fix(mw-s+0.5)):int(np.fix(mw+s))+1]
        ####I = I(fix(mh-s+.5):fix(mh+s),fix(mw-s+.5):fix(mw+s))
        #log.update()
        #log.start('centers',1,1)  
        G = centers['gist_centers']
        #log.update()
        #log.start('shape',1,1)
        (ro,co,_) = G.shape#[ro co ch] = size(G)
        #log.update()
        #log.start('imresizecrop',1,1)
        I = self.imresizecrop(I, (ro, co))
        #print 'aha'
        #log.update()
        #log.start('Copy',1,1)
        I = I.copy()
        #log.update()
        #log.start('fromArray',1,1)
        I = Image.fromarray(I)
        #log.update()
        #log.start('convert_L',1,1)
        I = I.convert('L')
        #log.update()
        #log.start('descriptors',1,1)
        descriptors = leargist.color_gist(I, numberBlocks)  # @UndefinedVariable
        #log.update()
        
        return descriptors[0:320]
#         #I = scipy.misc.imresize(I, (ro, co), interp='bicubic')  # @UndefinedVariable
#         ####I = imresizecrop(I,[ro co], 'bicubic')
# #anschauen output = prefilt(im2double(I),4)
# #anschauen   #desc = gistGabor(output,numberBlocks,G)
# #        return desc
         
         
    #===========================================================================
    # APPROVED
    #===========================================================================         
    def int_text_hist_mr(self, mask_crop, centers, textons):
        '''
        description needs to be inserted
        '''
         
        mask = mask_crop.flatten(1)#mask = mask_crop(:)
        textonIm = textons['mr_filter'].flatten(1)#textonIm = textons.mr_filter(:)
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv]  
             
        dictionarySize = centers['mr_resp_centers'].shape[0]#dictionarySize = size(centers.mr_resp_centers,1)
        ##desc = calculate_texton_hist( textonIm, dictionarySize )
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        
        return desc
    
    #===========================================================================
    # APPROVED
    #===========================================================================
    def left_text_hist_mr(self, centers, textons, borders):
        '''
        description needs to be inserted
        1x100 double
        '''
        mask = borders[:,:,0]#mask= borders(:,:,1)
        mask = mask.flatten(1)#mask = mask(:)
        textonIm = textons['mr_filter'].flatten(1)#textonIm = textons.mr_filter(:)
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv]  
        
        dictionarySize = centers['mr_resp_centers'].shape[0]#dictionarySize = size(centers.mr_resp_centers,1)
        #desc = calculate_texton_hist( textonIm, dictionarySize )
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        
        return desc
    
    #===========================================================================
    # APPROVED
    #===========================================================================     
    def mean_color(self, im, mask_crop):
        '''
        description needs to be inserted
        3x1 double
        '''
         
        desc = np.zeros((3,1))#desc = zeros(3,1)
        for c in range(3):#for c = 1:3
            r = im[:,:,c]#r = im[:,:,c]
            
            r = r*1.0

            mask_filter = (mask_crop==1).flatten(1)
            rm = r.flatten(1)[mask_filter]
            
            desc[c] = np.mean(rm)#r[mask_crop].mean(axis=0)#desc(c) = mean(r(mask_crop))
        return desc
         
    #===========================================================================
    # APPROVED
    #===========================================================================
    def pixel_area(self, mask, mask_crop):
        '''
        description needs to be inserted
        1x1 double
        '''
        m = mask_crop==1
        desc = np.array([len(m[m])*1.0/mask.size])
        #desc = np.sum(mask_crop.flatten(1))*1.0/mask.size#desc = sum(mask_crop[:])/numel(mask)
        return desc 
         
    #===========================================================================
    # APPROVED
    #===========================================================================     
    def right_text_hist_mr(self, centers, textons, borders):
        '''
        description needs to be inserted
        '''
        mask = borders[:,:,1]#mask= borders[:,:,2]
        mask = mask.flatten(1)#mask = mask[:]
        textonIm = textons['mr_filter'].flatten(1)#textonIm = textons.mr_filter[:]
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv]  
        
        dictionarySize = centers['mr_resp_centers'].shape[0]#dictionarySize = size(centers.mr_resp_centers,1)
        #desc = calculate_texton_hist( textonIm, dictionarySize )
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
         
        return desc
    
    #===========================================================================
    # APPROVED
    #=========================================================================== 
    def sift_hist_bottom(self, centers, textons, borders):
        '''
        description needs to be inserted
        1x100 double
        '''
        mask = borders[:,:,3]#mask= borders[:,:,4]
        mask = mask.flatten(1)#mask = mask[:]
        textonIm = textons['sift_textons'].flatten(1)#textonIm = textons.sift_textons[:]
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv]  
         
        dictionarySize = centers['sift_centers'].shape[0]#dictionarySize = size(centers.sift_centers,1)
        #desc = calculate_texton_hist( textonIm, dictionarySize )
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        return desc
    
    #===========================================================================
    # APPROVED
    #===========================================================================
    def sift_hist_dial(self, centers, textons, borders):
        '''
        description needs to be inserted
        1x100 double
        '''
         
        mask = borders[:,:,4]#mask=borders[:,:,5]
        mask = mask.flatten(1)#mask = mask[:]
        textonIm = textons['sift_textons'].flatten(1)
        #textonIm(~mask) = []
        mask_inv = (mask==0).flatten(1)
        mask_inv = np.array([not x for x in mask_inv])
        textonIm = textonIm[mask_inv]  
        
        dictionarySize = centers['sift_centers'].shape[0]#dictionarySize = size(centers.sift_centers,1)
        desc = self.calculate_texton_hist(textonIm, dictionarySize)
        #anschauen        desc = calculate_texton_hist( textonIm, dictionarySize )
        return desc
    
    def imresizecrop(self, img, shape):
        '''
        '''
        
        scaling = max(shape[0]*1.0/img.shape[0], shape[1]*1.0/img.shape[1])
        
        newsize = (int(np.around(img.shape[0]*scaling)),
                   int(np.around(img.shape[1]*scaling))) 
        
        img = img.copy().astype(float)
        I = Image.fromarray(img)
        I = I.resize(newsize, Image.ANTIALIAS)#Image.ANTIALIAS)
        img = np.array(I)

        (nr, nc) = img.shape
        
        sr = np.floor((nr-shape[0])/2)
        sc = np.floor((nc-shape[1])/2)

        img = img[sr:sr+shape[0]+1, sc:sc+shape[1]+1]
        
        return img

    def rgb2gray(self, rgb):
        '''
        '''
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    
    
    def im2double(self, img):
        '''
        '''
        img = img/255.0
         
        return img
#     def prefilt(self, img, fc):
#         '''
#         prefilt for the gist feature
#         '''
#         
#         w = 5
#         s1 = fc/np.sqrt(np.log(2))
#         
#         #Pad images to reduce boundary artifacts
#         img = np.log(img+1)
#         img = self.pad(img, (w, w), symmetric=True)
#         
#         sn = img.shape[0]
#         sm = img.shape[1]
#         
#         n = max(sn, sm)
#         n = n + n%2
#         
#         img = self.pad(img, (n-sn, n-sm), symmetric=True)#'post'
#         
#         #Filter
#         [fx, fy] = np.meshgrid(-n/2, y)

    def calculate_centers(self):
        '''
        '''
        pass
        
#     def mr_resp_centers(self, file_list, HOMEIMAGES, dictionarySize):
#         '''
#         '''
#         
# 
# #         [uV sV] = memory;
# #         ndata_max = min(sV.PhysicalMemory.Available/304/10,100000); %use 10% avalible memory if its smaller than the default
#         num_texton_images = min(1000,len(file_list))
#         im_indexs = np.random.permutation(len(file_list))
#         full_resp = []
# #         imIndexs = randperm(length(fileList));
# #         fullresp = [];
#         for i in im_indexs[:num_texton_images]:
#             filename = fullfile(HOMEIMAGES,fileList{i});
#             im = imread(filename);
#             [foo data2add] = FullMR8(im);
            
#         for i = imIndexs(1:numTextonImages)
#             filename = fullfile(HOMEIMAGES,fileList{i});
#             im = imread(filename);
#             [foo data2add] = FullMR8(im);
#             
#             if(size(data2add,2)>ndata_max/numTextonImages )
#                 p = randperm(size(data2add,2));
#                 data2add = data2add(:,p(1:floor(ndata_max/numTextonImages)));
#             end
#             fullresp = [fullresp data2add];
#         end
#         fullresp = fullresp';
#         
#         opts = statset('MaxIter',40,'Display','iter');
#         %% run kmeans
#         fprintf('\nRunning k-means\n');
#         [foo centers] = kmeans(fullresp, dictionarySize,'Options',opts);


# function [ centers ] = sift_centers( fileList, HOMEIMAGES, dictionarySize)
# %MAX_RESP_CENTERS Summary of this function goes here
# %   Detailed explanation goes here
# SIFTparam.grid_spacing = 1; 
# SIFTparam.patch_size = 16;
# [uV sV] = memory;
# ndata_max = min(sV.PhysicalMemory.Available/512/10,100000); %use 10% avalible memory if its smaller than the default
# numTextonImages = min(1000,length(fileList));
# imIndexs = randperm(length(fileList));
# fullresp = [];
# for i = imIndexs(1:numTextonImages)
#     filename = fullfile(HOMEIMAGES,fileList{i});
#     im = imread(filename);
#     data2add = sp_dense_sift(im,SIFTparam.grid_spacing,SIFTparam.patch_size);
#     data2add = reshape(data2add,[size(data2add,1)*size(data2add,2) size(data2add,3)]);
#     if(size(data2add,1)>ndata_max/numTextonImages )
#         p = randperm(size(data2add,1));
#         data2add = data2add(p(1:floor(ndata_max/numTextonImages)),:);
#     end
#     fullresp = [fullresp; data2add];
# end
# 
# opts = statset('MaxIter',40,'Display','iter');
# %% run kmeans
# fprintf('\nRunning k-means\n');
# [foo centers] = kmeans(fullresp, dictionarySize,'Options',opts);
        
        