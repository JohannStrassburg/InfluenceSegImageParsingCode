

def eaw_concat():
    '''
    '''
    from utils import eaw_concatenate
    
    
    sp_path = "/data/jstrassb/SuperParsing/Barcelona/Experiments/Shrinked_Experiments/EAW2/Data/Descriptors/SP_Desc_k200/super_pixels"
    scaling_path = "/data/jstrassb/SuperParsing/Barcelona/eaw2/level_2"
    out_path = "/data/jstrassb/SuperParsing/Barcelona/eaw2/level_summed"
    
    eaw_concatenate(sp_path, scaling_path, out_path)
    print 'hae'


if __name__ == '__main__':
    
    
    eaw_concat()
