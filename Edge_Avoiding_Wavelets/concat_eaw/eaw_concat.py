

def eaw_concat():
    '''
    '''
    from utils.utils import eaw_concatenate
    
    
    sp_path = "/media/johann/Patrec_external5/SuperParsing/Barcelona/Experiments/EAW1/Data/Descriptors/SP_Desc_k200/super_pixels"
    scaling_path = "/home/johann/eaw_level1"
    out_path = "/home/johann/output_eaw"
    
    eaw_concatenate(sp_path, scaling_path, out_path)
    print 'hae'


if __name__ == '__main__':
    
    
    eaw_concat()
