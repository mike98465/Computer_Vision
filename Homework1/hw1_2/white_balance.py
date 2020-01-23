from matplotlib import pyplot as plt
import numpy as np

def generate_wb_mask(img, pattern, fr, fb):
    '''
    Input:
        img: H*W numpy array, RAW image
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
        fr: float, white balance factor of red channel
        fb: float, white balance factor of blue channel 
    Output:
        mask: H*W numpy array, white balance mask
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create a numpy array with shape of input RAW image.             #
    #   2. According to the given Bayer pattern, fill the fr into          #
    #      correspinding red channel position and fb into correspinding    #
    #      blue channel position. Fill 1 into green channel position       #
    #      otherwise.                                                      #
    ########################################################################
    
    height, width = img.shape
    mask = np.ones((height, width))
    
    #print(mask)
    
    if pattern == 'GRBG':
        # Red 
        mask[::2, 1::2] = fr
        # Blue
        mask[1::2, ::2] = fb
    elif pattern == 'RGGB':
        # Red
        mask[::2, ::2] = fr
        # Blue
        mask[1::2, 1::2] = fb
    elif pattern == 'GBRG':
        # Blue 
        mask[::2, 1::2] = fb
        # Red
        mask[1::2, ::2] = fr
    elif pattern == 'BGGR':
        # Blue
        mask[::2, ::2] = fb
        # Red
        mask[1::2, 1::2] = fr
    
    #plt.imshow(mask)
    #plt.show()
    
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    
    return mask