from matplotlib import pyplot as plt
import numpy as np

from demosaic_2004 import demosaicing_CFA_Bayer_Malvar2004

def mosaic(img, pattern):
    '''
    Input:
        img: H*W*3 numpy array, input image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W numpy array, output image after mosaic.
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create the H*W output numpy array.                              #   
    #   2. Discard other two channels from input 3-channel image according #
    #      to given Bayer pattern.                                         #
    #                                                                      #
    #   e.g. If Bayer pattern now is BGGR, for the upper left pixel from   #
    #        each four-pixel square, we should discard R and G channel     #
    #        and keep B channel of input image.                            #     
    #        (since upper left pixel is B in BGGR bayer pattern)           #
    ########################################################################

    height, width, dim = img.shape
    output = np.zeros((height, width))
    
    if pattern == 'BGGR':
        # Blue
        output[::2, ::2] = img[::2, ::2, 2]

        # Green (top row of the Bayer matrix)  
        output[::2, 1::2] = img[::2, 1::2, 1]

        # Green (bottom row of the Bayer matrix)
        output[1::2, ::2] = img[1::2, ::2, 1]

        # Red
        output[1::2, 1::2] = img[1::2, 1::2, 0]
        
        #print("BGGR mosaic")
        #plt.imshow(output)
        #plt.show()
    elif pattern == 'RGGB':
        # Red
        output[::2, ::2] = img[::2, ::2, 0]

        # Green (top row of the Bayer matrix)  
        output[1::2, ::2] = img[1::2, ::2, 1]

        # Green (bottom row of the Bayer matrix)
        output[::2, 1::2] = img[::2, 1::2, 1]

        # Blue
        output[1::2, 1::2] = img[1::2, 1::2, 2]
        
        #print("RGGB mosaic")
        #plt.imshow(output)
        #plt.show()
    elif pattern == 'GBRG':
        # Green
        output[::2, ::2] = img[::2, ::2, 1]

        # Blue (top row of the Bayer matrix)  
        output[::2, 1::2] = img[::2, 1::2, 2]

        # Red (bottom row of the Bayer matrix)
        output[1::2, ::2] = img[1::2, ::2, 0]

        # Green
        output[1::2, 1::2] = img[1::2, 1::2, 1]
        
        #print("GBRG mosaic")
        #plt.imshow(output)
        #plt.show()
    elif pattern == 'GRBG':
        # Green
        output[::2, ::2] = img[::2, ::2, 1]

        # Red (top row of the Bayer matrix)  
        output[::2, 1::2] = img[::2, 1::2, 0]

        # Blue (bottom row of the Bayer matrix)
        output[1::2, ::2] = img[1::2, ::2, 2]

        # Green
        output[1::2, 1::2] = img[1::2, 1::2, 1]
        
        #print("GRBG mosaic")
        #plt.imshow(output)
        #plt.show()
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    
    return output


def demosaic(img, pattern):
    '''
    Input:
        img: H*W numpy array, input RAW image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W*3 numpy array, output de-mosaic image.
    '''
    #### Using Python colour_demosaicing library
    #### You can write your own version, too
    output = demosaicing_CFA_Bayer_Malvar2004(img, pattern)
    output = np.clip(output, 0, 1)

    return output

