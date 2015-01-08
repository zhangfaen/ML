'''
Created on Jan 8, 2015
@author: zhangfaen@gmail.com

This simple script demos how to compress pictures or images by SVD.
Note: Need install module PIL (Python Image Library)

For SVD, here is a very good introduction:
Latent Semantic Analysis (LSA) Tutorial
http://www.puffinwarellc.com/index.php/news-and-articles/articles/33-latent-semantic-analysis-tutorial.html?showall=1
'''

import numpy as np
import os
import Image as img

## This function converts a vector to a diagonal matrix
def cs(s, shape):
    ret = np.zeros(shape)
    for i in range(min(shape[0], shape[1])):
        ret[i][i] = s[i]
    return np.mat(ret)
    
if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    data_dir = os.path.dirname(__file__) + "/../data/"
    i = img.open(data_dir + "bao.png")
    print i.mode
    ia = np.array(i)
    
    ia_r = ia[:, :, 0] # red channel 
    #ia_g = ia[:, :, 0] # green channel
    #ia_b = ia[:, :, 0] # blue channel
    
    print ia_r.shape
#     print ia_r
    new_img = img.fromarray(ia_r, mode='L' )
    new_img.save(data_dir + "bao_rb.bmp")
    u, s, v = np.linalg.svd(ia_r)
    ms = cs(s, ia_r.shape)
    print "s"
    print s
    left = 50
    # new_ia_r = u * ms * v
    new_ia_r = u[:, 0:left] * ms[0:left, 0:left] * v[0:left, :]
    new_ia_r = np.array(new_ia_r).astype(np.uint8)
    print new_ia_r.shape
#     print new_ia_r
    new_img2 = img.fromarray(new_ia_r, mode='L' )
    new_img2.save(data_dir + "bao_rb_compressed.bmp")

    
