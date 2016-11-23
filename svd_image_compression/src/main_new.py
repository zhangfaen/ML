import numpy as np
from PIL import Image as img

def cs(s, shape):
    ret = np.zeros(shape)
    for i in range(min(shape[0], shape[1])):
        ret[i][i] = s[i]
    return np.mat(ret)

if __name__ == '__main__':
    ia = np.array(img.open("1.jpg"))
    ia_r = ia[:, :, 0] # red channel 
    new_img = img.fromarray(ia_r, mode='L' ).save("1.bmp")
    u, s, v = np.linalg.svd(ia_r)
    ms = cs(s, ia_r.shape)
    left = 70
    new_ia_r = u[:, 0:left] * ms[0:left, 0:left] * v[0:left, :]
    new_ia_r = np.array(new_ia_r).astype(np.uint8)
    new_img2 = img.fromarray(new_ia_r, mode='L' ).save("1_compressed.bmp")
    print "ratio:"
    print 1.0 * (ia_r.shape[0]*left + left + left * ia_r.shape[1]) / (ia_r.shape[0] * ia_r.shape[1])

    
