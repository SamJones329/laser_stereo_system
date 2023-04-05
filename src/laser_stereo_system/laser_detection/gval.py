from numba import cuda
from laser_detection import maxthreadsperblock2d
import math
from laser_stereo_system.constants import LaserDetection
import numpy as np
from laser_stereo_system.debug.perftracker import PerfTracker

WINLEN = LaserDetection.GVAL_WINLEN

@cuda.jit
def gpu_gvals(img, out):
    winstartrow, winstartcol = cuda.grid(2)
    # if(winstartrow % 64 == 0 and winstartcol % 64 == 0): print(winstartrow, winstartcol)
    if(winstartrow + WINLEN < img.shape[0] and winstartcol < img.shape[1]):
        G = 0
        for row in range(winstartrow, winstartrow+WINLEN):
            G += (1 - 2*abs(winstartrow - row + (WINLEN - 1) / 2)) * img[row, winstartcol] #idk if this last part is right
        G *= -1 # TODO figure out why have to do this
        out[winstartrow,winstartcol] = G

@PerfTracker.track("gval_gpu")
def calculate_gaussian_integral_windows_gpu(reward_img) -> cuda.devicearray:
    '''Calculates discretized Gaussian integral over window 
    of size WINLEN. Takes in a mono laser intensity image. 
    The resulting values can be used with a tuned threshold 
    value to be considered as laser points in an image.
    ''' 
    #memoization?
        
    threadsperblock = (maxthreadsperblock2d, maxthreadsperblock2d)# (32,32) # thread dims multiplied must not exceed max threads per block
    blockspergrid_x = int(math.ceil(reward_img.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(reward_img.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_reward_img = cuda.to_device(reward_img)
    output_global_mem = cuda.device_array(reward_img.shape)
    gpu_gvals[blockspergrid, threadsperblock](d_reward_img, output_global_mem)

    return output_global_mem

@PerfTracker.track("gval")
def calculate_gaussian_integral_windows(reward_img) -> np.ndarray:
    # G_v_w = sum from v=v_0 to v_0 + l_w of (1 - 2*abs(v_0 - v + (l_w-1) / 2)) * I_L(u,v)
    rows = reward_img.shape[0]
    cols = reward_img.shape[1]
    gvals = []
    for col in range(cols):
        maxwin = 0
        maxg = 0
        for winstart in range(rows-WINLEN):
            G = 0
            for row in range(winstart, winstart+WINLEN):
                G += (1 - 2*abs(winstart - row + (WINLEN - 1) / 2)) * reward_img[row,col] #idk if this last part is right
            gvals.append((col, winstart+WINLEN//2, G))
    

    return np.array(gvals)
    gvals.sort(key=lambda x: x[2])
    num_lines = 15
    expectedgoodgvals = int(rows * num_lines * 1.6)#1.4) # room for plenty of outliers
    gvals = gvals[:expectedgoodgvals]