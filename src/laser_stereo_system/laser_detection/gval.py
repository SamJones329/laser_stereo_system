from numba import cuda, njit, prange, jit
from laser_detection import maxthreadsperblock2d
import math
from constants import LaserDetection
import numpy as np
from debug.perftracker import PerfTracker

WINLEN = LaserDetection.GVAL_WINLEN
MIN_GVAL = LaserDetection.DEFAULT_GVAL_MIN_VAL

@cuda.jit
def gpu_gvals(img, min_gval, out):
    '''CUDA kernel to calculate G values for a pixel in an image'''
    winstartrow, winstartcol = cuda.grid(2)
    # if(winstartrow % 64 == 0 and winstartcol % 64 == 0): print(winstartrow, winstartcol)
    if(winstartrow + WINLEN < img.shape[0] and winstartcol < img.shape[1]):
        G = 0
        for row in range(winstartrow, winstartrow+WINLEN):
            G += (1 - 2*abs(winstartrow - row + (WINLEN - 1) / 2)) * img[row, winstartcol] #idk if this last part is right
        G *= -1 # TODO figure out why have to do this
        if G > min_gval:
            out[winstartrow,winstartcol] = G

@PerfTracker.track("gval_gpu")
def calculate_gaussian_integral_windows_gpu(reward_img, min_gval) -> cuda.devicearray:
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
    gpu_gvals[blockspergrid, threadsperblock](d_reward_img, min_gval, output_global_mem)

    return output_global_mem

@njit
def jit_gvals(img, winstartrow, col, outimg):
    '''Helper function to calculate G values in parallel for all pixels in an image using Numba parallelization.'''
    # if(winstartrow % 64 == 0 and winstartcol % 64 == 0): print(winstartrow, winstartcol)
    if(winstartrow + WINLEN < img.shape[0] and col < img.shape[1]):
        G = 0
        for row in range(winstartrow, winstartrow+WINLEN):
            G += (1 - 2*abs(winstartrow - row + (WINLEN - 1) / 2)) * img[row, col] #idk if this last part is right
        G *= -1 # TODO figure out why have to do this
        outimg[winstartrow, col] = G
        return G

@PerfTracker.track("gval_jit")
@njit(parallel=True)
def calculate_gaussian_integral_windows_jit(reward_img) -> np.ndarray:
    '''Calculates discretized Gaussian integral over window 
    of size WINLEN. Takes in a mono laser intensity image. 
    The resulting values can be used with a tuned threshold 
    value to be considered as laser points in an image.
    ''' 
    rows = reward_img.shape[0]
    cols = reward_img.shape[1]
    # gvals = []
    gvalimg = np.zeros(reward_img.shape)
    for col in prange(cols):
        for winstart in prange(rows-WINLEN):
            G = jit_gvals(reward_img, winstart, col, gvalimg)
            # if G >= MIN_GVAL:
            # gvals.append((col, winstart+WINLEN//2, -G))

    return gvalimg

@PerfTracker.track("gval")
@jit(forceobj=True)
def calculate_gaussian_integral_windows(reward_img, min_gval) -> np.ndarray:
    '''Calculates discretized Gaussian integral over window 
    of size WINLEN. Takes in a mono laser intensity image. 
    Only returns G values that are greater than min_gval.
    ''' 
    # G_v_w = sum from v=v_0 to v_0 + l_w of (1 - 2*abs(v_0 - v + (l_w-1) / 2)) * I_L(u,v)
    rows = reward_img.shape[0]
    cols = reward_img.shape[1]
    gvals = []
    for col in range(cols):
        for winstart in range(rows-WINLEN):
            G = 0
            for row in range(winstart, winstart+WINLEN):
                G += (1 - 2*abs(winstart - row + (WINLEN - 1) / 2)) * reward_img[row,col] #idk if this last part is right
            if -G >= min_gval:
                gvals.append((col, winstart+WINLEN//2, -G))
    
    # gvals.sort(key=lambda x: x[2])
    # num_lines = 15
    # expectedgoodgvals = int(rows * num_lines * 1.6)#1.4) # room for plenty of outliers
    # gvals = gvals[:expectedgoodgvals]
    return np.array(gvals)