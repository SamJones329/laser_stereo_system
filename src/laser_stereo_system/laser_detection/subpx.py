from numba import cuda, njit, prange, jit
import numpy as np
from constants import LaserDetection
from laser_detection import maxthreadsperblock2d, cupy
import math
from debug.perftracker import PerfTracker

WINLEN = LaserDetection.GVAL_WINLEN

@cuda.jit
def find_subpixel_gpu(gvals, reward_img, minval, offset_from_winstart_to_center, output):
    '''CUDA kernel to find subpixel offset of a single pixel (if valid candidate)'''
    #row, col = cuda.grid(2)
    # if (row,col) not in gvalset: return
    # center of window
    for row in prange(2, gvals.shape[0]-2):
        for col in prange(2, gvals.shape[1]-2):
            center = row + offset_from_winstart_to_center
            if gvals[row,col] < minval: 
                continue
            #if not (0 < row < ln_reward.shape[0]-1 and 0 < col < ln_reward.shape[1]-1) or gvals[row,col] < minval: 
            # TODO if abs(offset) >= 1, move the pixel the offset is relative to 
            # correspondingly subtracting the offset until the abs(offset) < 1. If 
            # new pixel comes along that lands in there that has to travel less from its offset, 
            # use the new pixel instead

            # ln(f(x)), ln(f(x-1)), ln(f(x+1))
            lnfx = math.log(reward_img[center, col])
            lnfxm = math.log(reward_img[center-1, col])
            lnfxp = math.log(reward_img[center+1, col])
            denom = lnfxm - 2 * lnfx + lnfxp
            if denom == 0:
                # # 5px Center of Mass (CoM5) detector
                fx = reward_img[center, col] # f(x)
                fxp = reward_img[center+1, col] # f(x+1)
                fxm = reward_img[center-1, col] # f(x-1)
                fxp2 = reward_img[center+2, col] # f(x+2)
                fxm2 = reward_img[center-2, col] # f(x-2)
                num = 2*fxp2 + fxp - fxm - 2*fxm2
                denom = fxm2 + fxm + fx + fxp + fxp2
                subpixel_offset = num / denom
                subpixel_offset = -1
            else:
                numer = lnfxm - lnfxp
                subpixel_offset = 0.5 * numer / denom
            output[center, col] = subpixel_offset

@njit(parallel=True) # TODO: retest this
def find_subpixel(gvals, reward_img, minval, offset_from_winstart_to_center, output):
    '''Helper method to find subpixel offset of all pixels using CPU parallelization'''
    row, col = cuda.grid(2)
    # if (row,col) not in gvalset: return
    # center of window
    center = row + offset_from_winstart_to_center
    if gvals[row,col] < minval: 
        return
    #if not (0 < row < ln_reward.shape[0]-1 and 0 < col < ln_reward.shape[1]-1) or gvals[row,col] < minval: 
    # TODO if abs(offset) >= 1, move the pixel the offset is relative to 
    # correspondingly subtracting the offset until the abs(offset) < 1. If 
    # new pixel comes along that lands in there that has to travel less from its offset, 
    # use the new pixel instead

    # ln(f(x)), ln(f(x-1)), ln(f(x+1))
    lnfx = math.log(reward_img[center, col])
    lnfxm = math.log(reward_img[center-1, col])
    lnfxp = math.log(reward_img[center+1, col])
    denom = lnfxm - 2 * lnfx + lnfxp
    if denom == 0:
        # # 5px Center of Mass (CoM5) detector
        fx = reward_img[center, col] # f(x)
        fxp = reward_img[center+1, col] # f(x+1)
        fxm = reward_img[center-1, col] # f(x-1)
        fxp2 = reward_img[center+2, col] # f(x+2)
        fxm2 = reward_img[center-2, col] # f(x-2)
        num = 2*fxp2 + fxp - fxm - 2*fxm2
        denom = fxm2 + fxm + fx + fxp + fxp2
        subpixel_offset = num / denom
        subpixel_offset = -1
    else:
        numer = lnfxm - lnfxp
        subpixel_offset = 0.5 * numer / denom
    output[center, col] = subpixel_offset

@PerfTracker.track("subpx_gpu")
def find_gval_subpixels_gpu(gvals: np.ndarray, reward_img: np.ndarray, min_gval=LaserDetection.DEFAULT_GVAL_MIN_VAL, gpu=False):
    '''Calculates subpixel offsets of candidate laser pixels based on the assumption of a 
    Gaussian distribution of laser intensity. When Gaussian approximation is not possible, 
    a center of mass detector (CoM5) is used instead. Can use CPU or GPU parallelization.'''
    if gvals.shape != reward_img.shape: raise Exception("gval array should be same size as reward_img (gval.shape != reward_img.shape)")
    offset_from_winstart_to_center = WINLEN // 2

    threadsperblock = (maxthreadsperblock2d, maxthreadsperblock2d)# (32,32) # thread dims multiplied must not exceed max threads per block
    blockspergrid_x = int(math.ceil(gvals.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(gvals.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    if gpu: # TODO: retest this
        d_reward_img = cuda.to_device(reward_img)
        with np.errstate(divide='ignore'):
            d_reward_img = cupy.array(reward_img)
            ln_reward = cupy.log(d_reward_img)
            ln_reward[ln_reward == cupy.nan] = 0
            ln_reward[abs(ln_reward) == cupy.inf] = 0
        d_ln_reward = cuda.to_device(ln_reward)
        output_global_mem = cuda.to_device(np.zeros(gvals.shape))#cuda.device_array(gvals.shape)
        find_subpixel[blockspergrid, threadsperblock](cuda.to_device(gvals), d_reward_img, min_gval, offset_from_winstart_to_center, output_global_mem)
        output = output_global_mem.copy_to_host()
    else:
        output = np.zeros(gvals.shape)
        find_subpixel(gvals, reward_img, min_gval, offset_from_winstart_to_center, output)
    return output


@PerfTracker.track("subpx")
@jit(forceobj=True)
def find_gval_subpixels(gvals: np.ndarray, reward_img: np.ndarray):
    '''Calculates subpixel offsets of candidate laser pixels based on the assumption of a 
    Gaussian distribution of laser intensity. When Gaussian approximation is not possible, 
    a center of mass detector (CoM5) is used instead. Uses Numba JIT compilation.'''
    # subpixel detection via Gaussian approximation
    # delta = 1/2 * ( ( ln(f(x-1)) - ln(f(x+1)) ) / ( ln(f(x-1)) - 2ln(f(x)) + ln(f(x+1)) ) )
    # f(x) = intensity value of particular row at pixel x
    # laser_subpixels = {}
    laser_subpixels = np.full(reward_img.shape, 0.0, dtype=np.float)
    # laser_img = np.full(reward_img.shape, 0.0, dtype=np.float)
    badoffsets = 0
    for window in gvals:
        # center of window
        x, y = int(window[0]), int(window[1])
        if y < 2 or y > reward_img.shape[0]-3 or x < 2 or x > reward_img.shape[1]-3:
            continue
        # f(x), f(x-1), f(x+1)
        fx = reward_img[y,x]
        fxm = reward_img[y,x-1]
        fxp = reward_img[y,x+1]
        denom = math.log(fxm) - 2 * math.log(fx) + math.log(fxp)
        if denom == 0:
            # 5px Center of Mass (CoM5) detector
            fxp2 = reward_img[y,x+2] # f(x+2)
            fxm2 = reward_img[y,x-2] # f(x-2)
            num = 2*fxp2 + fxp - fxm - 2*fxm2
            denom = fxm2 + fxm + fx + fxp + fxp2
            subpixel_offset = num / denom
        else:
            numer = math.log(fxm) - math.log(fxp)
            subpixel_offset = 0.5 * numer / denom
        if abs(subpixel_offset) > WINLEN//2:
            badoffsets += 1
        if x + subpixel_offset < 0 or x + subpixel_offset > laser_subpixels.shape[1]: 
            continue
        # laser_img[y,int(x+subpixel_offset)] = 1.0
        laser_subpixels[y,int(x+subpixel_offset)] = (subpixel_offset % 1) + 1e-5
    # print("badoffsets: ", badoffsets, "count: ", count, "windows: ", windows, "comcount: ", comcount, "lncount: ", lncount, "oobcount: ", oobcount, "ooboffsetcount: ", ooboffsetcount)
    return laser_subpixels