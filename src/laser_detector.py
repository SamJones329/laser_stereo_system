from functools import wraps
import math
import os
import time
import numpy as np
from numba import cuda # if cuda is not available, should set variable NUMBA_CUDA_SIM = 1 in terminal
from PIL import Image
from helpers import recurse_patch, maximumSpanningTree, angle_wrap, merge_polar_lines, draw_polar_lines
import cv2 as cv
import sys

DEBUG_MODE = True

# if cupy not available (i.e. system w/out nvidia GPU), use numpy
try:
    import cupy
    CUDASIM = False
    gpu = cuda.get_current_device()
    maxthreadsperblock2d = math.floor(math.sqrt(gpu.MAX_THREADS_PER_BLOCK))
except:
    import numpy as cupy
    CUDASIM = True
    gpu = cuda.current_context().device
    maxthreadsperblock2d = math.floor(math.sqrt(1024))

# Constants, should move these to param yaml file if gonna use with ROS
NUM_LASER_LINES = 15
WINLEN = 5 # works for 1080p and 2.2k for Zed mini
GVAL_MIN_VAL = 1975.
MERGE_HLP_LINE_DIST_THRESH = 20
MERGE_HLP_LINES_ANG_THRESH = 3

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        printStr = f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds'
        if len(printStr) > 100:
            printStr = f'Function {func.__name__} Took {total_time:.4f} seconds'
        print(printStr)
        print()
        return result
    return timeit_wrapper

@cuda.jit
def gpu_gvals(img, out):
    winstartrow, winstartcol = cuda.grid(2)
    # if(winstartrow % 64 == 0 and winstartcol % 64 == 0): print(winstartrow, winstartcol)
    if(winstartrow + WINLEN < img.shape[0] and winstartcol < img.shape[1]):
        G = 0
        for row in range(winstartrow, winstartrow+WINLEN):
            G += (1 - 2*abs(winstartrow - row + (WINLEN - 1) / 2)) * img[row, winstartcol] #idk if this last part is right
        out[winstartrow,winstartcol] = G

# @timeit
# @nb.jit
# def cpu_mt_gvals(img):
#     rows = img.shape[0]
#     cols = img.shape[1]
#     gvals = []
#     for col in prange(cols):
#         for winstart in prange(rows-WINLEN):
#             G = 0
#             for row in prange(winstart, winstart+WINLEN):
#                 G += (1 - 2*abs(winstart - row + (WINLEN - 1) / 2)) * img[row,col] #idk if this last part is right
#             gvals.append((col, winstart+WINLEN//2, G))
#     return gvals

@timeit
def cpu_gvals(img):
    rows = img.shape[0]
    cols = img.shape[1]
    gvals = []
    for col in range(cols):
        for winstart in range(rows-WINLEN):
            G = 0
            for row in range(winstart, winstart+WINLEN):
                G += (1 - 2*abs(winstart - row + (WINLEN - 1) / 2)) * img[row,col] #idk if this last part is right
            gvals.append((col, winstart+WINLEN//2, G))
    return gvals

def calculate_gaussian_integral_windows(img) -> cuda.devicearray:
    '''Calculates discretized Gaussian integral over window 
    of size WINLEN. Takes in a mono laser intensity image. 
    The resulting values can be used with a tuned threshold 
    value to be considered as laser points in an image.
    ''' 

    #memoization?
        
    threadsperblock = (maxthreadsperblock2d, maxthreadsperblock2d)# (32,32) # thread dims multiplied must not exceed max threads per block
    blockspergrid_x = int(math.ceil(img.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(img.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # print(f"\nBlockDimX = {blockspergrid[0]} \nBlockDimY = {blockspergrid[1]} \nthreadsPerBlock = {threadsperblock}\n")
    # print(f"Image shape: {img.shape}\n")

    start_time = time.perf_counter()
    d_img = cuda.to_device(img)
    output_global_mem = cuda.device_array(img.shape)
    gpu_gvals[blockspergrid, threadsperblock](d_img, output_global_mem)
    # output = output_global_mem.copy_to_host()
    cupy.multiply(output_global_mem, -1)
    # output_global_mem *= -1
    end_time = time.perf_counter()
    total_time = end_time - start_time
    # print(output)
    print(f'\nGPU gval gen took {total_time:.4f} seconds')

    # start_time = time.perf_counter()
    # cpu_gvals(img)
    # end_time = time.perf_counter()
    # total_time = end_time - start_time
    # print(f'CPU gval gen took {total_time:.4f} seconds')    

    return output_global_mem
    # return output * -1 # i have no idea why they are negative but they should be positive so...

@cuda.jit
def find_subpixel(gvals, ln_reward, reward_img, minval, offset_from_winstart_to_center, output):
    row, col = cuda.grid(2)
    # center of window
    center = row + offset_from_winstart_to_center
    if not (0 < row < ln_reward.shape[0]-1 and 0 < col < ln_reward.shape[1]-1) or gvals[row,col] < minval: 
        return
        #output[center, col] = 0.
    else:    
        # ln(f(x)), ln(f(x-1)), ln(f(x+1))
        lnfx = ln_reward[center, col]
        lnfxm = ln_reward[center-1, col]
        lnfxp = ln_reward[center+1, col]
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

@timeit
def find_gval_subpixels_gpu(gvals: cuda.devicearray, reward_img: np.ndarray):
    if gvals.shape != reward_img.shape: raise Exception("gval array should be same size as reward_img (gval.shape != reward_img.shape)")
    offset_from_winstart_to_center = WINLEN // 2

    threadsperblock = (maxthreadsperblock2d, maxthreadsperblock2d)# (32,32) # thread dims multiplied must not exceed max threads per block
    blockspergrid_x = int(math.ceil(gvals.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(gvals.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    with np.errstate(divide='ignore'):
            d_reward_img = cupy.array(reward_img)
            d_ln_reward = cupy.log(d_reward_img)
            d_ln_reward[d_ln_reward == cupy.nan] = 0
            d_ln_reward[abs(d_ln_reward) == cupy.inf] = 0
            # ln_reward = np.log(reward_img)
            # ln_reward[ln_reward == np.nan] = 0
            # ln_reward[abs(ln_reward) == np.inf] = 0
            # d_ln_reward = cuda.to_device(ln_reward)
    # d_reward_img = cuda.to_device(reward)
    # d_gvals = cuda.to_device(gvals)
    output_global_mem = cuda.to_device(np.zeros(gvals.shape))#cuda.device_array(gvals.shape)
    find_subpixel[blockspergrid, threadsperblock](gvals, d_ln_reward, d_reward_img, GVAL_MIN_VAL, offset_from_winstart_to_center, output_global_mem)
    output = output_global_mem.copy_to_host()

    return output

@timeit
def find_gval_subpixels(gvals, reward_img):
    if gvals.shape != reward_img.shape: raise Exception("gval array should be same size as reward_img (gval.shape != reward_img.shape)")
    subpixel_offsets = np.zeros(gvals.shape)
    offset_from_winstart_to_center = WINLEN // 2

    with np.errstate(divide='ignore'):
        ln_reward = np.log(reward_img) 
    for row in range(gvals.shape[0]-WINLEN//2):
        for col in range(gvals.shape[1]-WINLEN//2):
            if gvals[row,col] < GVAL_MIN_VAL: continue
            # center of window
            center = row + offset_from_winstart_to_center
            
            # ln(f(x)), ln(f(x-1)), ln(f(x+1))
            lnfx = ln_reward[center, col]
            lnfxm = ln_reward[center-1, col]
            lnfxp = ln_reward[center+1, col]
            denom = lnfxm - 2 * lnfx + lnfxp
            if denom == 0 or math.isnan(denom):
                # 5px Center of Mass (CoM5) detector
                fx = reward_img[center, col] # f(x)
                fxp = reward_img[center+1, col] # f(x+1)
                fxm = reward_img[center-1, col] # f(x-1)
                fxp2 = reward_img[center+2, col] # f(x+2)
                fxm2 = reward_img[center-2, col] # f(x-2)
                num = 2*fxp2 + fxp - fxm - 2*fxm2
                denom = fxm2 + fxm + fx + fxp + fxp2
                subpixel_offset = num / denom
            else:
                numer = lnfxm - lnfxp
                subpixel_offset = 0.5 * numer / denom
            subpixel_offsets[center, col] = subpixel_offset
    return subpixel_offsets

@cuda.jit
def gpu_patch(img, minval, out):
    outrow, outcol = cuda.grid(2) 
    row, col = outrow * 7, outcol * 7
    pxs = 0
    for i in range(-3,4): # [-3, -2, -1, 0, 2, 3]
        searchingrow = row + i
        for j in range(-3,4):
            searchingcol = col + j
            if abs(img[searchingrow, searchingcol]) > minval:
                pxs += 1
    out[outrow, outcol] = pxs                        

@timeit
def throw_out_small_patches_gpu(subpixel_offsets):
    threadsperblock = (maxthreadsperblock2d // 2, maxthreadsperblock2d // 2)# (16,16) # thread dims multiplied must not exceed max threads per block
    # we want each thread to have a 7x7 area to go over. we don't have 
    # to worry about going all the way to the edge since there won't be 
    # laser points there anyways and we are only throwing out max 6 rows and columns
    blockspergrid_x = int(math.ceil(subpixel_offsets.shape[0] / 7 / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(subpixel_offsets.shape[1] / 7 / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_arr = cuda.to_device(subpixel_offsets)
    output_global_mem = cuda.device_array(subpixel_offsets.shape, dtype=np.float64)
    gpu_patch[blockspergrid, threadsperblock](d_arr, sys.float_info.min, output_global_mem)
    # return output_global_mem
    output = output_global_mem.copy_to_host()

    return output

@timeit
def throw_out_small_patches(gval_subpixels):
    laser_patch_img = np.copy(gval_subpixels)
    patches = []
    if DEBUG_MODE:
        patch_lengths = []
        patches_explored = 0
    # find patches
    for row in range(gval_subpixels.shape[0]):
        for col in range(gval_subpixels.shape[1]):
            val = laser_patch_img[row,col]
            if abs(val) > sys.float_info.min: # found laser px, look for patch
                if DEBUG_MODE: patches_explored += 1
                # patch = [(row,col)]
                # laser_patch_img[row,col] = 0.
                # recurse_patch(row, col, patch, laser_patch_img, False)
                # patch_lengths.append(len(patch))

                patch = []
                toexplore = [(row,col)]

                while toexplore:
                    px = toexplore.pop(0)
                    patch.append(px)
                    explorerow, explorecol = px
                    laser_patch_img[explorerow, explorecol] = 0
                    for i in range(-3,4): # [-3, -2, -1, 0, 2, 3]
                        searchingrow = row + i
                        for j in range(-3,4):
                            searchingcol = col + j
                            if searchingrow > 0 and searchingrow <= laser_patch_img.shape[0] and searchingcol > 0 and searchingcol <= laser_patch_img.shape[1]:   
                                if abs(laser_patch_img[searchingrow, searchingcol]) > 1e-6:
                                    toexplore.append((searchingrow, searchingcol))
                if len(patch) >= 5:
                    patches.append(patch)

    if DEBUG_MODE: print(f"{patches_explored} patches explored, {len(patches)} patches found, avg patch size {np.average(patch_lengths)}")
    for patch in patches:
        for val in patch:
            row, col = val
            laser_patch_img[row, col] = gval_subpixels[row, col]
    return laser_patch_img            



SEGMENT_HOUGH_LINES_P = 0
SEGMENT_HOUGH_LINES = 1
SEGMENT_MAX_SPAN_TREE = 2
@timeit
def segment_laser_lines(img, segment_mode, orig):
    if segment_mode == SEGMENT_HOUGH_LINES_P:
        lines = cv.HoughLinesP(laserpxbinary, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)
        print(lines)
        if lines is not None: lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))
        return lines
        # merge lines
        mergedlines = []
        for line in lines:
            start, end = line
            for otherline in lines:
                otherstart, otherend = otherline
        return lines
    elif segment_mode == SEGMENT_HOUGH_LINES:
        lines = cv.HoughLines(img, 1, np.pi / 180, threshold=200, srn=2, stn=2) 
        if lines is not None: lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))

        cv.imshow("ulines", draw_polar_lines(orig.copy(), lines))
        mergedlines = merge_polar_lines(lines, MERGE_HLP_LINES_ANG_THRESH, MERGE_HLP_LINE_DIST_THRESH, NUM_LASER_LINES)

        # make all line angles positive
        for idx, line in enumerate(mergedlines):
            r, th = line
            if r < 0:
                th = angle_wrap(th + math.pi)
                r *= -1
            mergedlines[idx,:] = r, th

        # sort by radius increaseing
        mergedlines = mergedlines[mergedlines[:, 0].argsort()] 
        return mergedlines
    elif segment_mode == SEGMENT_MAX_SPAN_TREE:
        # Let 15 parallel lines be defined as 15 groups of indexes or labels k = {1, 2, . . . , 25}. Let
        # Pl = {p1, p2, . . . , pm} a group of pixels who share at least one corner. We define pa a
        # neighbour pixel of pb if there exists one or more rows of pa shared with pb without any other
        # detected laser peak between them. The Pl groups of pixels can then be drawn as nodes in a
        # directed graph G, whose edge weight equals the number of common rows. This directed graph
        # is the input of a MST algorithm. The resulting simplified directed graph is then indexed as
        # follows: the node that does not have any parent is indexed as index 1. Then, the graph is
        # traversed and its indexing increased when an edge is followed from parents to children. This
        # yields an index for every connected vertex. An example of this approach can be seen in figure
        # 4.5. In our application, the pattern has a central dot which belongs to the central line (e.g.
        # index 13). The node belonging to that dot is labelled as k = 13 and the indexing occurs
        # traversing the graph forwards and backwards.

        # create pixel groups
        groups = []
        numgroups = len(groups)
        
        # create graph where a pixel group is connected to another pixel group by an edge with 
        # weight representing the number of shared rows
        graph = np.zeros((numgroups, numgroups), dtype=np.int)

        mst = maximumSpanningTree(graph)


def imagept_laserplane_assoc(img, planes):
    # type:(cp.ndarray, list[tuple(float, float, float)]) -> list[list[tuple(float,float,float)]]
    '''
    Associates each laser points in the image with one of the provided laser planes or throws it out.

    :param img: MxN binary cv.Mat image or otherwise compatible arraylike
    :param planes: (list[tuple(float,float,float)]) Planes of light projecting from a laser equiped with a diffractive optical element (DOE) described by their normal vectors in the camera reference frame.

    :return: (list[list[tuple(float,float,float)]]) Image points organized plane, with the index of the plane in the planes array corresponding to the index of its member points in the returned array.
    '''
    pass

def extract_laser_points(img):
    '''
    Finds 3D coordinates of laser points in an image

    img - OpenCV Mat or otherwise compatible arraylike
    '''
    pass


if __name__ == "__main__":

    if CUDASIM:
        devices = cuda.list_devices()
        print(f"devices: {devices}")
        gpu = cuda.current_context().device
        print(gpu)
        print(type(gpu))
    else:
        gpu = cuda.get_current_device()
        print(gpu)
        print(type(gpu))
        print("name = %s" % gpu.name)
        print("maxThreadsPerBlock = %s" % str(gpu.MAX_THREADS_PER_BLOCK))
        print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
        print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
        print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
        print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
        print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
        print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
        print("maxSharedMemoryPerBlock = %s" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
        print("asyncEngineCount = %s" % str(gpu.ASYNC_ENGINE_COUNT))
        print("canMapHostMemory = %s" % str(gpu.CAN_MAP_HOST_MEMORY))
        print("multiProcessorCount = %s" % str(gpu.MULTIPROCESSOR_COUNT))
        print("warpSize = %s" % str(gpu.WARP_SIZE))
        print("unifiedAddressing = %s" % str(gpu.UNIFIED_ADDRESSING))
        print("pciBusID = %s" % str(gpu.PCI_BUS_ID))
        print("pciDeviceID = %s" % str(gpu.PCI_DEVICE_ID))

    # gvals generation comparison
    curdir = os.getcwd()
    folders = curdir.split('/')
    if folders[-1] == "catkin_ws":
        img_folder = "src/laser_stereo_system/calib_imgs"
    else: 
        img_folder = "calib_imgs"
    imgs = []
    cpimgs = []
    filenames = []
    for filename in os.listdir(img_folder):
        print(f"opening {filename}")
        img = Image.open(os.path.join(img_folder, filename))
        if img is not None:
            imgs.append(np.asarray(img))
            cpimgs.append(cupy.asarray(img))
            filenames.append(filename)
    
    cupyrewardtimes = 0
    for img in imgs:
        print(f"Img shape: {img.shape}")
        # cv.imshow("img", img)
        start_time = time.perf_counter()
        color_weights = (0.18, 0.85, 0.12)
        reward = cupy.sum(img * color_weights, axis=2) 
        end_time = time.perf_counter()
        total_time = end_time - start_time
        cupyrewardtimes += total_time
        print(f'Cupy reward img took {total_time:.4f} seconds')
        # cv.imshow("rew", reward / np.max(reward))

        gvals = calculate_gaussian_integral_windows(reward)
        # cv.imshow("gvals", gvals / np.max(gvals))

        subpxs = find_gval_subpixels_gpu(gvals, reward)
        a = subpxs.copy()
        a[subpxs != 0] = 1
        # cv.namedWindow("subpxs", cv.WINDOW_NORMAL)
        # cv.imshow("subpxs", a)

        subpxsfiltered = throw_out_small_patches_gpu(subpxs)
        # cv.imshow("subpxgfilt", subpxsfiltered)
        # print(subpxsfiltered)

        # laserpxbinary = np.zeros(subpxsfiltered.shape, dtype=np.uint8)
        # print(f"there are {np.count_nonzero(subpxsfiltered)} subpxs")
        # laserpxbinary[subpxsfiltered != 0] = 255
        # # retval, laserpxbinary = cv.threshold(subpxsfiltered, sys.float_info.min, 255, type=cv.THRESH_BINARY)#, dst=laserpxbinary)
        # print(f"thresholded {np.count_nonzero(laserpxbinary)} pixels")
        # print(f"laserpxbinary {type(laserpxbinary)} {laserpxbinary}")
        # cv.imshow("bin", laserpxbinary)

        # # lines = segment_laser_lines(laserpxbinary, SEGMENT_HOUGH_LINES_P)
        # # dispimg = np.copy(img)
        # # print("\nLines: ")
        # # if lines is not None:
        # #     lines = lines[lines[:, 0].argsort()] 
        # #     for i in range(0, len(lines)):
        # #         print("line %s" % lines[i])
        # #         cv.line(dispimg, lines[i,:2], lines[i,2:], (0,0,255), 3, cv.LINE_AA)
        # # cv.imshow("lines", dispimg)

        # lines = segment_laser_lines(laserpxbinary, SEGMENT_HOUGH_LINES, img.copy())
        # linesimg = img.copy()
        # draw_polar_lines(linesimg, lines)
        # cv.imshow("lines", linesimg)

        # cv.waitKey(0)
    cv.destroyAllWindows()

    
    # print(f'\nNumpy reward img avg = {numpyrewardtimes/len(imgs):.4f} seconds')
    print(f'\nCupy reward img avg = {cupyrewardtimes/len(imgs):.4f} seconds')

    

