from functools import wraps
import math
import os
import time
import numpy as np
from numba import cuda
from PIL import Image
from helpers import recurse_patch, maximumSpanningTree
import cv2 as cv
import cupy

# Constants, should move these to param yaml file if gonna use with ROS
WINLEN = 5 # works for 1080p and 2.2k for Zed mini
GVAL_MIN_VAL = 2000.

gpu = cuda.get_current_device()
maxthreadsperblock2d = math.floor(math.sqrt(gpu.MAX_THREADS_PER_BLOCK))

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

@timeit
def generate_laser_reward_image(img, color_weights):
    # type:(cp.ndarray, tuple[float,float,float]) -> cp.ndarray
    '''Creates a greyscale image where each pixel's value represents the likely of the pixel being part of a laser.

    :param img: (cv.Mat) MxNx3 RGB image, assumes Red-Green-Blue (RGB) color order
    :param color_weights: (tuple[float,float,float]) Values to weight colors of 
    pixels to detect laser lines, in the order (R,G,B). In the conversion to greyscale, 
    determines how much of a color's value is added to the greyscale value. e.g. if 
    (R,G,B)=(0.1, 0.5, 0.2), for a pixel with RGB(128,255,50), the resulting greyscale 
    value would be 0.1 * 128 + 0.5 * 255 + 0.2 * 50 = 150.3 ≈ 150. These values are 
    usually obtained by cross-referencing the color response charts of your camera to 
    the wavelength of your laser light.

    :return: MxN greyscale cv.Mat image or otherwise compatible arraylike
    '''
    # maybe force OpenCV to use integer images instead of floats to save memory?
    return img * color_weights
    # return img[:,:,0] * color_weights[0] + img[:,:,1] * color_weights[1] + img[:,:,2] * color_weights[2]


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

def calculate_gaussian_integral_windows(img):
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
    output = output_global_mem.copy_to_host()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    # print(output)
    print(f'\nGPU gval gen took {total_time:.4f} seconds')

    # start_time = time.perf_counter()
    # cpu_gvals(img)
    # end_time = time.perf_counter()
    # total_time = end_time - start_time
    # print(f'CPU gval gen took {total_time:.4f} seconds')    

    return output * -1 # i have no idea why they are negative but they should be positive so...

@timeit
def find_gval_subpixels(gvals, reward_img):
    if gvals.shape != reward_img.shape: raise Exception("gval array should be same size as reward_img (gval.shape != reward_img.shape)")
    subpixel_offsets = np.zeros(gvals.shape)
    offset_from_winstart_to_center = WINLEN // 2

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
            if denom == 0:
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

@timeit
def throw_out_small_patches(gval_subpixels):
    laser_patch_img = np.copy(gval_subpixels)
    patches = []
    # find patches
    for row in range(gval_subpixels.shape[0]):
        for col in range(gval_subpixels.shape[1]):
            val = laser_patch_img[row,col]
            if val > 1e-6: # found laser px, look for patch
                patch = [(row,col)]
                laser_patch_img[row,col] = 0.
                recurse_patch(row, col, patch, laser_patch_img)
                if len(patch) >= 5:
                    patches.append(patch)

    for patch in patches:
        for val in patch:
            row, col = val
            laser_patch_img[row, col] = gval_subpixels[row, col]
    return laser_patch_img            



SEGMENT_HOUGH_LINES_P = 0
SEGMENT_MAX_SPAN_TREE = 1
@timeit
def segment_laser_lines(img, segment_mode):
    if segment_mode == SEGMENT_HOUGH_LINES_P:
        return cv.HoughLines(img, 1, np.pi / 180, threshold=100, srn=2, stn=2)
        return cv.HoughLinesP(img, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)
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
    # type:(cp.ndarray) -> list[tuple(float,float,float)]
    '''
    Finds 3D coordinates of laser points in an image

    img - OpenCV Mat or otherwise compatible arraylike
    '''
    pass


if __name__ == "__main__":

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

    # example
    # x = cp.arange(10, dtype=cp.float32).reshape(2,5)
    # y = cp.arange(5, dtype=cp.float32)
    # parallel_sum = cp.ElementwiseKernel(
    #     'float32 x, float32 y',
    #     'float32 z',
    #     'z = x + y',
    #     'parallel_sum'
    # )
    # z = cp.empty((2,5), dtype=cp.float32)
    # parallel_sum(x,y,z)


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
    print("found images: %s\n" % filenames)

    # numpyrewardtimes = 0
    # for img in imgs:
    #     start_time = time.perf_counter()
    #     color_weights = (0.08, 0.85, 0.2)
    #     reward = np.sum(img * color_weights, axis=2)
    #     end_time = time.perf_counter()
    #     total_time = end_time - start_time
    #     numpyrewardtimes += total_time
    #     print(f'\nNumpy reward img took {total_time:.4f} seconds')

    #     gvals = generate_candidate_laser_pt_img(reward)
    #     goodgvals = threshold_gvals(gvals)
    
    cupyrewardtimes = 0
    for img in imgs:
        start_time = time.perf_counter()
        color_weights = (0.08, 0.85, 0.2)
        reward = cupy.sum(img * color_weights, axis=2) 
        end_time = time.perf_counter()
        total_time = end_time - start_time
        cupyrewardtimes += total_time
        print(f'Cupy reward img took {total_time:.4f} seconds')
        gvals = calculate_gaussian_integral_windows(reward)
        subpxs = find_gval_subpixels(gvals, reward)
        subpxsfiltered = throw_out_small_patches(subpxs)
        print(subpxsfiltered)
        laserpxbinary = np.zeros(subpxsfiltered.shape, dtype=np.uint8)
        cv.threshold(subpxsfiltered, 1e-6, 255, type=cv.THRESH_BINARY, dst=laserpxbinary)
        print(f"laserpxbinary {type(laserpxbinary)} {laserpxbinary}")
        # laserpxbinary = subpxsfiltered.copy()
        # laserpxbinary[laserpxbinary > 0] = 1
        # lines = segment_laser_lines(laserpxbinary, SEGMENT_HOUGH_LINES_P)
        lines = cv.HoughLines(laserpxbinary, 1, np.pi / 180, threshold=10, srn=2, stn=2)
        print(lines)
        lines = cv.HoughLinesP(laserpxbinary, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)
        print(lines)
        lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))
        dispimg = img.copy()
        print("\nLines: ")
        if lines is not None:
            lines = lines[lines[:, 0].argsort()] 
            for i in range(0, len(lines)):
                print("lines %s" % lines[i])
                rho = lines[i][0]
                theta = lines[i][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
                pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
                cv.line(dispimg, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    cv.waitKey(0)
    cv.destroyAllWindows()

    
    # print(f'\nNumpy reward img avg = {numpyrewardtimes/len(imgs):.4f} seconds')
    print(f'\nCupy reward img avg = {cupyrewardtimes/len(imgs):.4f} seconds')

    

