from enum import Enum
from functools import wraps
import math
import os
import time
import numpy as np
from numba import cuda # if cuda is not available, should set variable NUMBA_CUDA_SIM = 1 in terminal
from PIL import Image
from helpers import DISP_COLORS, DISP_COLORSf, maximumSpanningTree, angle_wrap, merge_polar_lines, draw_polar_lines
import cv2 as cv
import sys
import matplotlib.pyplot as plt
from camera_info import ZedMini

class LaserDetectorStep(Enum):
    ORIG = 1
    REWARD = 2
    GVAL = 3
    SUBPX = 4
    FILTER = 5
    BIN = 6
    SEGMENT = 7
    ASSOC = 8
    PCL = 9

IMG_DISPLAYS = [
    LaserDetectorStep.ORIG, 
    LaserDetectorStep.REWARD, 
    LaserDetectorStep.GVAL, 
    LaserDetectorStep.SUBPX, 
    LaserDetectorStep.FILTER,
    LaserDetectorStep.BIN,
    LaserDetectorStep.SEGMENT,
    LaserDetectorStep.ASSOC,
    LaserDetectorStep.PCL
]
TIMED_STEPS = []
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
DEFAULT_COLOR_WEIGHTS = (0.12, 0.85, 0.18)
DEFAULT_GVAL_MIN_VAL = 2010.
DEFAULT_ROI = ((0.1, 0.25), (0.9, 0.75)) # region of interest defined as (tl, tr) where tl and tr as defined by (height%, width%)
WINLEN = 5 # works for 1080p and 2.2k for Zed mini
NUM_LASER_LINES = 15
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

def timeitstep(step: LaserDetectorStep):
    if step not in TIMED_STEPS: return lambda x : x
    return timeit

@timeitstep(LaserDetectorStep.REWARD)
def reward_img(img, weights=DEFAULT_COLOR_WEIGHTS):
    return cupy.sum(img * weights, axis=2)

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

@timeitstep(LaserDetectorStep.GVAL)
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

    d_img = cuda.to_device(img)
    output_global_mem = cuda.device_array(img.shape)
    gpu_gvals[blockspergrid, threadsperblock](d_img, output_global_mem)

    return output_global_mem

@cuda.jit
def find_subpixel(gvals, ln_reward, reward_img, minval, offset_from_winstart_to_center, output):
    row, col = cuda.grid(2)
    # center of window
    center = row + offset_from_winstart_to_center
    if not (0 < row < ln_reward.shape[0]-1 and 0 < col < ln_reward.shape[1]-1) or gvals[row,col] < minval: 
        return
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

@timeitstep(LaserDetectorStep.SUBPX)
def find_gval_subpixels_gpu(gvals: cuda.devicearray, reward_img: np.ndarray, min_gval=DEFAULT_GVAL_MIN_VAL):
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
    output_global_mem = cuda.to_device(np.zeros(gvals.shape))#cuda.device_array(gvals.shape)
    find_subpixel[blockspergrid, threadsperblock](gvals, d_ln_reward, d_reward_img, min_gval, offset_from_winstart_to_center, output_global_mem)
    output = output_global_mem.copy_to_host()

    return output

@timeitstep(LaserDetectorStep.SUBPX)
def find_gval_subpixels(gvals, reward_img):
    if gvals.shape != reward_img.shape: raise Exception("gval array should be same size as reward_img (gval.shape != reward_img.shape)")
    subpixel_offsets = np.zeros(gvals.shape)
    offset_from_winstart_to_center = WINLEN // 2

    with np.errstate(divide='ignore'):
        ln_reward = np.log(reward_img) 
    for row in range(gvals.shape[0]-WINLEN//2):
        for col in range(gvals.shape[1]-WINLEN//2):
            if gvals[row,col] < DEFAULT_GVAL_MIN_VAL: continue
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
    out[outrow, outcol, 0] = 0
    out[outrow, outcol, 1] = 0 # top
    out[outrow, outcol, 2] = 0 # bottom
    out[outrow, outcol, 3] = 0 # left
    out[outrow, outcol, 4] = 0 # right
    for i in range(-3,4): # [-3, -2, -1, 0, 2, 3]
        searchingrow = row + i
        for j in range(-3,4):
            searchingcol = col + j
            if abs(img[searchingrow, searchingcol]) > minval:
                if i < 0: # top
                    out[outrow, outcol, 1] = max(abs(i), out[outrow, outcol, 1])
                else: # bottom
                    out[outrow, outcol, 2] = max(abs(i), out[outrow, outcol, 2])
                if j < 0: # left
                    out[outrow, outcol, 3] = max(abs(j), out[outrow, outcol, 3])
                else: # right
                    out[outrow, outcol, 4] = max(abs(j), out[outrow, outcol, 4])
                pxs += 1
    out[outrow, outcol, 0] = pxs                        

@timeitstep(LaserDetectorStep.FILTER)
def throw_out_small_patches_gpu(subpixel_offsets):
    threadsperblock = (maxthreadsperblock2d // 2, maxthreadsperblock2d // 2)# (16,16) # thread dims multiplied must not exceed max threads per block
    # we want each thread to have a 7x7 area to go over. we don't have 
    # to worry about going all the way to the edge since there won't be 
    # laser points there anyways and we are only throwing out max 6 rows and columns
    blockspergrid_x = int(subpixel_offsets.shape[0] / 7 / threadsperblock[0])
    blockspergrid_y = int(subpixel_offsets.shape[1] / 7 / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_arr = cuda.to_device(subpixel_offsets)
    output_global_mem = cuda.device_array((blockspergrid[0] * threadsperblock[0], blockspergrid[1] * threadsperblock[1], 5), dtype=np.float64)
    gpu_patch[blockspergrid, threadsperblock](d_arr, sys.float_info.min, output_global_mem)
    # return output_global_mem
    output = output_global_mem.copy_to_host()

    # merge
    goodblockpatches = []
    badblockpatches = []
    for i in range(1,output.shape[0]-1):
        for j in range(1,output.shape[1]-1):
            # num pxs in this patch, range to connect to other patch px on top, bottom, left, and right
            pxs, top, bottom, left, right = output[i, j]
            # use similar algo to normal throw out small patches here
            if pxs == 0: continue

            patch = []
            patchpxs = output[i,j,0]
            output[i,j,0] = 0
            toexplore = [(i,j)]

            canexplore = lambda i_x, j_x : 0 <= i_x < output.shape[0] and 0 <= j_x < output.shape[1] and output[i_x,j_x,0] > 0

            while toexplore:
                block = toexplore.pop(0)
                patch.append(block)
                explorerow, explorecol = block
                exploretop = output[explorerow, explorecol, 1]
                explorebottom = output[explorerow, explorecol, 2]
                exploreleft = output[explorerow, explorecol, 3]
                exploreright = output[explorerow, explorecol, 4]
                
                # top
                row_neighbor, col_neighbor = explorerow-1, explorecol
                if canexplore(row_neighbor, col_neighbor) and exploretop + output[row_neighbor, col_neighbor, 2] >= 3:
                    patchpxs += output[row_neighbor, col_neighbor, 0]
                    output[row_neighbor, col_neighbor] = 0
                    toexplore.append((row_neighbor, col_neighbor))
                
                # bottom
                row_neighbor, col_neighbor = explorerow+1, explorecol
                if canexplore(row_neighbor, col_neighbor) and explorebottom + output[row_neighbor, col_neighbor, 1] >= 3:
                    patchpxs += output[row_neighbor, col_neighbor, 0]
                    output[row_neighbor, col_neighbor] = 0
                    toexplore.append((row_neighbor, col_neighbor))
                
                # left
                row_neighbor, col_neighbor = explorerow, explorecol-1
                if canexplore(row_neighbor, col_neighbor) and exploreleft + output[row_neighbor, col_neighbor, 4] >= 3:
                    patchpxs += output[row_neighbor, col_neighbor, 0]
                    output[row_neighbor, col_neighbor] = 0
                    toexplore.append((row_neighbor, col_neighbor))
                
                # right
                row_neighbor, col_neighbor = explorerow, explorecol+1
                if canexplore(row_neighbor, col_neighbor) and exploreright + output[row_neighbor, col_neighbor, 3] >= 3:
                    patchpxs += output[row_neighbor, col_neighbor, 0]
                    output[row_neighbor, col_neighbor] = 0
                    toexplore.append((row_neighbor, col_neighbor))

            if patchpxs < 5: badblockpatches.append(patch)
            else: goodblockpatches.append(patch)
    
    filteredoffsets = subpixel_offsets.copy()
    for blockpatch in badblockpatches:
        for block in blockpatch:
            blockrow, blockcol = block[0] * 7, block[1] * 7
            for i in range(-3,4):
                for j in range(-3,4):
                    row, col = blockrow + i, blockcol + j
                    filteredoffsets[row,col] = 0
    goodpatches = []
    for blockpatch in goodblockpatches:
        patch = []
        for block in blockpatch:
            blockrow, blockcol = block[0] * 7, block[1] * 7
            for i in range(-3,4):
                for j in range(-3,4):
                    row, col = blockrow + i, blockcol + j
                    if(subpixel_offsets[row,col] > 0): patch.append((row,col,subpixel_offsets[row,col]))
        goodpatches.append(patch)

    return filteredoffsets, goodpatches

@timeitstep(LaserDetectorStep.FILTER)
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
@timeitstep(LaserDetectorStep.SEGMENT)
def segment_laser_lines(img, segment_mode):
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

        # cv.imshow("ulines", draw_polar_lines(orig.copy(), lines))
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


@timeitstep(LaserDetectorStep.ASSOC)
def imagept_laserplane_assoc(patches, polarlines):
    '''
    Associates each laser points in the image with one of the provided laser planes or throws it out.

    :param img: MxN binary cv.Mat image or otherwise compatible arraylike
    :param planes: (list[tuple(float,float,float)]) Planes of light projecting from a laser equiped with a diffractive optical element (DOE) described by their normal vectors in the camera reference frame.

    :return: (list[list[tuple(float,float,float)]]) Image points organized plane, with the index of the plane in the planes array corresponding to the index of its member points in the returned array.
    '''
    # associate each laser patch with a line
        # for more accuracy could calculate each patch's centroid or calculate an average distance from line of all points of a patch
        # for speed could just pick one point from patch, will likely be enough given circumstances
        # we throw out patches too far from any lines
    maxdistfromline = 50 # px
    patchgroups = [[0, []] for _ in range(NUM_LASER_LINES)]
    for patch in patches:
        y, x, subpixel_offset_x = patch[0] # get point in patch
        x += subpixel_offset_x # add subpixel offset to x
        r_p = math.sqrt(x**2 + y**2) # get radius from origin to pt
        th_p = math.atan2(y, x) # get theta from +x axis to pt
        bestline = 0 # idx of best line
        minval = float('inf') # distance from point to best line
        for idx, line in enumerate(polarlines):
            r, th = polarlines[idx] # r, th for cur line
            d = abs(r - r_p * math.cos(th - th_p)) # distance from pt to cur line
            if d < minval: # if found shorter distance
                minval = d
                bestline = idx
        if minval < maxdistfromline:
            patchgroups[bestline][1].append(patch)
            patchgroups[bestline][0] += len(patch)
    return patchgroups

@timeitstep(LaserDetectorStep.PCL)
def extract_laser_points(laserplanes, patchgroups, px_coord_offset=(0,0)) -> list[np.ndarray]:
    '''
    Finds 3D coordinates of laser points in an image

    img - OpenCV Mat or otherwise compatible arraylike
    '''
    linepts = []
    for idx, patchgroup in enumerate(patchgroups):
        c_x, c_y, f_x, f_y = ZedMini.LeftRectHD2K.P[0,2], ZedMini.LeftRectHD2K.P[1,2], ZedMini.LeftRectHD2K.P[0,0], ZedMini.LeftRectHD2K.P[1,1]
        a, b, c, d = laserplanes[idx]
        numpts, patches = patchgroup
        ptarr = np.empty((numpts,3))
        ptarridx = 0
        for patch in patches:
            for px in patch:
                u, v, offset = px
                u += offset + px_coord_offset[0]
                v += px_coord_offset[1]
                x = (u - c_x) / f_x
                y = (v - c_y) / f_y
                t = - d / (a * x + b * y + c)
                x *= t
                y *= t
                z = t
                ptarr[ptarridx,:] = x, y, z
                ptarridx += 1
        linepts.append(ptarr)
    return linepts


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
        if(filename.lower().endswith((".png", ".jpg", ".jpeg"))):
            print(f"opening {filename}")
            img = Image.open(os.path.join(img_folder, filename))
            if img is not None:
                imgs.append(np.asarray(img))
                cpimgs.append(cupy.asarray(img))
                filenames.append(filename)
    
    
    laserplanes = np.load(os.path.join(curdir, img_folder, "Camera_Relative_Laser_Planes.npy"), allow_pickle=True)
    planes = [
        (u, v, w, -u*x -v*y -w*z) 
        for x,y,z,u,v,w in laserplanes
    ]
    fig = plt.figure("Calib Planes")
    ax = fig.add_subplot(projection='3d')
    ax.set_title('calib planes')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    for idx, plane in enumerate(planes):

        A, B, C, D = plane
        centroid = laserplanes[idx, :3]
        np.linspace(-6, 6, 30)
        x = np.array([
            centroid[0] + 0.5,
            centroid[0] - 0.5,
            centroid[0] + 0.5,
            centroid[0] - 0.5
        ])
        y = np.array([
            centroid[1] - 0.5,
            centroid[1] - 0.5,
            centroid[1] + 0.5,
            centroid[1] + 0.5,
        ])
        X, Y = np.meshgrid(x,y)
        f = lambda x, y : (D - A*x - B*y) / C
        Z = f(X,Y)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        color=DISP_COLORSf[idx], edgecolor='none')
    # ax.scatter([0], [0], [0], color=[0,0,0], marker='o')
    r = 0.02
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

    imgproctimes = 0
    count = 0
    for img in imgs:

        if DEBUG_MODE:
            start_time = time.perf_counter()

        rowmin = int(DEFAULT_ROI[0][0] * img.shape[0])
        rowmax = int(DEFAULT_ROI[1][0] * img.shape[0])+1
        colmin = int(DEFAULT_ROI[0][1] * img.shape[1])
        colmax = int(DEFAULT_ROI[1][1] * img.shape[1])+1
        roi_img = img[rowmin:rowmax,colmin:colmax]
        # if LaserDetectorStep.ORIG in IMG_DISPLAYS:
        #     print(f"Img shape: {img.shape} -> {roi_img.shape}")
        #     origwin = f"Img{count}"
        #     cv.namedWindow(origwin, cv.WINDOW_NORMAL)
        #     cv.imshow(origwin, roi_img)
        #     np.save(os.path.join(img_folder, f"img{count}roi"), roi_img)

        # reward = reward_img(roi_img)
        # if LaserDetectorStep.REWARD in IMG_DISPLAYS:
        #     rewwin = f"rew{count}"
        #     cv.namedWindow(rewwin, cv.WINDOW_NORMAL)
        #     cv.imshow(rewwin, reward / np.max(reward))
        #     np.save(os.path.join(img_folder, f"img{count}rew"), reward)

        # gvals = calculate_gaussian_integral_windows(reward)
        # if LaserDetectorStep.GVAL in IMG_DISPLAYS:
        #     gvalwin = f"gvals{count}"
        #     cv.namedWindow(gvalwin, cv.WINDOW_NORMAL)
        #     hostgvals = gvals.copy_to_host()
        #     cv.imshow(gvalwin, hostgvals / np.max(hostgvals))
        #     np.save(os.path.join(img_folder, f"img{count}gvals"), hostgvals)

        # subpxs = find_gval_subpixels_gpu(gvals, reward)
        # if LaserDetectorStep.SUBPX in IMG_DISPLAYS:
        #     a = subpxs.copy()
        #     a[subpxs != 0] = 1
        #     print(f"{np.count_nonzero(a)} subpxs")
        #     subpxwin = f"subpxs {count}"
        #     cv.namedWindow(subpxwin, cv.WINDOW_NORMAL)
        #     cv.imshow(subpxwin, a)
        #     np.save(os.path.join(img_folder, f"img{count}subpxs"), subpxs)

        # subpxsfiltered, patches = throw_out_small_patches_gpu(subpxs)
        # if LaserDetectorStep.FILTER in IMG_DISPLAYS:
        #     print(f"Found ${len(patches)} good patches")
        #     filtwin = f"subpxgfilt {count}"
        #     cv.namedWindow(filtwin, cv.WINDOW_NORMAL)
        #     cv.imshow(filtwin, subpxsfiltered)
        #     np.save(os.path.join(img_folder, f"img{count}filt"), subpxsfiltered)

        # laserpxbinary = np.zeros(subpxsfiltered.shape, dtype=np.uint8)
        # laserpxbinary[subpxsfiltered != 0] = 255
        # if LaserDetectorStep.BIN in IMG_DISPLAYS:
        #     binwin = f"bin{count}"
        #     cv.namedWindow(binwin, cv.WINDOW_NORMAL)
        #     cv.imshow(binwin, laserpxbinary)
        #     np.save(os.path.join(img_folder, f"img{count}bin"), laserpxbinary)

        # lines = segment_laser_lines(laserpxbinary, SEGMENT_HOUGH_LINES)
        # if LaserDetectorStep.SEGMENT in IMG_DISPLAYS:
        #     dispimg = np.copy(roi_img)
        #     print("\nLines: ")
        #     if lines is not None:
        #         lines = lines[lines[:, 0].argsort()] 
        #         draw_polar_lines(dispimg, lines)
            
        #     lineswin = f"lines{count}"
        #     cv.namedWindow(lineswin, cv.WINDOW_NORMAL)
        #     cv.imshow(lineswin, dispimg)
        #     np.save(os.path.join(img_folder, f"img{count}lines"), lines)
        
        patchgroups = np.load(os.path.join(img_folder, f"img{count}patches.npy"), allow_pickle=True) #imagept_laserplane_assoc(patches, lines)
        if LaserDetectorStep.ASSOC in IMG_DISPLAYS:
            mergedlinespatchimg = roi_img.copy()
            # mergedlinespatchimgclear = frame.copy()
            print(f"Image {count}")
            for idx, group in enumerate(patchgroups): 
                print("line %d has %d patches" % (idx, len(group)))
                numpts, patches = group
                for patch in patches:
                    for pt in patch:
                        row, col, x_offset = pt
                        mergedlinespatchimg[row, col] = DISP_COLORS[idx]
                        # cv.circle(mergedlinespatchimgclear, (col, row), 2, DISP_COLORS[idx])
            assocwin = f"assoc{count}"
            cv.namedWindow(assocwin, cv.WINDOW_NORMAL)
            cv.imshow(assocwin, mergedlinespatchimg)
            # np.save(os.path.join(img_folder, f"img{count}patches"), patchgroups)

        linepts = extract_laser_points(planes, patchgroups, (rowmin, colmin))
        if LaserDetectorStep.PCL in IMG_DISPLAYS:
            pclfig = plt.figure(f"points{count}")
            ax = pclfig.add_subplot(projection="3d")
            ax.set_title(f"points{count}")
            for idx, line in enumerate(linepts): 
                print(f"line {idx} has {len(line)} points")
                ax.scatter(line[:,0], line[:,1], line[:,2], marker='o', color=DISP_COLORSf[idx])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            # np.save(os.path.join(img_folder, f"img{count}points"), linepts)

        if DEBUG_MODE:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            imgproctimes += total_time
            printStr = f'Image pipeline took {total_time:.4f} seconds'

        count += 1
    
    imgproctimes /= len(imgs)
    print(f"Average image processing time of {imgproctimes:.4f} seconds or {1 / imgproctimes:.4f} images per second achieved. ")

    plt.show(block=False)
    while True:
        k = cv.waitKey(0) & 0xFF
        if ord('q') == k:
            cv.destroyAllWindows()
            exit(0)