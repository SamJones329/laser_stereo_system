import numpy as np
from numba import cuda, jit
from laser_detection import maxthreadsperblock2d
import sys
from debug.perftracker import PerfTracker

@jit(forceobj=True)
def recurse_patch(row: int, col: int, patch: list, img: np.ndarray, onlyCheckImmediateNeighbors=True):
    '''Helper method to recurse through a patch of pixels to find contiguous pixels
    WARNING: onlyCheckImmediateNeighbors should be left to True in general, and this method should be replaced
    with an iterative method, as it is common to exceed max recursion depth when using non-strict 
    contiguity (AKA onlyCheckImmediateNeighbors=False)'''
    if onlyCheckImmediateNeighbors:
      # check neighbors
      up = row-1
      if up >= 0 and img[up, col] > 1e-6: # up
          patch.append((up, col, img[up,col]))
          img[up,col] = 0.
          recurse_patch(up, col, patch, img)

      down = row+1
      if down <= img.shape[0] and img[down, col] > 1e-6: # down
          patch.append((down, col, img[down,col]))
          img[down,col] = 0.
          recurse_patch(down, col, patch, img)

      left = col-1
      if left >= 0 and img[row, left] > 1e-6: # left
          patch.append((row, left, img[row,left]))
          img[row,left] = 0.
          recurse_patch(row, left, patch, img)

      right = col+1
      if right <= img.shape[1] and img[row, right] > 1e-6: # right
          patch.append((row, right, img[row,right]))
          img[row,right] = 0.
          recurse_patch(row, right, patch, img)
    else:
      # we define contiguity by being within 3 pixels of the source pixel
      # therefore there is a 7x7 box around the original pixel in which to search for pixels
      for i in range(-3,4): # [-3, -2, -1, 0, 2, 3]
          searchingrow = row + i
          for j in range(-3,4):
              searchingcol = col + j
              if searchingrow > 0 and searchingrow <= img.shape[0] and searchingcol > 0 and searchingcol <= img.shape[1]:   
                  if img[searchingrow, searchingcol] > 1e-6:
                      patch.append((searchingrow,searchingcol))
                      img[searchingrow, searchingcol] = 0
                      recurse_patch(searchingrow, searchingcol, patch, img, False)

@PerfTracker.track("patch")
@jit(forceobj=True)
def throw_out_small_patches(subpixel_offsets):
    '''Throws out small patches of laser points, defined as a group of less than 5 contiguous laser points.
    Currently the loose definition of contiguity is not implemented for this function due to recursion depth issues.'''
    patches = []
    # find patches
    for row in range(subpixel_offsets.shape[0]):
        for col in range(subpixel_offsets.shape[1]):
            val = subpixel_offsets[row,col]
            if val > 1e-6: # found laser px, look for patch
                patch = [(row,col,val)]
                subpixel_offsets[row,col] = 0.
                recurse_patch(row, col, patch, subpixel_offsets, True)
                if len(patch) >= 5:
                    patches.append(patch)

    laser_patch_img = np.zeros(subpixel_offsets.shape)
    numpts = 0
    for patch in patches:
        for val in patch:
            row, col, _ = val
            laser_patch_img[row, col] = 1.
            numpts += 1
    return laser_patch_img, patches

@cuda.jit
def gpu_patch(img, minval, out):
    '''Helper method to calculate outcodes for a 7x7 box in the image to reduce 
    the non-parallel work of the patching algorithm.'''
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

@PerfTracker.track("patch_gpu")
def throw_out_small_patches_gpu(subpixel_offsets) -> tuple[np.ndarray, list[list[tuple[int,int,float]]]]:
    '''Throws out small patches of laser points, defined as a group of less than 5 contiguous laser points.
    Contiguity is defined as being within 3 pixels of the source pixel, or within a 7x7 box.'''
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

    # TODO figure out why seemingly good patches are being thrown out

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