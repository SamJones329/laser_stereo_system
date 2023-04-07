import math
from numba import cuda
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

if __name__ == "__main__":

    if CUDASIM:
        devices = cuda.list_devices()
        print(f"devices: {devices}")
        gpu = cuda.current_context().device
        print(gpu)
        print(type(gpu))
        print("maxthreadsperblock2d = %s" % str(maxthreadsperblock2d))
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
        print("maxthreadsperblock2d = %s" % str(maxthreadsperblock2d))
        