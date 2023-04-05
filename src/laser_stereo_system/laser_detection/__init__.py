from numba import cuda
from laser_stereo_system.debug import fancylogging
import math
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
    fancylogging.log_warn("Couldn't import Cupy, assuming it is not supported and Numba CUDASIM is on.")