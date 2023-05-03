from constants import LaserDetection
from laser_detection import cupy
import numpy as np
from debug.perftracker import PerfTracker
from numba import jit

@PerfTracker.track("reward")
@jit(forceobj=True)
def get_reward(img, weights=LaserDetection.DEFAULT_COLOR_WEIGHTS):
    '''Converts an RGB image to a single channel image by computing a 
    linear combination of the channels scaled by the given weights.'''
    return np.sum(img * weights, axis=2)

@PerfTracker.track("reward_gpu")
def get_reward_gpu(img, weights=LaserDetection.DEFAULT_COLOR_WEIGHTS):
    '''Converts an RGB image to a single channel image by computing a 
    linear combination of the channels scaled by the given weights.
    Uses CuPy for GPU acceleration.'''
    return cupy.sum(img * weights, axis=2)
