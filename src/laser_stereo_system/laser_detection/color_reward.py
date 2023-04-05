from laser_stereo_system.constants import LaserDetection
from laser_detection import cupy
import numpy as np
from laser_stereo_system.debug.perftracker import PerfTracker

@PerfTracker.track("reward")

def get_reward(img, weights=LaserDetection.DEFAULT_COLOR_WEIGHTS):
    return np.sum(img * weights, axis=2)

@PerfTracker.track("reward_gpu")
def get_reward_gpu(img, weights=LaserDetection.DEFAULT_COLOR_WEIGHTS):
    return cupy.sum(img * weights, axis=2)