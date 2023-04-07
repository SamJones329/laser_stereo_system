# from laser_detector.gval import calculate_gaussian_integral_windows
from laser_detection.gval import calculate_gaussian_integral_windows_jit
import cv2 as cv
import sys
import os
import numpy as np

WIN_NAME = "Tune Color Reward"

gval_thresh = 1859#2010
weights = (0.18, 0.85, 0.12) # (k_b, k_g, k_r) (249, 255, 249) 254 255 251

def reward(img):
    global weights
    return np.sum(img * weights, axis=2)

def normalize(img):
    return img / np.max(img)

def calc(img):
    global gval_thresh
    gvals, gvalimg = calculate_gaussian_integral_windows_jit(reward(img))#.copy_to_host()
    print(f"Gvals:\n\tavg: {np.average(gvals)}\
          \n\tmin: {np.min(gvals)}\
          \n\tmax: {np.max(gvals)}\
          \n\tmed: {np.median(gvals)}\
          \n\t%over2000: {(gvals >= gval_thresh).sum() / (gvals.shape[0] * gvals.shape[1])}")
    gvalimg[gvalimg < gval_thresh] = 0
    gvalimg[gvalimg >= gval_thresh] = 1

    return gvalimg

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "calib_imgs/set4/image1.png"
    path = os.path.join(os.getcwd(), filename)
    print(f"Looking for file at {path}")
    img = cv.imread(path) # BGR img

    cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
    cv.imshow("Original Image", img)
    cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
    cv.imshow(WIN_NAME, calc(img))
    def bluechange(val):
        global weights
        weights = (val/100, weights[1], weights[2])
        cv.imshow(WIN_NAME, calc(img))
    def greenchange(val):
        global weights
        weights = (weights[0], val/100, weights[2])
        cv.imshow(WIN_NAME, calc(img))
    def redchange(val):
        global weights
        weights = (weights[0], weights[1], val/100)
        cv.imshow(WIN_NAME, calc(img))
    def gval_thresh_change(val):
        global gval_thresh
        gval_thresh = val
        cv.imshow(WIN_NAME, calc(img))
    cv.createTrackbar('Blue', WIN_NAME, int(weights[0]*100), 100, bluechange)
    cv.createTrackbar('Green', WIN_NAME, int(weights[1]*100), 100, greenchange)
    cv.createTrackbar('Red', WIN_NAME, int(weights[2]*100), 100, redchange)
    cv.createTrackbar('GvalThresh', WIN_NAME, gval_thresh, 5000, gval_thresh_change)

    while True:
        k = cv.waitKey(0) & 0xFF
        if ord('q') == k:
            cv.destroyAllWindows()
            exit(0)
