from laser_detector import calculate_gaussian_integral_windows
import cv2 as cv
import sys
import os
import numpy as np

WIN_NAME = "Tune Color Reward"

def on_change():
    pass

if __name__ == "__main__":
    filename = sys.argv[1]
    img = cv.imread(os.path.join(os.getcwd(), filename))
    cv.imshow("img", img)
    weights = (0.08, 0.85, 0.2)
    cv.imshow("")
    def redchange(val):
        global weights
        weights = (val, weights[1], weights[2])
        newimg = calculate_gaussian_integral_windows(np.sum(img * weights, axis=2))
        cv.imshow(WIN_NAME, newimg / np.max(newimg))
    def greenchange(val):
        global weights
        weights = (weights[0], val, weights[2])
        newimg = calculate_gaussian_integral_windows(np.sum(img * weights, axis=2))
        cv.imshow(WIN_NAME, newimg / np.max(newimg))
    def bluechange(val):
        global weights
        weights = (weights[0], weights[1], val)
        newimg = calculate_gaussian_integral_windows(np.sum(img * weights, axis=2))
        cv.imshow(WIN_NAME, newimg / np.max(newimg))
    cv.createTrackbar('Red', WIN_NAME, 0, 100, redchange)
    cv.createTrackbar('Green', WIN_NAME, 0, 100, greenchange)
    cv.createTrackbar('Blue', WIN_NAME, 0, 100, bluechange)

    cv.waitKey(0)
