import cv2 as cv
from numpy.typing import ArrayLike
from constants import ImageDisplay

names_shown = {}

def debugshow(img: ArrayLike, name: str):
    '''
    Shows an image with the given name based on parameters set in constants.py
    
    if ImageDisplay.DEBUG is False, this function does nothing.

    if ImageDisplay.SHOW_ALL_IMGS is True, this function will show all images
    by appending a number to the end of the window name for each subsequent image 
    of the same name. Otherwise, it will replace the image in the window with the
    new image.
    '''
    if ImageDisplay.DEBUG:
        if name in names_shown:
            if ImageDisplay.SHOW_ALL_IMGS:
                names_shown[name] += 1
                winname = f"{name}{names_shown[name]}"
                cv.namedWindow(winname, cv.WINDOW_NORMAL)
            else:
                winname = name
        else:
            names_shown[name] = 1
            if ImageDisplay.SHOW_ALL_IMGS:
                winname = f"{name}{names_shown[name]}"
            else:
                winname = name
            cv.namedWindow(winname, cv.WINDOW_NORMAL)
        cv.imshow(winname, img)