import cv2 as cv
import numpy as np

def generate_laser_reward_image(img):
    # type:(cv.Mat) -> np.ndarray
    '''Creates a greyscale image where each pixel's value represents the likely of the pixel being part of a laser.

    :param img: (cv.Mat) MxNx3 RGB image

    :return: (np.ndarray) MxN greyscale img
    '''
    pass

def generate_candidate_laser_pt_img(img):
    '''Takes a greyscale laser reward image and thresholds it to a binary image

    :param img: (cv.Mat) MxN greyscale image

    :return: (cv.Mat) MxN binary image
    '''
    pass

def imagept_laserplane_assoc(img, planes):
    # type:(cv.Mat, list[tuple(float, float, float)]) -> list[list[tuple(float,float,float)]]
    '''
    Associates each laser points in the image with one of the provided laser planes or throws it out.

    :param img: (cv.Mat) MxN binary image
    :param planes: (list[tuple(float,float,float)]) Planes of light projecting from a laser equiped with a diffractive optical element (DOE) described by their normal vectors in the camera reference frame.

    :return: (list[list[tuple(float,float,float)]]) Image points organized plane, with the index of the plane in the planes array corresponding to the index of its member points in the returned array.
    '''
    pass

def extract_laser_points(img):
    # type:(cv.Mat) -> list[tuple(float,float,float)]
    '''
    Finds 3D coordinates of laser points in an image

    img - OpenCV Mat
    '''
    pass

