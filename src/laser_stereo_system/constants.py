import numpy as np
from enum import Enum

class ImageDisplay:
    DISP_COLORS = [ #BGR
        (255,0,0), # royal blue
        (0,255,0), # green
        (0,0,255), # brick red
        (255,255,0), # cyan
        (255,0,255), # magenta
        (0,255,255), # yellow
        (255,160,122), # light salmon
        (180,0,0), # dark blue
        (0,180,0), # forest green
        (0,0,180), # crimson
        (180,180,0), # turquoise
        (180,0,180), # purple
        (0,180,180), # wheat
        (180,180,180), # gray
        (255,180,100), # cerulean
    ]

    DISP_COLORSf = [ #BGR
        (1,0,0), # royal blue
        (0,1,0), # green
        (0,0,1), # brick red
        (1,1,0), # cyan
        (1,0,1), # magenta
        (0,1,1), # yellow
        (1,.627,.478), # light salmon
        (.7,0,0), # dark blue
        (0,.7,0), # forest green
        (0,0,.7), # crimson
        (.7,.7,0), # turquoise
        (.7,0,.7), # purple 
        (0,.7,.7), # wheat
        (.7,.7,.7), # gray
        (1,.7,.4), # cerulean
    ]
    SHOW_ALL_IMGS = True
    DEBUG = False

class LaserCalibration:
    MERGE_HLP_LINE_DIST_THRESH = 20
    MERGE_HLP_LINES_ANG_THRESH = 3
    RECORD_DATA = True

class LaserDetection:
    NUM_LASER_LINES = 15
    DEFAULT_GVAL_MIN_VAL = 1910.
    DEFAULT_COLOR_WEIGHTS = (0.12,0.85,.12)#(0.12, 0.85, 0.18) # RGB
    GVAL_WINLEN = 5 # px
    class LaserDetectorStep(Enum):
        ORIG = 1
        REWARD = 2
        GVAL = 3
        SUBPX = 4
        FILTER = 5
        SEGMENT = 6
        ASSOC = 7
        GET3D = 8
    DEFAULT_ROI = ((0.1, 0.25), (0.9, 0.75)) # region of interest defined as (tl, tr) where tl and tr as defined by (height%, width%)

class Camera:

    D = np.array([])
    '''
    The distortion parameters, size depending on the distortion model.
    For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
    '''

    K = np.array([])
    '''
    Intrinsic camera matrix for the raw (distorted) images.
        [fx  0 cx]
    K = [ 0 fy cy]
        [ 0  0  1]
    Projects 3D points in the camera coordinate frame to 2D pixel
    coordinates using the focal lengths (fx, fy) and principal point
    (cx, cy).
    '''

    R = np.array([])
    '''
    Rectification matrix (stereo cameras only)
    A rotation matrix aligning the camera coordinate system to the ideal
    stereo image plane so that epipolar lines in both stereo images are
    parallel.
    '''

    P = np.array([])
    '''
    Projection/camera matrix
        [fx'  0  cx' Tx]
    P = [ 0  fy' cy' Ty]
        [ 0   0   1   0]
    By convention, this matrix specifies the intrinsic (camera) matrix
    of the processed (rectified) image. That is, the left 3x3 portion
    is the normal camera intrinsic matrix for the rectified image.
    It projects 3D points in the camera coordinate frame to 2D pixel
    coordinates using the focal lengths (fx', fy') and principal point
    (cx', cy') - these may differ from the values in K.
    For monocular cameras, Tx = Ty = 0. Normally, monocular cameras will
    also have R = the identity and P[1:3,1:3] = K.
    For a stereo pair, the fourth column [Tx Ty 0]' is related to the
    position of the optical center of the second camera in the first
    camera's frame. We assume Tz = 0 so both cameras are in the same
    stereo image plane. The first camera always has Tx = Ty = 0. For
    the right (second) camera of a horizontal stereo pair, Ty = 0 and
    Tx = -fx' * B, where B is the baseline between the cameras.
    Given a 3D point [X Y Z]', the projection (x, y) of the point onto
    the rectified image is given by:
    [u v w]' = P * [X Y Z 1]'
            x = u / w
            y = v / w
    This holds for both images of a stereo pair.
    '''

    def __init__(self, D, K, R, P):
        # type:(np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> Camera
        self.D = D
        self.K = K
        self.R = R
        self.P = P

class ZedMini:
    LeftRectHD2K = Camera(
        D=np.array([]),
        K=np.array([[1407.8599853515625, 0.0, 1083.25], 
                    [0.0, 1407.1199951171875, 623.6740112304688], 
                    [0.0, 0.0, 1.0]]),
        R=np.array([[1.0, 0.0, 0.0], 
                    [0.0, 1.0, 0.0], 
                    [0.0, 0.0, 1.0]]),
        P=np.array([[1407.8599853515625, 0.0, 1083.25, 0.0], 
                      [0.0, 1407.1199951171875, 623.6740112304688, 0.0], 
                      [0.0, 0.0, 1.0, 0.0]]))
