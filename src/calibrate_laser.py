#!/usr/bin/python

import rospy
import numpy as np
import cv2
import os
from optparse import OptionParser
import math

# modifies in place and returns
def calc_chessboard_corners(board_size, square_size):
    # type:(tuple[int, int], float) -> list[tuple[float, float, float]]
    corners = []
    for i in range(board_size[0]): # height
        for j in range(board_size[1]): # width
            corners.append((j*square_size, i*square_size, 0))
    return corners

# translation of opencv drawFrameAxes fn (not available in 3.2) https://github.com/opencv/opencv/blob/b77330bafc497ddf65074783c0e3fb989604b555/modules/calib3d/src/solvepnp.cpp#L92
def draw_frame_axes(image, cameraMatrix, dist_coeffs, rvec, tvec, length, thickness=3):
    # type:(cv2.Mat, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int) -> cv2.Mat

    # project axes points
    axes_points = []
    axes_points.append([0, 0, 0]);
    axes_points.append([length, 0, 0]);
    axes_points.append([0, length, 0]);
    axes_points.append([0, 0, length]);
    axes_points = np.array(axes_points, dtype=np.float64)
    # image_pts = [] # type: list[tuple[float,float]];
    img_pts, jacob = cv2.projectPoints(axes_points, rvec, tvec, cameraMatrix, dist_coeffs);
    img_pts = np.array(img_pts, dtype=np.int)
    
    # draw axes lines
    cv2.line(image, tuple(img_pts[0].ravel()), tuple(img_pts[1].ravel()), (0, 0, 255), thickness);
    cv2.line(image, tuple(img_pts[0].ravel()), tuple(img_pts[2].ravel()), (0, 255, 0), thickness);
    cv2.line(image, tuple(img_pts[0].ravel()), tuple(img_pts[3].ravel()), (255, 0, 0), thickness);
    return image

def calibrate(data, chessboard_interior_dimensions=(9,6), square_size_m=0.1):
    # type:(list[cv2.Mat], tuple[int, int], float) -> None
    '''
    Extracts extrinsic parameters between calibrated 
    camera and horizontal lines laser light structure
    '''
    P = [] # 3D pts
    for frame in data:
        # detect calibration plane pi_c
        # use opencv stuff
        # reference: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
        # findchessboardcorner fn: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
        # image plane extraction: https://docs.opencv.org/3.4/d0/d92/samples_2cpp_2tutorial_code_2features2D_2Homography_2pose_from_homography_8cpp-example.html#a10
        # find homography fn: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
        img_grey, img_corners, img_pose = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY), frame.copy(), frame.copy()
        # corners is np.ndarray((N,2), dtype=np.float64)
        ret, corners = cv2.findChessboardCorners(img_grey, chessboard_interior_dimensions)
        if not ret:
            print("Cannot find chessboard corners.")
            return
        # corners2 = cv2.cornerSubPix(greyimg, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(img_corners, chessboard_interior_dimensions, corners, ret)
        cv2.namedWindow("Chessboard corners", cv2.WINDOW_NORMAL)
        img_corners_resized = cv2.resize(img_corners, (500,500))
        cv2.imshow("Chessboard corners", img_corners_resized)

        objPts = calc_chessboard_corners(chessboard_interior_dimensions, square_size_m)
        objPtsPlanarList = [] # type: list[tuple[float, float]]
        for pt in objPts:
            objPtsPlanarList.append((pt[0], pt[1]))
        objPtsPlanar = np.array(objPtsPlanarList) # type: np.ndarray
        
        # homogenous coordinate representation
        H, mask = cv2.findHomography(objPtsPlanar, corners)
        norm = H[0,0]**2 + H[1,0]**2 + H[2,0]**2
        
        H /= norm
        c1 = H[:,0]
        c2 = H[:,1]
        c3 = np.cross(c1, c2)

        tvec = H[:,2] # extract translation vector from homog coords
        # print("tvec: %s" % tvec)
        R = np.zeros((3,3), dtype=np.float64)
        for i in range(3):
            R[i,0] = c1[i]
            R[i,1] = c2[i]
            R[i,2] = c3[i]

        # print("R (before polar decomposition):")
        # print(R) 
        # print("det(R): %f\n" % np.linalg.det(R))
        
        # TODO - fix stuff under here vvv
        # W, U, Vt # Mat_<double>
        # check if this is right
        W, U, Vt = cv2.SVDecomp(R)
        R = U*Vt
        det = np.linalg.det(R)
        if det < 0:
            Vt[2,0] *= -1
            Vt[2,1] *= -1
            Vt[2,2] *= -1
            R = U*Vt
        # print("R (after polar decomposition):")
        # print(R)
        # print("det(R): %f\n" % np.linalg.det(R))

        camera_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1] 
        ], dtype=np.float64)
        dist_coeffs = np.array([], dtype=np.float64)
        rvec, jacobian = cv2.Rodrigues(R) # converts rotation matrix to rotation vector using Rodrigues transform
        # print("rvec: \n%s" % rvec)
        draw_frame_axes(
            img_pose, # img
            camera_matrix, # cameraMatrix - ideal
            dist_coeffs, # distCoeffs - 0
            rvec, # rot vec
            tvec, # trans vec
            2*square_size_m) # length
        # img_pose = draw_axes(img_pose, corners, objPtsPlanar)
        # optional thickness not included
        img_pose_resized = cv2.resize(img_pose, (500,500))
        cv2.imshow("Pose from coplanar points", img_pose_resized)
        cv2.waitKey(0);
    cv2.destroyAllWindows()

    # RANSAC: form M subjects of k points from P
    subsets = []
    for subset in subsets:
        # compute centroid c_j of p_j
        # subtrct centroid c_j to all points P
        # use SVD to find the plane normal n_j
        # define pi_j,L : (n_j, c_j)
        # computer distance su d_j of all points P to the plane pi_j,L
        pass
    # return plane that fits most points/minimizes distance d_j

if __name__ == "__main__":
    print("\nparsing args...\n")
    parser = OptionParser("%prog --size SIZE --square SQUARE --folder FOLDER",
                          description=None)
    parser.add_option("-f", "--folder", dest="folder",
                      help="FOLDER containing calib imgs", metavar="FOLDER")
    parser.add_option("-s", "--size", dest="size",
                      help="chessboard size as NxM, counting interior corners (e.g. a standard chessboard is 7x7)")
    parser.add_option("-q", "--square", dest="square",
                      help="chessboard square size in meters")
    options, args = parser.parse_args()
    # folder where chessboard images are located
    img_folder = options.folder
    # dimensions of chessboard in interior corners of the outer layer in form NUMxNUM
    chessboard_dims = options.size
    w_h = chessboard_dims.split('x')
    chessboard_dims = (int(w_h[0]), int(w_h[1]))
    # defining dimension of a square of the board
    cb_sq_size = float(options.square)
    print("got args chessboard size: \n%dx%d \nsquare size (m): %f \nimage folder: %s\n" % (chessboard_dims[0], chessboard_dims[1], cb_sq_size, img_folder))
    imgs = []
    filenames = []
    for filename in os.listdir(img_folder):
        img = cv2.imread(os.path.join(img_folder,filename))
        if img is not None:
            imgs.append(img)
            filenames.append(filename)
    print("found images: %s\n" % filenames)
    calibrate(imgs, chessboard_dims, cb_sq_size)

# run command
# rosrun laser_stereo_system calibrate_laser.py -s 9x6 -q 0.02884 -f /home/active_stereo/catkin_ws/src/laser_stereo_system/calib_imgs