#!/usr/bin/python

import rospy
import numpy as np
import cv2
import os
from optparse import OptionParser

# modifies in place and returns
def calc_chessboard_corners(board_size, square_size):
    # type:(tuple[int, int], float) -> list[tuple[float, float, float]]
    corners = []
    for i in range(board_size[0]): # height
        for j in range(board_size[1]): # width
            corners.append((j*square_size, i*square_size, 0))
    return corners

def draw_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    print("corner", corner)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    return img

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
        cv2.imshow("Chessboard corners", img_corners)

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
        print("tvec: %s" % tvec)
        R = np.zeros((3,3), dtype=np.float64)
        for i in range(3):
            R[i,0] = c1[i]
            R[i,1] = c2[i]
            R[i,2] = c3[i]

        print("R (before polar decomposition):")
        print(R) 
        print("det(R): %f\n" % np.linalg.det(R))
        
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
        print("R (after polar decomposition):")
        print(R)
        print("det(R): %f\n" % np.linalg.det(R))

        camera_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1] 
        ], dtype=np.float64)
        dist_coeffs = np.array([], dtype=np.float64)
        rvec, jacobian = cv2.Rodrigues(R) # converts rotation matrix to rotation vector using Rodrigues transform
        print("rvec: \n%s" % rvec)
        # cv2.drawFrameAxes(
        #     img_pose, # img
        #     camera_matrix, # cameraMatrix - ideal
        #     dist_coeffs, # distCoeffs - 0
        #     rvec, # rot vec
        #     tvec, # trans vec
        #     2*square_size_m) # length
        img_pose = draw_axes(img_pose, corners, objPtsPlanar)
        # optional thickness not included
        cv2.imshow("Pose from coplanar points", img_pose)
        cv2.waitKey(0);
        
        
        
        # cbx, cby = chessboard_interior_dimensions
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # objp = np.zeros(
        #     (cbx*cby,3),#(6*7,3), 
        #     np.float32)
        # objp[:,:2] = np.mgrid[0:cby,0:cbx].T.reshape(-1,2)
        # axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        # corners2 = cv2.cornerSubPix(
        #     img_grey,
        #     corners,(11,11),(-1,-1),
        #     criteria)
        # # Find the rotation and translation vectors.
        # ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
        # # project 3D points to image plane
        # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coeffs)
        # img = draw_axes(img_pose, corners2, imgpts)
        # cv2.namedWindow("image w/ axes", cv2.WINDOW_NORMAL)
        # cv2.imshow("image w/ axes", img)
        # cv2.waitKey(0)

        # detect laser points P_i and triangulate to form lines
        # intersect each line with the plane pi_c and append the 3D points into P
        pass
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