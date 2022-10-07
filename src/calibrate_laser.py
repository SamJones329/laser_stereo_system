#!/usr/bin/python

import rospy
import numpy as np
import cv2
import os
from optparse import OptionParser
import math

class Cam:
    '''Left Camera Params'''

    K_VGA = np.array([
        [351.9649963378906, 0.0,               314.0625         ],
        [0.0,               351.7799987792969, 179.9185028076172], 
        [0.0,               0.0,               1.0              ]
    ])
    K = np.array([
        [703.9299926757812, 0.0, 469.625], 
        [0.0, 703.5599975585938, 271.3370056152344], 
        [0.0, 0.0, 1.0]
    ])
    K_HD2K = np.array([
        [703.9299926757812, 0.0, 541.625], 
        [0.0, 703.5599975585938, 311.8370056152344], 
        [0.0, 0.0, 1.0]
    ])
    '''
    Intrinsic camera matrix for the raw (distorted) images.
        [fx  0 cx]
    K = [ 0 fy cy]
        [ 0  0  1]
    Projects 3D points in the camera coordinate frame to 2D pixel
    coordinates using the focal lengths (fx, fy) and principal point
    (cx, cy).
    '''

    D_RAW = np.array(
        [-0.17503899335861206, 0.02804959937930107, 0.0, 5.8294201153330505e-05, 0.000261220004176721]
    )
    D = np.array([], dtype=np.float64)
    '''
    The distortion parameters, size depending on the distortion model.
    For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
    '''

    R = np.array([
        [1.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0], 
        [0.0, 0.0, 1.0]
    ])
    '''
    Rectification matrix (stereo cameras only)
    A rotation matrix aligning the camera coordinate system to the ideal
    stereo image plane so that epipolar lines in both stereo images are
    parallel.
    '''

    P_VGA = np.array([
        [351.9649963378906, 0.0,               314.0625,          0.0],
        [0.0,               351.7799987792969, 179.9185028076172, 0.0], 
        [0.0,               0.0,               1.0,               0.0]
    ])
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

# modifies in place and returns
def calc_chessboard_corners(board_size, square_size):
    # type:(tuple[int, int], float) -> list[tuple[float, float, float]]
    corners = []
    for i in range(board_size[0]): # height
        for j in range(board_size[1]): # width
            corners.append((j*square_size, i*square_size,0))
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
    print(axes_points)
    print()
    print("rvec %s" % rvec)
    print("tvec %s" % tvec)
    # image_pts = [] # type: list[tuple[float,float]];
    print("\ncamera mat")
    print(cameraMatrix)
    cam_mat_m = cameraMatrix / 1000.
    print(cam_mat_m)
    img_pts, jacob = cv2.projectPoints(axes_points * 1000, rvec, tvec, cameraMatrix, dist_coeffs);
    print(img_pts)
    img_pts = np.array(img_pts, dtype=np.int)
    print(img_pts)

    # draw axes lines
    cv2.line(image, tuple(img_pts[0].ravel()), tuple(img_pts[1].ravel()), (0, 0, 255), thickness)
    cv2.line(image, tuple(img_pts[0].ravel()), tuple(img_pts[2].ravel()), (0, 255, 0), thickness)
    cv2.line(image, tuple(img_pts[0].ravel()), tuple(img_pts[3].ravel()), (255, 0, 0), thickness)
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
        img_corners_resized = cv2.resize(img_corners, (img_corners.shape[1] / 2, img_corners.shape[0] / 2))
        cv2.imshow("Chessboard corners", img_corners_resized)
        cv2.waitKey(0)

        obj_pts = np.array(calc_chessboard_corners(chessboard_interior_dimensions, square_size_m))
        # objPtsPlanarList = [] # type: list[tuple[float, float]]
        # for pt in objPts:
        #     objPtsPlanarList.append((pt[0], pt[1]))
        # objPtsPlanar = np.array(objPtsPlanarList) # type: np.ndarray
        # print("objptsplan %s" % objPtsPlanar)
        
        # try using raw image and getting camera mat from camera info topic of camera
        camera_matrix = Cam.K
        # np.array([
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, 1] 
        # ], dtype=np.float64)
        dist_coeffs = Cam.D #np.array([], dtype=np.float64)
        img_pts = cv2.undistortPoints(corners, camera_matrix, dist_coeffs);

        retval, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, Cam.K, Cam.D)
        # print(rvec, tvec)
        # # homogenous coordinate representation
        # H, mask = cv2.findHomography(obj_pts, img_pts)
        # norm = math.sqrt(H[0,0]**2 + H[1,0]**2 + H[2,0]**2)
        
        # H /= norm
        # c1 = H[:,0]
        # print("c1 %s" %c1)
        # c2 = H[:,1]
        # c3 = np.cross(c1, c2)

        # tvec = H[:,2] # extract translation vector from homog coords
        # print("tvec %s" % tvec)
        # # print("tvec: %s" % tvec)
        # R = np.zeros((3,3), dtype=np.float64)
        # for i in range(3):
        #     R[i,0] = c1[i]
        #     R[i,1] = c2[i]
        #     R[i,2] = c3[i]

        # print("R (before polar decomposition):")
        # print(R) 
        # print("det(R): %f\n" % np.linalg.det(R))
        
        # TODO - fix stuff under here vvv
        # W, U, Vt # Mat_<double>
        # check if this is right
        # W, U, Vt = cv2.SVDecomp(R)
        # print("w: %s\nu: %s\n vt: %s" % (W, U, Vt))
        # R = U*Vt
        # det = np.linalg.det(R)
        # if det < 0:
        #     Vt[2,0] *= -1
        #     Vt[2,1] *= -1
        #     Vt[2,2] *= -1
        #     R = U*Vt
        # print("R (after polar decomposition):")
        # print(R)
        # print("det(R): %f\n" % np.linalg.det(R))

        # rvec, jacobian = cv2.Rodrigues(R) # converts rotation matrix to rotation vector using Rodrigues transform
        # rvec = rvec.ravel()
        # print("rvec: \n%s" % rvec)
        draw_frame_axes(
            img_pose, # img
            camera_matrix, # cameraMatrix - pulled from camera_info topic
            dist_coeffs, # distCoeffs - 0
            rvec, # rot vec
            tvec, # trans vec
            2*square_size_m) # length
        # img_pose = draw_axes(img_pose, corners, objPtsPlanar)
        # optional thickness not included
        img_pose_resized = cv2.resize(img_pose, (img_corners.shape[1] / 2, img_corners.shape[0] / 2))
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
    print("numpy version: ", np.__version__)
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