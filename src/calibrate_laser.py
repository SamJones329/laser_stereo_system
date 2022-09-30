from curses.ascii import VT
from cv2 import findHomography
import rospy
import numpy as np
import cv2

# modifies in place and returns
def calc_chessboard_corners(board_size, square_size):
    # type:(tuple[int, int], float) -> list[tuple[float, float, float]]
    corners = []
    for i in range(board_size[0]): # height
        for j in range(board_size[1]): # width
            corners.append((j*square_size, i*square_size, 0))
    return corners

def calibrate(data, chessboard_interior_dimensions=(9,6), square_size_mm=100):
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
        img_corners, img_pose = frame.copy(), frame.copy()
        corners # type: list[tuple[float, float]]
        ret, corners = cv2.findChessboardCorners(img_corners, chessboard_interior_dimensions)
        if not ret:
            print("Cannot find chessboard corners.")
            return
        # corners2 = cv2.cornerSubPix(greyimg, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(img_corners, chessboard_interior_dimensions, corners, ret)
        
        objPts = calc_chessboard_corners(chessboard_interior_dimensions, square_size_mm)
        objPtsPlanar = [] # type: list[tuple[float, float]]
        for pt in objPts:
            objPtsPlanar.append((pt[0]), pt[1])
        
        H = findHomography(objPtsPlanar, corners)
        norm = H[0,0]**2 + H[1,0]**2 + H[2,0]**2
        
        H /= norm
        c1 = H[:,0]
        c2 = H[:,1]
        c3 = np.cross(c1, c2)

        tvec = H[:,2]
        R = np.zeros((3,3), dtype=np.float64)
        for i in range(3):
            R[i,0] = c1[i]
            R[i,1] = c2[i]
            R[i,2] = c3[i]

        print("R (before polar decomposition):\n", R, 
            "\ndet(R): ", np.linalg.det(R))
        
        # TODO - fix stuff under here vvv
        W, U, Vt # Mat_<double>
        cv2.SVDecomp(R, W, U, Vt)
        R = U*Vt
        det = np.linalg.det(R)
        if det < 0:
            Vt[2,0] *= -1
            Vt[2,1] *= -1
            Vt[2,2] *= -1
            R = U*Vt
        print("R (after polar decomposition):\n", R,
            "\ndet(R): ", np.linalg.det(R))

        camera_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1] 
        ], dtype=np.float64)
        dist_coeffs = np.array([], dtype=np.float64)
        rvec = cv2.Rodrigues(R)
        cv2.drawFrameAxes(img_pose, camera_matrix, dist_coeffs, rvec, tvec, 2*square_size_mm)
        
        cv2.imshow("Pose from coplanar points", img_pose)
        cv2.waitKey(0);


        # detect laser points P_i and triangulate to form lines
        # intersect each line with the plane pi_c and append the 3D points into P
        pass

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
    pass