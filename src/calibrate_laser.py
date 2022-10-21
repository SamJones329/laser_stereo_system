#!/usr/bin/python

import rospy
import numpy as np
import cv2 as cv
import os
from optparse import OptionParser
import math
from datetime import datetime as dt

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

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def recurse_patch(row, col, patch, img):
    # type:(int, int, set, cv.Mat) -> None
    # check neighbors
    up = row-1
    if up >= 0 and img[up, col] > 1e-6: # up
        patch.append((up, col, img[up,col]))
        img[up,col] = 0.
        recurse_patch(up, col, patch, img)

    down = row+1
    if down <= img.shape[0] and img[down, col] > 1e-6: # down
        patch.append((down, col, img[down,col]))
        img[down,col] = 0.
        recurse_patch(down, col, patch, img)

    left = col-1
    if left >= 0 and img[row, left] > 1e-6: # left
        patch.append((row, left, img[row,left]))
        img[row,left] = 0.
        recurse_patch(row, left, patch, img)

    right = col+1
    if right <= img.shape[1] and img[row, right] > 1e-6: # right
        patch.append((row, right, img[row,right]))
        img[row,right] = 0.
        recurse_patch(row, right, patch, img)


def calibrate(data, chessboard_interior_dimensions=(9,6), square_size_m=0.1):
    # type:(list[cv.Mat], tuple[int, int], float) -> None
    '''
    Extracts extrinsic parameters between calibrated 
    camera and horizontal lines laser light structure
    '''
    s = chessboard_interior_dimensions
    P = [] # 3D pts
    for frame in data:
        # scale down display images to have 500 px height and preserve aspect ratio
        disp_size = ( int( frame.shape[1] * (500./frame.shape[0]) ), 500 )

        # ==== Find chessboard plane homography ====
        # https://www.youtube.com/watch?v=US9p9CL9Ywg
        mtx = Cam.K
        # assume img rectified already
        dist = None
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # init container for obj pts
        objp = np.zeros((s[0]*s[1], 3), dtype=np.float32)

        # makes a grid of points corresponding to each chessboard square from the chessboards 
        # ref frame, meaning that each chessboard square has a defining dimension of 1 "unit"
        # therefore must scale these according to your selected units (m) in order to get actual 
        # object points
        objp[:,:2] = np.mgrid[0:s[0],0:s[1]].T.reshape(-1,2) 
        objp *= square_size_m 
        print("object points")
        print(objp)

        # scale axes to be in appropriate coords as well
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) * square_size_m

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, s)

        if not ret:
            continue
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # find rotation and translation vectors
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # project 3d pts to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = frame.copy()
        img = draw(img, corners2, imgpts)
        imgdisp = cv.resize(img, disp_size)
        cv.imshow('img', imgdisp)

        # ==== Find laser line homography ====
        # in our case for calibration we assume the laser line we see is the 
        # intersection of the chessboard plane and the laser planes
        # meaning we only have to find the laser lines to extract their planes
        # and thus the extrinsic parameters

        # color reponse coefficients - need to get these from color spectrum 
        # response graph, am waiting on response on that
        # these are from green laser light with 532 nm wavelength and 
        # a Sony ICX674 sensor
        # these values seems to work reasonably well so I will use them in the mean time
        k_r = 0.08
        k_g = 0.85
        k_b = 0.2

        # I_L = f(k_r,k_g,k_b) = k_r*I_R + k_g*I_G + k_b*I_B, ||(k_r,k_g,k_b)|| <= 1
        I_L = k_r * frame[:,:,0] + k_g * frame[:,:,1] + k_b * frame[:,:,2]
        I_L_img = cv.resize(I_L, disp_size)
        I_L_img /= 255.0
        cv.imshow('reward', I_L_img)
        # TODO - figure out if should normalize I_L, but don't think it matter since are looking for a max
        # G_v_w = sum from v=v_0 to v_0 + l_w of (1 - 2*abs(v_0 - v + (l_w-1) / 2)) * I_L(u,v)
        winlen = 5
        rows = frame.shape[0]
        cols = frame.shape[1]
        gvals = []
        for col in range(cols):
            # print("col %d" % col)
            maxwin = 0
            maxg = 0
            for winstart in range(rows-winlen):
                G = 0
                for row in range(winstart, winstart+winlen):
                    G += (1 - 2*abs(winstart - row + (winlen - 1) / 2)) * I_L[row,col] #idk if this last part is right
                gvals.append((col, winstart+winlen//2, G))
        
        gvals.sort(key=lambda x: x[2])
        num_lines = 15
        expectedgoodgvals = int(rows * num_lines * 1.4) # room for plenty of outliers
        gvals = gvals[:expectedgoodgvals]
        gvals = np.array(gvals)
        # ming = np.min(gvals)
        # gvals += ming * -1
        # gvalsnorm = np.array(gvals, dtype=np.float)
        # gsum = np.sum(gvals)
        # print("gsum %f" % gsum)
        # gmean = np.mean(gvals)
        # print("gmean %f" % gmean)
        # gvalsnorm /= gsum
        # gnormmedian = np.median(gvalsnorm)
        # print("gnormmedian %f" % gnormmedian)
        # gnormscale = 0.5 / gnormmedian
        # print("gnormscale %f" % gnormscale)
        # gvalsnorm *= gnormscale

             
        potential_lines = frame.copy()
        # potential_lines2 = np.zeros(gray.shape)
        for idx, val in enumerate(gvals):
            x = int(val[0])
            y = int(val[1])
            cv.circle(potential_lines, (x,y), 3, (0,0,255))
            # potential_lines2[y,x] = gvalsnorm[idx,2]
            # print("gval @ x,y (%d,%d) = %f" % (x, y, gvalsnorm[idx,2]))
        potential_lines = cv.resize(potential_lines, disp_size)
        # print(potential_lines2)
        # potential_lines2 = cv.resize(potential_lines2, disp_size)
        cv.imshow("pot lines", potential_lines)
        # cv.imshow("pot lines2", potential_lines2)

        # figure out how to throw out outliers
        # maybe generate histogram and use to throw out some points

        # subpixel detection via Gaussian approximation
        # delta = 1/2 * ( ( ln(f(x-1)) - ln(f(x+1)) ) / ( ln(f(x-1)) - 2ln(f(x)) + ln(f(x+1)) ) )
        # f(x) = intensity value of particular row at pixel x
        # laser_subpixels = {}
        laser_subpixels = np.full(gray.shape, 0.0, dtype=np.float)
        laser_img = np.full(gray.shape, 0.0, dtype=np.float)
        for window in gvals:
            # center of window
            x, y = int(window[0]), int(window[1])
            # f(x), f(x-1), f(x+1)
            fx = I_L[y,x]
            fxm = I_L[y,x-1]
            fxp = I_L[y,x+1]
            denom = math.log(fxm) - 2 * math.log(fx) + math.log(fxp)
            if denom == 0:
                # 5px Center of Mass (CoM5) detector
                fxp2 = I_L[y,x+2] # f(x+2)
                fxm2 = I_L[y,x-2] # f(x-2)
                num = 2*fxp2 + fxp - fxm - 2*fxm2
                denom = fxm2 + fxm + fx + fxp + fxp2
                subpixel_offset = num / denom
            else:
                numer = math.log(fxm) - math.log(fxp)
                subpixel_offset = 0.5 * numer / denom
            if subpixel_offset > winlen//2 \
                    or x + subpixel_offset < 0 \
                    or x + subpixel_offset > laser_img.shape[1]: 
                continue
            # if laser_subpixels.has_key(y):
                # laser_subpixels[y].append(x + subpixel_offset)
            # else:
                # laser_subpixels[y] = [x + subpixel_offset]
            laser_img[y,int(x+subpixel_offset)] = 1.0
            laser_subpixels[y,int(x+subpixel_offset)] = (subpixel_offset % 1) + 1e-5

        
        # laser_disp_img = cv.resize(laser_img, disp_size)
        cv.imshow("laserimg", laser_img)

        # laser_patch_img = laser_img.copy()
        patches = []
        # find patches
        for row in range(laser_subpixels.shape[0]):
            for col in range(laser_subpixels.shape[1]):
                val = laser_subpixels[row,col]
                if val > 1e-6: # found laser px, look for patch
                    patch = [(row,col,val)]
                    laser_subpixels[row,col] = 0.
                    recurse_patch(row, col, patch, laser_subpixels)
                    if len(patch) >= 5:
                        patches.append(patch)
        # print(patches)

        laser_patch_img = np.zeros(gray.shape)
        for patch in patches:
            for val in patch:
                # print(val)
                row, col, _ = val
                laser_patch_img[row, col] = 1.
        
        cv.imshow("laserpatchimg", laser_patch_img)

        # just multiply img pts by calibration plane homography to get 3D pts
        H, mask = cv.findHomography(corners, objp)
        print(H, mask)
        pts = []
        for patch in patches:
            for pt in patch:
                # x, y
                imgpt = np.reshape([pt[1] + pt[2], pt[0], 1], (3,1))
                newpt = np.dot(H, imgpt)
                pts.append(newpt)
                # pts.append((pt[1] + pt[2], pt[0]))
        # print(pts) figure out way to plot this

        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            t = str(dt.now())
            cv.imwrite('pose' + t + '.png', img)
            cv.imwrite('laserreward' + t + '.png', I_L)
            cv.imwrite('laserlinepts' + t + '.png', potential_lines)
            cv.imwrite('lasersubpxpts' + t + '.png', laser_img * 255)
            cv.imwrite('laserpatches' + t + '.png', laser_patch_img * 255)
        cv.destroyAllWindows()

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
        img = cv.imread(os.path.join(img_folder,filename))
        if img is not None:
            imgs.append(img)
            filenames.append(filename)
    print("found images: %s\n" % filenames)
    calibrate(imgs, chessboard_dims, cb_sq_size)

# run command
# rosrun laser_stereo_system calibrate_laser.py -s 9x6 -q 0.02884 -f /home/active_stereo/catkin_ws/src/laser_stereo_system/calib_imgs