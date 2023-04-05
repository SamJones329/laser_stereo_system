import numpy as np
from laser_detection import cupy
import os
from PIL import Image
from laser_stereo_system.debug.fancylogging import *
from constants import ZedMini, LaserDetection, LaserCalibration, ImageDisplay
import cv2 as cv
from typing import Tuple
import sys
from laser_stereo_system.laser_detection import color_reward, gval, pxpatch, subpx
from laser_stereo_system.util import mathutil
import math
from datetime import datetime as dt
import matplotlib.pyplot as plt

DEFAULT_IMG_DATA = [
    ("calib_imgs/set1", 0.0224, (8,6)), 
    ("calib_imgs/set2", 0, (6,9)), 
    ("calib_imgs/set3", 0, (6,9)), 
    ("calib_imgs/set4", 0, (6,9))]


def throw_out_outlier_clusters(img, gvals):
    # K-means clustering on img to segment good points from bad points
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_PP_CENTERS
    pot_pts = np.float32(gvals[:,:2])
    # 3 clusters, no given labels, 10 attempts
    compactness, labels, centers = cv.kmeans(pot_pts, 3, None, criteria, 10, flags)
    A = pot_pts[labels.ravel()==0]
    B = pot_pts[labels.ravel()==1]
    C = pot_pts[labels.ravel()==2]

    # decide whether clusters are part of laser lights or not
    # cluster into 3 sections
    # find mean of all sections together
    # find mean of each individual section
    # find cluster closest to center of image
    # throw out clusters if they are too far away
    # NOTE: this mean we use distance from mean value of cluster to 
    # center of image as a heuristic to judge it this will only be 
    # relevant if the laser is relatively nearly coaxial with the 
    # camera (similar angle, small translation)
    clusters = [A,B,C]
    goodclusters = []
    img_center = (img.shape[1]//2, img.shape[0]//2)
    quarter_x = img.shape[1]//4
    third_y = img.shape[0]//3
    minx = img_center[0] - quarter_x
    miny = img_center[1] - third_y
    maxx = img_center[0] + quarter_x
    maxy = img_center[1] + third_y
    log_info("ROI(minX,maxX,minY,maxY) = (%d, %d, %d, %d)" % (minx, maxx, miny, maxy))
    
    numgoodclusters = len(goodclusters)
    if numgoodclusters == 0:
        log_warn("No good clusters, discarding image...")
        return None
    
    oldnumgvals = gvals.shape[0]
    newgvals = goodclusters[0]
    for i in range(1, numgoodclusters):
        newgvals = np.append(newgvals, goodclusters[i], axis=0)
    
    log_info(f"\nNumber of gvals went from {oldnumgvals} to {newgvals.shape[0]}\n")

    return newgvals

def segmentation(img):
    lines = cv.HoughLines(img, 1, np.pi / 180, threshold=200, srn=2, stn=2) 
    if lines is not None: lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))

    # cv.imshow("ulines", draw_polar_lines(orig.copy(), lines))
    mergedlines = mathutil.merge_polar_lines(
        lines, 
        LaserCalibration.MERGE_HLP_LINES_ANG_THRESH, 
        LaserCalibration.MERGE_HLP_LINE_DIST_THRESH, 
        LaserDetection.NUM_LASER_LINES)

    # make all line angles positive
    for idx, line in enumerate(mergedlines):
        r, th = line
        if r < 0:
            th = mathutil.angle_wrap(th + math.pi)
            r *= -1
        mergedlines[idx,:] = r, th

    # sort by radius increaseing
    mergedlines = mergedlines[mergedlines[:, 0].argsort()] 
    return mergedlines

def imagept_laserplane_assoc(patches, polarlines):
    # associate each laser patch with a line
    # for more accuracy could calculate each patch's centroid or calculate an average distance from line of all points of a patch
    # for speed could just pick one point from patch, will likely be enough given circumstances
    # we throw out patches too far from any lines
    maxdistfromline = 50 # px
    patchgroups = [[0, []] for _ in range(LaserDetection.NUM_LASER_LINES)]
    for patch in patches:
        y, x, subpixel_offset_x = patch[0] # get point in patch
        x += subpixel_offset_x # add subpixel offset to x
        r_p = math.sqrt(x**2 + y**2) # get radius from origin to pt
        th_p = math.atan2(y, x) # get theta from +x axis to pt
        bestline = 0 # idx of best line
        minval = float('inf') # distance from point to best line
        for idx, line in enumerate(polarlines):
            r, th = polarlines[idx] # r, th for cur line
            d = abs(r - r_p * math.cos(th - th_p)) # distance from pt to cur line
            if d < minval: # if found shorter distance
                minval = d
                bestline = idx
        if minval < maxdistfromline:
            patchgroups[bestline][1].append(patch)
            patchgroups[bestline][0] += len(patch)
    return patchgroups

def calibrate(imgs: list[np.ndarray], square_size_m: float, chessboard_dims: Tuple[int,int], gpu=False, debug=False):
    Pts3d = [[] for _ in range(LaserDetection.NUM_LASER_LINES)] # 3D pts
    for imgidx, img in enumerate(imgs):
        log_info(f"Processing image {imgidx}")

        ####### Finding Chessboard #######

        mtx = ZedMini.LeftRectHD2K.K
        dist = None
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # init container for obj pts
        objp = np.zeros((chessboard_dims[0]*chessboard_dims[1], 3), dtype=np.float32)

        # makes a grid of points corresponding to each chessboard square from the chessboards 
        # ref frame, meaning that each chessboard square has a defining dimension of 1 "unit"
        # therefore must scale these according to your selected units (m) in order to get actual 
        # object points
        objp[:,:2] = np.mgrid[0:chessboard_dims[0],0:chessboard_dims[1]].T.reshape(-1,2) 
        objp *= square_size_m 

        # scale axes to be in appropriate coords as well
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) * square_size_m

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_dims)

        if not ret:
            log_warn(f"Couldn't find chessboard, discarding image {imgidx}...")
            continue
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # find rotation and translation vectors
        ret, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

        rotmat, jac = cv.Rodrigues(rvec)
        # world (chessboard) reference frame to camera reference frame transformation matrix
        world2cam = np.identity(4)
        world2cam[:3,:3] = rotmat
        world2cam[:3, 3] = tvec.flatten()
        log_info("Chessboard -> Cam Transformation Mat:")
        for row in world2cam: log_info(row)

        chessboard_plane_point = world2cam.dot([0,0,0,1])
        chessboard_normal_vec_to_point = world2cam.dot([0,0,1,1])
        chessboard_normal_vec = chessboard_normal_vec_to_point[:3] - chessboard_plane_point[:3]
        chessboard_plane = (
            # (u, v, w, -u*x -v*y -w*z) 
            chessboard_normal_vec[0], chessboard_normal_vec[1], chessboard_normal_vec[2],
            -chessboard_normal_vec[0] * chessboard_plane_point[0]
            -chessboard_normal_vec[1] * chessboard_plane_point[1]
            -chessboard_normal_vec[2] * chessboard_plane_point[2]
        )
        log_info(f"Chessboard plane:\n{chessboard_plane}")

        ####### Done Finding Chessboard #######


        if gpu: 
            reward = color_reward.get_reward_gpu(img)
            gvals = gval.calculate_gaussian_integral_windows_gpu(reward)
        else: 
            reward = color_reward.get_reward()
            gvals = gval.calculate_gaussian_integral_windows(reward)


        gvals = throw_out_outlier_clusters(img, gvals)
        if gvals is None: continue


        if gpu:
            subpxs = subpx.find_gval_subpixels_gpu(gvals, reward)
            filtered, patches = pxpatch.throw_out_small_patches_gpu(subpxs)
        else:
            subpxs = subpx.find_gval_subpixels(gvals, reward)
            filtered, patches = pxpatch.throw_out_small_patches(subpxs)


        laserpxbinary = np.zeros(filtered.shape, dtype=np.uint8)
        laserpxbinary[filtered != 0] = 255
        lines = segmentation(laserpxbinary)
        patchgroups = imagept_laserplane_assoc(patches, lines)
        

        ####### Extract 3D Points #######
        for idx, group in enumerate(patchgroups): 
            print("line %d has %d patches" % (idx, len(group)))
            for patch in group:
                for pt in patch:
                    row, col, x_offset = pt
                    pt3d = mathutil.px_2_3d(row, col+x_offset, chessboard_plane, ZedMini.LeftRectHD2K.K) #H.dot([col + x_offset, row, 1])
                    Pts3d[idx].append(pt3d)
                    # PtsImg[idx].append((col + x_offset, row))
        
        ####### Done Extracting 3D Points #######


        ####### RANSAC Plane Extraction #######

        # RANSAC: form M subjects of k points from P
        planes = []
        for lineP in Pts3d:
            potplanes = []

            M = 3
            npLineP = np.array(lineP) # type: np.ndarray

            shuffled = npLineP.copy()
            np.random.shuffle(shuffled)
            subsets = np.array_split(shuffled, M)
            for subset in subsets:
                #1.calculate centroid of points and make points relative to it
                centroid = subset.mean(axis = 0)

                #points relative to centroid
                subsetRelative = subset - centroid

                #2. calculate the singular value decomposition of the xyzT matrix and get the normal as the last column of u matrix
                u, s, v = np.linalg.svd(subsetRelative)
                log_info(f"SVD shapes:\n\tu:{u.shape}\n\t{s.shape}\n\t{v.shape}")
                normal = v[2]
                normal = normal / np.linalg.norm(normal)

                # compute distance sum d_j of all points P to the plane pi_j,L
                # this distance is the dot product of the vector from the centroid to the point with the normal vector
                distsum = subsetRelative.dot(normal).sum()
                potplanes.append((centroid, normal, distsum))

        bestplane = potplanes[0]
        for plane in potplanes:
            if plane[2] < bestplane[2]:
                bestplane = plane
        planes.append(plane[:2]) # we don't need the distance anymore
        # return plane that fits most points/minimizes distance d_j

        ####### Done RANSAC Plane Extraction #######
        
        planes = np.array(planes)
        planes = np.reshape(planes, (planes.shape[0], 6))
        log_info("Planes (centroid, normal)=<X,Y,Z,U,V,W>")
        log_info(planes)
        np.save("Camera_Relative_Laser_Planes_" + str(dt.now()), planes) 

        if debug:
            stdplanes = [
                (u, v, w, -u*x -v*y -w*z) 
                for x,y,z,u,v,w in planes
            ]
            fig = plt.figure("Calib Planes")
            ax = fig.add_subplot(projection='3d')
            ax.set_title('calib planes')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            for idx, plane in enumerate(stdplanes):

                A, B, C, D = plane
                centroid = planes[idx, :3]
                np.linspace(-6, 6, 30)
                x = np.array([
                    centroid[0] + 0.5,
                    centroid[0] - 0.5,
                    centroid[0] + 0.5,
                    centroid[0] - 0.5
                ])
                y = np.array([
                    centroid[1] - 0.5,
                    centroid[1] - 0.5,
                    centroid[1] + 0.5,
                    centroid[1] + 0.5,
                ])
                X, Y = np.meshgrid(x,y)
                f = lambda x, y : (-D - A*x - B*y) / C
                Z = f(X,Y)

                ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                color=ImageDisplay.DISP_COLORSf[idx], edgecolor='none')
            u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            ax.plot_surface(x * 0.25, y * 0.25, z * 0.25, cmap=plt.cm.YlGnBu_r)



        

def main(
        calibration_img_info: list[Tuple[str, float, Tuple[int,int]]], *, debug=False, record_data=False):
    for img_folder, square_size_m, chessboard_dims in calibration_img_info:
        imgs = []
        cpimgs = []
        filenames = []
        for filename in os.listdir(img_folder):
            if(filename.lower().endswith((".png", ".jpg", ".jpeg"))):
                print(f"Opening {filename}")
                img = Image.open(os.path.join(img_folder, filename))
                if img is not None:
                    imgs.append((filename, np.asarray(img))) # RGB format
                    cpimgs.append(cupy.asarray(img))
                    filenames.append(filename)
        calibrate(imgs)

if __name__ == "__main__":
    numargs = len(sys.argv)
    if numargs == 0:
        main(DEFAULT_IMG_DATA, debug=True, record_data=True)
    elif numargs % 3 != 0:
        log_err("Incorrect number of command line arguments. Either pass none to use default arguments or pass 3 arguments for each set of calibration images in the following form: <relative_path_to_folder> <calibration_chessboard_square_size_m> <calibration_chessboard_interior_dimensions>", True, ValueError("Bad command line arguments"))
    # main()