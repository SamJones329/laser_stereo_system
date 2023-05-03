import numpy as np
from laser_detection import cupy, color_reward, gval, pxpatch, subpx
import os
from PIL import Image
from debug.fancylogging import *
from constants import ZedMini, LaserDetection, LaserCalibration, ImageDisplay
import cv2 as cv
from typing import Tuple
import sys
from util import mathutil
import math
from datetime import datetime as dt
from debug.perftracker import PerfTracker
from debug.debugshow import debugshow
import matplotlib.pyplot as plt
from numba import jit, njit
import uuid

DEFAULT_ROI = LaserDetection.DEFAULT_ROI
NUM_LASER_LINES = LaserDetection.NUM_LASER_LINES
mtx = ZedMini.LeftRectHD2K.K
DEBUG = ImageDisplay.DEBUG
DISP_COLORS = ImageDisplay.DISP_COLORS
DISP_COLORSf = ImageDisplay.DISP_COLORSf
MERGE_HLP_LINES_ANG_THRESH = LaserCalibration.MERGE_HLP_LINES_ANG_THRESH
MERGE_HLP_LINE_DIST_THRESH = LaserCalibration.MERGE_HLP_LINE_DIST_THRESH

DEFAULT_IMG_DATA = [
    ("calib_imgs/set1", 0.0224, (8,6), 1910.), 
        #("calib_imgs/set2", 0.02909, (6,9), 1151.), 
        #("calib_imgs/set3", 0.02909, (6,9), 1322.), 
    #("calib_imgs/set4", 0.02909, (6,9), 1387.)
]


@jit(forceobj=True)
def throw_out_outlier_clusters(img, gvals):
    '''Used to throw out clusters that are too far away from the center of 
    the image as a heuristic to determine if the cluster is a laser light or not.'''
    # K-means clustering on img to segment good points from bad points
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_PP_CENTERS
    pot_pts = np.float32(gvals[:,:2])
    # 3 clusters, no given labels, 10 attempts
    compactness, labels, centers = cv.kmeans(pot_pts, 3, None, criteria, 10, flags)
    A = pot_pts[labels.ravel()==0]
    B = pot_pts[labels.ravel()==1]
    C = pot_pts[labels.ravel()==2]
    clustering_img = img.copy()
    print("cluster centers")
    print(centers)
    for center in centers:
        x, y = int(center[1]), int(center[0])
        print("x, y: %d, %d" % (x,y))
        cv.circle(clustering_img, (x,y), 5, (0,255,0), cv.FILLED)

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
    print("ROI(minX,maxX,minY,maxY) = (%d, %d, %d, %d)" % (minx, maxx, miny, maxy))
    for idx, cluster in enumerate(clusters):
        for clusterptidx, val in enumerate(cluster):
            x, y = val
            x, y = int(x), int(y)
            cv.circle(clustering_img, (x, y), 2, DISP_COLORS[idx], cv.FILLED)
        center = centers[idx]
        if minx < center[0] < maxx and miny < center[1] < maxy:
            goodclusters.append(cluster)
            print("keeping cluster %d w/ center %f, %f" % (idx, center[0], center[1]))
    
    goodclustering_img = img.copy()
    for idx, cluster in enumerate(goodclusters):
        for clusterptidx, val in enumerate(cluster):
            x, y = val
            x, y = int(x), int(y)
            cv.circle(goodclustering_img, (x, y), 2, DISP_COLORS[idx], cv.FILLED)
    cv.rectangle(goodclustering_img, (minx, miny), (maxx, maxy), (255, 0, 0), 3)

    numgoodclusters = len(goodclusters)
    if numgoodclusters == 0:
        print("no good clusters, discarding img")
        return None
    
    oldnumgvals = gvals.shape[0]
    gvals = goodclusters[0]
    for i in range(1, numgoodclusters):
        gvals = np.append(gvals, goodclusters[i], axis=0)
    
    print("\nnum gvals went from %d to %d\n" % (oldnumgvals, gvals.shape[0]))
    # debugshow(clustering_img, "clustering")

    return gvals

@jit(forceobj=True)
def segmentation(img, orig):
    '''Segmentation of the laser points into NUM_LASER_LINES lines.'''
    lines = cv.HoughLines(img, 1, np.pi / 180, threshold=100, srn=2, stn=2) 
    if lines is not None: lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))
    else: 
        print("No hough lines!")
        return None

    # hlp_laser_img_disp = orig.copy()
    # print("\nLines: ")
    # if lines is not None:
    #     lines = lines[lines[:, 0].argsort()] 
    #     for i in range(0, len(lines)):
    #         print("lines %s" % lines[i])
    #         rho = lines[i][0]
    #         theta = lines[i][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
    #         pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
    #         cv.line(hlp_laser_img_disp, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    # debugshow(hlp_laser_img_disp, "houghlines")


    # cv.imshow("ulines", draw_polar_lines(orig.copy(), lines))
    mergedlines = mathutil.merge_polar_lines(
        lines, 
        MERGE_HLP_LINES_ANG_THRESH, 
        MERGE_HLP_LINE_DIST_THRESH, 
        NUM_LASER_LINES)

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

@jit(forceobj=True)
def imagept_laserplane_assoc(patches, polarlines):
    '''Associate each laser patch with a line.'''
    # associate each laser patch with a line
    # for more accuracy could calculate each patch's centroid or calculate an average distance from line of all points of a patch
    # for speed could just pick one point from patch, will likely be enough given circumstances
    # we throw out patches too far from any lines
    maxdistfromline = 50 # px
    patchgroups = [[0, []] for _ in range(NUM_LASER_LINES)]
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

#@jit(forceobj=True)
def calibrate(imgs: list[np.ndarray], square_size_m: float, chessboard_dims: tuple[int,int], min_gval, gpu=False):
    '''Extract laser projection planes from calibrated camera images of a parallel laser line pattern 
    projected coplanar to a calibration chessboard pattern whose interior dimensions and square size are provided.'''
    Pts3d = [[] for _ in range(NUM_LASER_LINES)] # 3D pts
    for imgidx, (filename, img) in enumerate(imgs):
        log_header(f"Processing image {imgidx}: {filename}")#logheader
        ####### Finding Chessboard #######

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

        gray = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY, dst=gray, dstCn=1)
        ret, corners = cv.findChessboardCorners(gray, chessboard_dims)

        if not ret:
            print(f"Couldn't find chessboard, discarding image {imgidx}...") # logwarn
            continue
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # find rotation and translation vectors
        ret, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

        rotmat, _ = cv.Rodrigues(rvec)
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
            rowmin = int(DEFAULT_ROI[0][0] * img.shape[0])
            rowmax = int(DEFAULT_ROI[1][0] * img.shape[0])+1
            colmin = int(DEFAULT_ROI[0][1] * img.shape[1])
            colmax = int(DEFAULT_ROI[1][1] * img.shape[1])+1
            roi_img = img[rowmin:rowmax,colmin:colmax]
            reward = color_reward.get_reward_gpu(roi_img)
            #reward = color_reward.get_reward_gpu(img)
            gvals = gval.calculate_gaussian_integral_windows_gpu(reward, min_gval).copy_to_host()
            #debugshow(gvals, "Gvals")
            subpxs = subpx.find_gval_subpixels_gpu(gvals, reward, min_gval)
            #debugshow(subpxs, "subpx")
            filtered, patches = pxpatch.throw_out_small_patches_gpu(subpxs)
        else: 
            reward = color_reward.get_reward(img)
            # debugshow(reward / np.max(reward), "rew")
            gvals = gval.calculate_gaussian_integral_windows(reward, min_gval)
            gvalimg = np.zeros(reward.shape, dtype=np.float)
            for col, row, val in gvals:
                gvalimg[int(row), int(col)] = val
            gvalimg /= np.max(gvalimg)
            # debugshow(gvalimg, "gval")
            gvals = throw_out_outlier_clusters(img, gvals)
            gvalimg = np.zeros(reward.shape, dtype=np.float)
            for g in gvals:
                col, row = int(g[0]), int(g[1]) 
                gvalimg[row, col] = 1.
            # debugshow(gvalimg, "filt")
            if gvals is None: continue
            subpxs = subpx.find_gval_subpixels(gvals, reward)
            dispsubpxs = np.zeros(subpxs.shape, dtype=np.float)
            dispsubpxs[subpxs != 0] = 1.
            # debugshow(dispsubpxs, "subpx")
            filtered, patches = pxpatch.throw_out_small_patches(subpxs)
            rowmin = 0
            colmin = 0


        laserpxbinary = np.zeros(filtered.shape, dtype=np.uint8)
        laserpxbinary[filtered != 0] = 255
        # debugshow(laserpxbinary, "Laser Pixels")
        # cv.waitKey(0)
        lines = segmentation(laserpxbinary, img)
        if lines is None: continue

        #mergedlinesimg = img.copy() if gpu else roi_img.copy()
        #print("\nMerged Lines")
        #for i in range(0, len(lines)):
        #    try:
        #        print("line %s" % lines[i])
        #        rho = lines[i][0]
        #        theta = lines[i][1]
        #        a = math.cos(theta)
        #        b = math.sin(theta)
        #        x0 = a * rho
        #        y0 = b * rho
        #        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        #        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        #        cv.line(mergedlinesimg, pt1, pt2, ImageDisplay.DISP_COLORS[i], 3, cv.LINE_AA)
        #    except:
        #        print("bad line (maybe vertical) %s" % lines[i])
        #debugshow(mergedlinesimg, "Merged Lines")
        #cv.waitKey(0)
        

        patchgroups = imagept_laserplane_assoc(patches, lines)
        

        ####### Extract 3D Points #######
        for idx, (numpts, group) in enumerate(patchgroups): 
            print("line %d has %d patches" % (idx, len(group)))
            for patch in group:
                for pt in patch:
                    row, col, x_offset = pt
                    pt3d = mathutil.px_2_3d(
                        row + rowmin, 
                        col + colmin + x_offset, 
                        chessboard_plane, 
                        mtx) #H.dot([col + x_offset, row, 1])
                    Pts3d[idx].append(pt3d)
                    # PtsImg[idx].append((col + x_offset, row))
        
        ####### Done Extracting 3D Points #######
    # /for imgidx, img in enumerate(imgs)
    

    ####### RANSAC Plane Extraction #######

    # RANSAC: form M subjects of k points from P
    planes = []
    for lineP in Pts3d:
        print(len(lineP))
        if len(lineP) == 0: 
            continue
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
    np.save("Camera_Relative_Laser_Planes_" + str(uuid.uuid4()), planes) 

    if DEBUG:
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
                            color=DISP_COLORSf[idx], edgecolor='none')
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x * 0.25, y * 0.25, z * 0.25, cmap=plt.cm.YlGnBu_r)
        plt.show()


def main(
        calibration_img_info: list[tuple[str, float, tuple[int,int], float]], 
        *, gpu=False, record_data=False):
    for img_folder, square_size_m, chessboard_dims, min_gval in calibration_img_info:
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
        calibrate(imgs, square_size_m, chessboard_dims, min_gval, gpu=gpu)


if __name__ == "__main__":
    numargs = len(sys.argv) - 1
    if numargs == 0:
        #main(DEFAULT_IMG_DATA, gpu=True, record_data=True)
        main(DEFAULT_IMG_DATA, gpu=False, record_data=True)
        PerfTracker.export_to_csv()
    elif numargs % 3 != 0:
        log_err(
            '''Incorrect number of command line arguments. Either pass none 
            to use default arguments or pass 3 arguments for each set of 
            calibration images in the following form: <relative_path_to_folder> 
            <calibration_chessboard_square_size_m> 
            <calibration_chessboard_interior_dimensions>''', 
            True, ValueError("Bad command line arguments"))
    else:
        pass