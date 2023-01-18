#!/usr/bin/python

from cv2 import minAreaRect
import rospy
import numpy as np
import cv2 as cv
import os
from optparse import OptionParser
import math
from datetime import datetime as dt
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PolygonStamped, Point32
from visualization_msgs.msg import Marker, MarkerArray
import random
import tf.transformations
from jsk_recognition_msgs.msg import PolygonArray
###from laser_stereo_system import CvFixes
###cvf = CvFixes()

from helpers import *

DEBUG_LINES = True

DISP_COLORS = [ #BGR
    (255,0,0), # royal blue
    (0,255,0), # green
    (0,0,255), # brick red
    (255,255,0), # cyan
    (255,0,255), # magenta
    (0,255,255), # yellow
    (255,255,255), # white
    (180,0,0), # dark blue
    (0,180,0), # forest green
    (0,0,180), # crimson
    (180,180,0), # turquoise
    (180,0,180), # purple
    (0,180,180), # wheat
    (180,180,180), # gray
    (255,180,100), # cerulean
]

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




def calibrate(data, chessboard_interior_dimensions=(9,6), square_size_m=0.1):
    # type:(list[cv.Mat], tuple[int, int], float) -> None
    '''
    Extracts extrinsic parameters between calibrated 
    camera and horizontal lines laser light structure
    '''

    ptpub = rospy.Publisher('chessboard_pts', PointCloud2, queue_size=10)
    planepub = rospy.Publisher('laser_planes', PolygonArray, queue_size=10)
    planenormpub = rospy.Publisher('laser_plane_normals', MarkerArray, queue_size=10)

    s = chessboard_interior_dimensions
    n = 15 # number of laser lines
    P = [[] for _ in range(n)] # 3D pts
    for idx, frame in enumerate(data):
        print("processing frame %d with size" % (idx+1))
        print(frame.shape)
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
        # print("object points")
        # print(objp)

        # scale axes to be in appropriate coords as well
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) * square_size_m

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, s)

        if not ret:
            print("couldn't find chessboard, discarding...")
            continue
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # find rotation and translation vectors
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # project 3d pts to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist) # can project more pts with jac?

        img = frame.copy()
        img = draw(img, corners2, imgpts)

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
        I_L_img = I_L.copy()
        # TODO - figure out if should normalize I_L, but don't think it matter since are looking for a max
        # G_v_w = sum from v=v_0 to v_0 + l_w of (1 - 2*abs(v_0 - v + (l_w-1) / 2)) * I_L(u,v)
        rows = frame.shape[0]
        cols = frame.shape[1]
        winlen = max(3, cols * 5 // 1920) # framecols * 5px / 1080p, but we don't want window size <3px
        gvals = []
        for col in range(cols):
            maxwin = 0
            maxg = 0
            for winstart in range(rows-winlen):
                G = 0
                for row in range(winstart, winstart+winlen):
                    G += (1 - 2*abs(winstart - row + (winlen - 1) / 2)) * I_L[row,col] #idk if this last part is right
                gvals.append((col, winstart+winlen//2, G))
        
        gvals.sort(key=lambda x: x[2])
        num_lines = 15
        expectedgoodgvals = int(rows * num_lines * 1.6)#1.4) # room for plenty of outliers
        gvals = gvals[:expectedgoodgvals]
        gvals = np.array(gvals)
             
        potential_lines = frame.copy()
        for idx, val in enumerate(gvals):
            x = int(val[0])
            y = int(val[1])
            cv.circle(potential_lines, (x,y), 3, (0,0,255))


        # K-means clustering on img to segment good points from bad points
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv.KMEANS_PP_CENTERS
        pot_pts = np.float32(gvals[:,:2])
        # 3 clusters, no given labels, 10 attempts
        compactness, labels, centers = cv.kmeans(pot_pts, 3, None, criteria, 10, flags)
        A = pot_pts[labels.ravel()==0]
        B = pot_pts[labels.ravel()==1]
        C = pot_pts[labels.ravel()==2]
        clustering_img = frame.copy()
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
        img_center = (frame.shape[1]//2, frame.shape[0]//2)
        quarter_x = frame.shape[1]//4
        third_y = frame.shape[0]//3
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
        
        goodclustering_img = frame.copy()
        for idx, cluster in enumerate(goodclusters):
            for clusterptidx, val in enumerate(cluster):
                x, y = val
                x, y = int(x), int(y)
                cv.circle(goodclustering_img, (x, y), 2, DISP_COLORS[idx], cv.FILLED)
        cv.rectangle(goodclustering_img, (minx, miny), (maxx, maxy), (255, 0, 0), 3)

        numgoodclusters = len(goodclusters)
        if numgoodclusters == 0:
            print("no good clusters, discarding img")
            continue
        
        oldnumgvals = gvals.shape[0]
        gvals = goodclusters[0]
        for i in range(1, numgoodclusters):
            gvals = np.append(gvals, goodclusters[i], axis=0)
        
        print("\nnum gvals went from %d to %d\n" % (oldnumgvals, gvals.shape[0]))

        # figure out how to throw out outliers
        # maybe generate histogram and use to throw out some points

        # subpixel detection via Gaussian approximation
        # delta = 1/2 * ( ( ln(f(x-1)) - ln(f(x+1)) ) / ( ln(f(x-1)) - 2ln(f(x)) + ln(f(x+1)) ) )
        # f(x) = intensity value of particular row at pixel x
        # laser_subpixels = {}
        laser_subpixels = np.full(gray.shape, 0.0, dtype=np.float)
        laser_img = np.full(gray.shape, 0.0, dtype=np.float)
        badoffsets = 0
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
            if abs(subpixel_offset) > winlen//2:
                badoffsets += 1
            if x + subpixel_offset < 0 or x + subpixel_offset > laser_img.shape[1]: 
                continue
            laser_img[y,int(x+subpixel_offset)] = 1.0
            laser_subpixels[y,int(x+subpixel_offset)] = (subpixel_offset % 1) + 1e-5
        print("%d bad offsets" % badoffsets)
        

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

        laser_patch_img = np.zeros(gray.shape)
        numpts = 0
        for patch in patches:
            for val in patch:
                row, col, _ = val
                laser_patch_img[row, col] = 1.
                numpts += 1

        laser_img_8uc1 = np.uint8(laser_img * 255)
        laser_patch_img_8uc1 = np.uint8(laser_patch_img * 255)
        hlp_laser_img = laser_patch_img_8uc1.copy()
        hlp_laser_img_disp = frame.copy()

        print("Number of line patch member points: %d" % numpts)
        # lines = np.empty((15,1,3))
        # lines = np.empty((100,1,2))
        # lines = cv.HoughLines(laser_img_8uc1, 1, np.pi / 180, threshold=numpts//80)#, lines=lines) #threshold=100)#numpts//80)@numpts==8000 # TODO - determine this value dynamically
        lines = cv.HoughLines(laser_img_8uc1, 1, np.pi / 180, threshold=numpts//80, srn=2, stn=2)#, lines=lines) #threshold=100)#numpts//80)@numpts==8000 # TODO - determine this value dynamically
        # lines = np.array(lines)
        # lines = cvf.HoughLinesFix(laser_img_8uc1, 1, np.pi / 180, threshold=numpts//80)
        print(lines.shape)
        lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))
        print("\nLines: ")
        if lines is not None:
            lines = lines[lines[:, 0].argsort()] 
            for i in range(0, len(lines)):
                print("lines %s" % lines[i])
                rho = lines[i][0]
                theta = lines[i][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
                pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
                cv.line(hlp_laser_img_disp, pt1, pt2, (0,0,255), 3, cv.LINE_AA)



        # merge similar lines
        r_thresh = 20
        a_thresh = 3#math.pi / 8
        groups = [[[],[]] for _ in range(n)]
        groupavgs = np.ndarray((n,2))
        groupsmade = 0
        threwout = 0

        # throw out bad angles
        avg_angle = np.average(lines[:,1])
        newlines = []
        for polarline in lines:
            r, angle = polarline
            if abs(angle - avg_angle) < a_thresh:
                newlines.append(polarline)
        lines = np.array(newlines)

        for polarline in lines:
            r, a = polarline
            goodgroup = -1
            for idx, avg in enumerate(groupavgs):
                r_avg, a_avg = avg
                if abs(r-r_avg) < r_thresh: # the thetas will all likely be the same or very similar so we dont care to compare them
                    goodgroup = idx
                    break
            if goodgroup == -1:
                pass
                if groupsmade == n:
                    threwout += 1
                    # find best fit? throw out? not sure, will just throw out for now
                    continue
                else:
                    groups[groupsmade][0].append(r)
                    groups[groupsmade][1].append(a)
                    groupavgs[groupsmade,:] = polarline
                    groupsmade += 1
            else:
                groups[goodgroup][0].append(r)
                groups[goodgroup][1].append(a)
                r_avg = sum(groups[goodgroup][0]) / len(groups[goodgroup][0])
                a_avg = sum(groups[goodgroup][1]) / len(groups[goodgroup][1])
                groupavgs[goodgroup,:] = r_avg, a_avg
        print("Threw out %d lines" % threwout)
        # should probably check if thetas are all withing TODO
        # threshold and if lines are spaced consistently 
        # and throw out first line and repeat if so
        # it seems that hough lines outputs lines in descending 
        # order of votes, so perhaps throwing out the last line 
        # in the list may be a good solution. WARNING though, because 
        # this feature is undocumented and not guarenteed across all implementations
        mergedlines = groupavgs
        for idx, line in enumerate(mergedlines):
            r, th = line
            if r < 0:
                th = angle_wrap(th + math.pi)
                r *= -1
            mergedlines[idx,:] = r, th

        # sort by radius increasing 
        # this allows us to assume lines pts at 
        # corresponding indices in the 3D points
        # array point to the same line, as the lines
        # should always be oriented the same 
        # relative to increasing radius due to the 
        # camera and laser being relatively fixed (and ideally rigid)
        mergedlines = mergedlines[mergedlines[:, 0].argsort()] 

        mergedlinesimg = frame.copy()
        print("\nMerged Lines")
        for i in range(0, len(groupavgs)):
            try:
                print("line %s" % groupavgs[i])
                rho = groupavgs[i][0]
                theta = groupavgs[i][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
                pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
                cv.line(mergedlinesimg, pt1, pt2, DISP_COLORS[i], 3, cv.LINE_AA)
            except:
                print("bad line (maybe vertical) %s" % groupavgs[i])

        # associate each laser patch with a line
            # for more accuracy could calculate each patch's centroid or calculate an average distance from line of all points of a patch
            # for speed could just pick one point from patch, will likely be enough given circumstances
        patchgroups = [[] for _ in range(n)]
        for patch in patches:
            y, x, subpixel_offset_x = patch[0] # get point in patch
            x += subpixel_offset_x # add subpixel offset to x
            r_p = math.sqrt(x**2 + y**2) # get radius from origin to pt
            th_p = math.atan2(y, x) # get theta from +x axis to pt
            bestline = 0 # idx of best line
            minval = float('inf') # distance from point to best line
            for idx, line in enumerate(mergedlines):
                r, th = mergedlines[idx] # r, th for cur line
                d = abs(r - r_p * math.cos(th - th_p)) # distance from pt to cur line
                if d < minval: # if found shorter distance
                    minval = d
                    bestline = idx
            patchgroups[bestline].append(patch)
        
        print("\nPatch Groups: ")

        # just multiply img pts by calibration plane homography to get 3D pts
        H, mask = cv.findHomography(corners, objp)
        print("\nH:")
        for row in H: print(row)

        mergedlinespatchimg = frame.copy()
        mergedlinespatchimgclear = frame.copy()
        for idx, group in enumerate(patchgroups): 
            print("line %d has %d patches" % (idx, len(group)))
            for patch in group:
                for pt in patch:
                    row, col, x_offset = pt
                    pt3d = H.dot([col + x_offset, row, 1])
                    mergedlinespatchimg[row, col] = DISP_COLORS[idx]
                    cv.circle(mergedlinespatchimgclear, (col, row), 2, DISP_COLORS[idx])
                    P[idx].append(pt3d)

        ###### This is just for rviz ######
        pts = []
        for patch in patches:
            for pt in patch:
                # x, y
                # imgpt = np.reshape([pt[1] + pt[2], pt[0], 1], (3,1))
                imgpt = [pt[1] + pt[2], pt[0], 1]
                newpt = np.dot(H, imgpt)
                pts.append(newpt)
        pts = np.array(pts)

        h = Header()
        h.frame_id = "/world"
        h.stamp = rospy.Time.now()
        pc2msg = point_cloud2.create_cloud_xyz32(h, pts)
        ptpub.publish(pc2msg)

        if DEBUG_LINES:
            imgdisp = cv.resize(img, disp_size)
            cv.imshow('img', imgdisp)

            I_L_img = cv.resize(I_L, disp_size)
            I_L_img /= 255.0
            cv.imshow('reward', I_L_img)

            potential_lines = cv.resize(potential_lines, disp_size)
            cv.imshow("pot lines", potential_lines)

            clustering_img_disp = cv.resize(clustering_img, disp_size)
            cv.imshow("clustered pot pts", clustering_img_disp)

            good_clustering_img_disp = cv.resize(goodclustering_img, disp_size)
            cv.imshow("filtered clustered pot pts", good_clustering_img_disp)

            # potential_lines_filtered_disp = cv.resize(potential_lines_filtered, disp_size)
            # cv.imshow("pot lines (filtered)", potential_lines_filtered_disp)

            laser_disp_img = cv.resize(laser_img, disp_size)
            cv.imshow("laserimg", laser_disp_img)

            laser_patch_img_disp = cv.resize(laser_patch_img, disp_size)
            cv.imshow("laserpatchimg", laser_patch_img_disp)

            hlp_laser_img_disp = cv.resize(hlp_laser_img_disp, disp_size)
            cv.imshow("Detected Lines (in red) - Hough Lines", hlp_laser_img_disp)

            laser_patch_img_8uc1_disp = cv.resize(laser_patch_img_8uc1, disp_size)
            cv.imshow("laser_img - 8UC1", laser_patch_img_8uc1_disp)

            mergedlinesimg_disp = cv.resize(mergedlinesimg, disp_size)
            cv.imshow("Merged Lines", mergedlinesimg_disp)

            mergedlinespatchimg_disp = cv.resize(mergedlinespatchimg, disp_size)
            cv.imshow("Grouped Patches", mergedlinespatchimg_disp)

            mergedlinespatchimgclear_disp = cv.resize(mergedlinespatchimgclear, disp_size)
            cv.imshow("Grouped Patches (Enlarged Pts)", mergedlinespatchimgclear_disp)

            k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
                t = str(dt.now())
                cv.imwrite('pose' + t + '.png', img)
                cv.imwrite('laserreward' + t + '.png', I_L)
                cv.imwrite('laserlinepts' + t + '.png', potential_lines)
                cv.imwrite('lasersubpxpts' + t + '.png', laser_img * 255)
                cv.imwrite('laserpatches' + t + '.png', laser_patch_img * 255)
                cv.imwrite('detectedlines' + t + '.png', hlp_laser_img_disp)
                cv.imwrite('mergedlines' + t + '.png', mergedlinesimg)
                cv.imwrite('groupedpatches' + t + '.png', mergedlinespatchimg)
                cv.imwrite('groupedpatchesbig' + t + '.png', mergedlinespatchimgclear)
            elif k == ord('x'):
                return
            cv.destroyAllWindows()



    # publish all 3D points together
    ###### This is just for rviz ######
    pts = []
    for lineP in P:
        for pt in lineP:
            # x, y
            # imgpt = np.reshape([pt[1] + pt[2], pt[0], 1], (3,1))
            imgpt = [pt[1], pt[0], 1]
            newpt = np.dot(H, imgpt)
            pts.append(newpt)
    pts = np.array(pts)
    
    h = Header()
    h.frame_id = "/world"
    h.stamp = rospy.Time.now()
    pc2msg = point_cloud2.create_cloud_xyz32(h, pts)
    ptpub.publish(pc2msg)

    # RANSAC: form M subjects of k points from P
    planes = []
    for lineP in P:
        potplanes = []

        M = 3
        npLineP = np.array(lineP) # type: np.ndarray

        # print("npLineP shape")
        # print(npLineP.shape)

        shuffled = npLineP.copy()
        np.random.shuffle(shuffled)
        subsets = np.array_split(shuffled, M)
        # print("subshape")
        # print(len(subsets))
        # for arr in subsets: print(arr.shape)
        print(len(subsets))
        for idx, subset in enumerate(subsets): 
            print("subset %d" % idx)
            print(subset.shape)
        for subset in subsets:
            print(subset.shape)
            # compute centroid c_j of p_j
            c = np.average(subset[:,0]), np.average(subset[:,1]), np.average(subset[:,2]) # x, y, z

            # subtract centroid c_j to all points P
            planeRefPts = np.array(lineP)
            planeRefPts[:,0] -= c[0]
            planeRefPts[:,1] -= c[1]
            planeRefPts[:,2] -= c[2]
            # print("planerefshape")
            # print(planeRefPts.shape) # (..., 3, 1)
            # planeRefPts = np.array(lineP) - c

            # use SVD to find the plane normal n_j
            # i think this is third third column vector (b/c 3D space -> 3x3 V mat) of V
            w, u, vt = cv.SVDecomp(planeRefPts)
            n = vt[2,:] # type: np.ndarray
            # print(n.shape) # (3,)
            # define pi_j,L : (n_j, c_j)

            # compute distance su d_j of all points P to the plane pi_j,L
            # this distance is the dot product of the vector from the centroid to the point with the normal vector
            distsum = planeRefPts.dot(n).sum()
            potplanes.append((c, vt, distsum))

        bestplane = potplanes[0]
        for plane in potplanes:
            if plane[2] < bestplane[2]:
                bestplane = plane
        planes.append(plane[:2]) # we don't need the distance anymore
        # return plane that fits most points/minimizes distance d_j

    # make planes polygons for rviz
    planemsgs = PolygonArray()
    planenormmsgs = MarkerArray()
    for idx, plane in enumerate(planes):
        t = rospy.Time.now()
        fid = "/world"
        planepolygon = PolygonStamped()
        planepolygon.header.frame_id = fid
        planepolygon.header.stamp = t

        norm = Marker()
        norm.header.frame_id = fid
        norm.header.stamp = t

        c = plane[0]
        vecs = plane[1]
        n = vecs[2]

        p1 = Point32()
        p1.x = c[0] + vecs[0,0] + vecs[1,0]
        p1.y = c[1] + vecs[0,1] + vecs[1,1]
        p1.z = c[2] + vecs[0,2] + vecs[1,2]
        planepolygon.polygon.points.append(p1)

        p2 = Point32()
        p2.x = c[0] + vecs[0,0] - vecs[1,0]
        p2.y = c[1] + vecs[0,1] - vecs[1,1]
        p2.z = c[2] + vecs[0,2] - vecs[1,2]
        planepolygon.polygon.points.append(p2)
        
        p3 = Point32()
        p3.x = c[0] -vecs[0,0] - vecs[1,0]
        p3.y = c[1] -vecs[0,1] - vecs[1,1]
        p3.z = c[2] -vecs[0,2] - vecs[1,2]
        planepolygon.polygon.points.append(p3)
        
        p4 = Point32()
        p4.x = c[0] -vecs[0,0] + vecs[1,0]
        p4.y = c[1] -vecs[0,1] + vecs[1,1]
        p4.z = c[2] -vecs[0,2] + vecs[1,2]
        planepolygon.polygon.points.append(p4)
        
        norm.type = Marker.ARROW
        norm.color.b = 0.0
        norm.color.g = 1.0
        norm.color.r = 0.0
        norm.color.a = 0.8
        norm.scale.x = 0.1
        norm.scale.y = 0.005
        norm.scale.z = 0.005
        norm.id = idx
        norm.pose.position.x = c[0]
        norm.pose.position.y = c[1]
        norm.pose.position.z = c[2]
        roll = 0
        pitch = 2*math.pi - math.atan2(n[2], n[0])
        yaw = math.atan2(n[1], n[0])
        nq = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        norm.pose.orientation.x = nq[0]
        norm.pose.orientation.y = nq[1]
        norm.pose.orientation.z = nq[2]
        norm.pose.orientation.w = nq[3]

        planemsgs.polygons.append(planepolygon)
        planenormmsgs.markers.append(norm)

    planepub.publish(planemsgs)
    planenormpub.publish(planenormmsgs)
        # could maybe just use information better to extract a homography for the plane instead of the plane normal and stuff


if __name__ == "__main__":
    rospy.init_node('calibrate_laser')

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
# 1 good, 2 bad, 3 meh, 4 meh, 5 bad, 6 meh, 7 meh, 8 meh, 9 meh, 10 meh
# lines are being well detected (except some extra outliers in one image) but are not grouped well
# maybe need to take less extra points
# also maybe need larger integral window because of higher resolution
# i think i may be extracting the 3d coords of the points wrong? maybe i just need more distance variation?
# also might be taking the wrong vector as the plane normal vector? not sure