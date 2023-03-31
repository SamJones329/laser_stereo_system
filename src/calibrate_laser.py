#!/usr/bin/python

# TODO - add visualization parameter and don't import ros stuff when set to false

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
from jsk_recognition_msgs.msg import PolygonArray

from camera_info import ZedMini
from helpers import *

def verify_planes(planes, imgpoints):
    linepts = np.empty((0,3))
    for i, planeimgpoints in enumerate(imgpoints):
        pts3d = np.empty((len(planeimgpoints),3))
        for j, pt in enumerate(planeimgpoints):
            pts3d[j] = px_2_3d(pt[1], pt[0], planes[i], ZedMini.LeftRectHD2K.K)
        linepts = np.append(linepts, pts3d, axis=0)
    return linepts

DEBUG_LINES = False
USE_PREV_DATA = True
# https://stackoverflow.com/questions/53591350/plane-fit-of-3d-points-with-singular-value-decomposition

def calibrate(data, chessboard_interior_dimensions=(9,6), square_size_m=0.1):
    # type:(list[cv.Mat], tuple[int, int], float) -> None
    '''
    Extracts extrinsic parameters between calibrated 
    camera and horizontal lines laser light structure
    '''

    ptpub = rospy.Publisher('chessboard_pts', PointCloud2, queue_size=10)
    planepub = rospy.Publisher('laser_planes', PolygonArray, queue_size=10)

    s = chessboard_interior_dimensions
    n = 15 # number of laser lines
    Pts3d = [[] for _ in range(n)] # 3D pts
    PtsImg = [[] for _ in range(n)] # Img pts
    for frameidx, frame in enumerate(data):
        if USE_PREV_DATA: break
        print("processing frame %d with size" % (frameidx+1))
        print(frame.shape)
        # scale down display images to have 500 px height and preserve aspect ratio
        disp_size = ( int( frame.shape[1] * (500./frame.shape[0]) ), 500 )

        # ==== Find chessboard plane homography ====
        # https://www.youtube.com/watch?v=US9p9CL9Ywg
        mtx = ZedMini.LeftRectHD2K.K
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
        ret, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

        rotmat, jac = cv.Rodrigues(rvec)
        # world (chessboard) reference frame to camera reference frame transformation matrix
        world2cam = np.identity(4)
        world2cam[:3,:3] = rotmat
        world2cam[:3, 3] = tvec.flatten()
        print("Chessboard -> Cam Transformation Mat:")
        for row in world2cam: print(row)

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
        print("Chessboard plane", chessboard_plane)

        # project 3d pts to image plane
        imgpts, jac = cv.projectPoints(axis, rvec, tvec, mtx, dist) # can project more pts with jac?

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
        I_L = k_b * frame[:,:,0] + k_g * frame[:,:,1] + k_r * frame[:,:,2]
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
        lines = cv.HoughLines(laser_img_8uc1, 1, np.pi / 180, threshold=numpts//80, srn=2, stn=2)#, lines=lines) #threshold=100)#numpts//80)@numpts==8000 # TODO - determine this value dynamically
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
        groupavgs = np.zeros((n,2))
        groupsmade = 0
        threwout = 0

        # throw out bad angles
        # avg_angle = np.average(lines[:,1])
        med_angle = np.median(lines[:,1])
        newlines = []
        for polarline in lines:
            r, angle = polarline
            if abs(angle - med_angle) < a_thresh:
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
            # we throw out patches too far from any lines
        maxdistfromline = 50 # px
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
            if minval < maxdistfromline:
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
                    pt3d = px_2_3d(row, col+x_offset, chessboard_plane, ZedMini.LeftRectHD2K.K) #H.dot([col + x_offset, row, 1])
                    mergedlinespatchimg[row, col] = DISP_COLORS[idx]
                    cv.circle(mergedlinespatchimgclear, (col, row), 2, DISP_COLORS[idx])
                    Pts3d[idx].append(pt3d)
                    PtsImg[idx].append((col + x_offset, row))

        ###### This is just for rviz ######
        pts = []
        for patch in patches:
            for pt in patch:
                # x, y
                # imgpt = np.reshape([pt[1] + pt[2], pt[0], 1], (3,1))
                imgpt = [pt[1] + pt[2], pt[0], 1]
                newpt = px_2_3d(imgpt[1], imgpt[0], chessboard_plane, ZedMini.LeftRectHD2K.K)
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

    if USE_PREV_DATA:
        try:
            Pts3d = np.load("calibpts.npy", allow_pickle=True) # reads from ~/.ros
            PtsImg = np.load("calibimgpts.npy", allow_pickle=True) # reads from ~/.ros
        except:
            print("error retrieving previous data, exiting...")
            return
    else:
        np.save("calibpts", np.array(Pts3d)) # saves points in ~/.ros
        np.save("calibimgpts", np.array(PtsImg)) # saves points in ~/.ros

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
            subsetT = subset.T
            subsetRelative = subset - centroid                         #points relative to centroid
            # xyzRT            = np.transpose(xyzR)                       

            #2. calculate the singular value decomposition of the xyzT matrix and get the normal as the last column of u matrix
            u, s, v = np.linalg.svd(subsetRelative)
            print("svd shapes u, sigma, v")
            print(u.shape)
            print(s.shape)
            print(v.shape)
            normal = v[2]
            normal = normal / np.linalg.norm(normal)       #we want normal vectors normalized to unity

            # compute distance su d_j of all points P to the plane pi_j,L
            # this distance is the dot product of the vector from the centroid to the point with the normal vector
            distsum = subsetRelative.dot(normal).sum()
            potplanes.append((centroid, normal, distsum))

        bestplane = potplanes[0]
        for plane in potplanes:
            if plane[2] < bestplane[2]:
                bestplane = plane
        planes.append(plane[:2]) # we don't need the distance anymore
        # return plane that fits most points/minimizes distance d_j

    # make planes polygons for rviz
    planemsgs = PolygonArray()
    planemsgs.header.frame_id = "/world"
    planemsgs.header.stamp = rospy.Time.now()
    planehomogs = []
    stdplanes = np.empty((n,4))
    for idx, plane in enumerate(planes):
        t = rospy.Time.now()
        fid = "/world"
        planepolygon = PolygonStamped()
        planepolygon.header.frame_id = fid
        planepolygon.header.stamp = t

        centroid, normal = plane

        # have to append points in right order to get surface
        # centroid = Q = (a,b,c)
        # normal = n = <A,B,C>
        # A(x-a) + B(y-b) + C(z-c) = 0
        # normal[0] * (x - centroid[0]) + normal[1] * (y - centroid[1]) + normal[2] * (z - centroid[2])
        # D = Aa + Bb + Cc
        A, B, C = normal
        D = np.dot(normal, centroid)
        stdplanes[idx] = A, B, C, -D
        print("\nPlane ABCD")
        print(A,B,C,-D)

        # Ax - Aa + By - Bb - Cc = 0
        # Ax + By = Aa + Bb + Cc
        # y = (D - Ax) / B
        p1 = Point32()
        p1.x = centroid[0] + 0.3
        p1.z = centroid[2] - 0.3
        p1.y = (D - A*p1.x - C*p1.z) / B

        p2 = Point32()
        p2.x = centroid[0] - 0.3
        p2.z = centroid[2] - 0.3
        p2.y = (D - A*p2.x - C*p2.z) / B
        
        # y = (D - Ax - Cz) / B
        p3 = Point32()
        p3.x = centroid[0] + 0.3
        p3.z = centroid[2] + 0.3
        p3.y = (D - A*p3.x - C*p3.z) / B
        
        p4 = Point32()
        p4.x = centroid[0] - 0.3
        p4.z = centroid[2] + 0.3
        p4.y = (D - A*p4.x - C*p4.z) / B
        
        planepolygon.polygon.points.append(p1)
        planepolygon.polygon.points.append(p2)
        planepolygon.polygon.points.append(p4)
        planepolygon.polygon.points.append(p3)

        print("\nadding polygon")
        print("P1: (%f,%f,%f)" % (p1.x, p1.y, p1.z))
        print("P2: (%f,%f,%f)" % (p2.x, p2.y, p2.z))
        print("P3: (%f,%f,%f)" % (p3.x, p3.y, p3.z))
        print("P4: (%f,%f,%f)" % (p4.x, p4.y, p4.z))
        planemsgs.likelihood.append(1)
        planemsgs.polygons.append(planepolygon)

    # save planes in file in <a,b,c,A,B,C> format where centroid is (a,b,c) and normal is (A,B,C)
    planes = np.array(planes)
    planes = np.reshape(planes, (planes.shape[0], 6))
    print("Planes (centroid, normal)=<X,Y,Z,U,V,W>")
    print(planes)
    np.save("Camera_Relative_Laser_Planes_" + str(dt.now()), planes) 
        
    verify_pts = verify_planes(stdplanes, PtsImg)
    print("verify_pts.shape: ", verify_pts.shape)
    h = Header()
    h.frame_id = "/world"
    h.stamp = rospy.Time.now()
    pc2msg = point_cloud2.create_cloud_xyz32(h, verify_pts)
    ptpub.publish(pc2msg)
    
    planepub.publish(planemsgs)
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