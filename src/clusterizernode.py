#!/usr/bin/python

import rospy

import numpy as np
from random import uniform
import math
from cv_bridge import CvBridge
import cv2

from sensor_msgs.msg import Image

def isclose(a, b, rel_tol=1e-09, abs_tol=1e-03):
  if a is float('nan') or b is float('nan'): return False
  abs_a = abs(a)
  abs_b = abs(b)
  if (a == float('inf') or a == float('-inf')) \
      and (b == float('inf') or b == float('-inf')):
    return True
  if a == 0:
    return abs_b < rel_tol and abs_b < abs_tol
  if b == 0: 
    return abs_a < rel_tol and abs_a < abs_tol
  abs_ab = abs(a - b)
  return abs_ab / min(a, b) < rel_tol and abs_ab < abs_tol

class ClusterizerNode(object):

  MAX_ITER = 50

  def __init__(self):
    self.pub_cluster_img = rospy.Publisher("/stereo_kmeans/cluster_image", Image, queue_size=10)
    self.sub_points = rospy.Subscriber("/zedm/zed_node/depth/depth_registered", Image, callback=self.points_callback, queue_size=10)
    self.cvBridge = CvBridge()

  def points_callback(self, msg):
    # type:(Image) -> None
    print("image received")
    # cvimg = self.cvBridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # cvimg = cv2.patchNaNs(cvimg)
    print(msg.encoding, msg.height, msg.width, msg.step)  

    #convert to 2d array of pixels
    h = msg.height #rows
    w = msg.width #cols
    img = np.frombuffer(msg.data, dtype=np.float32) # disparity pixels are 32bit floats
    img = img.reshape((h,w)) # type: np.ndarray[any, np.dtype[np.uint8]]

    # get depth range
    min_d = float('inf')
    max_d = 0
    for i in range(h):
      for j in range(w):
        d = img[i, j]
        if d < min_d and d > 0: min_d = d
        if d > max_d and d != float('inf'): max_d = d

    print('min', min_d, 'max', max_d)

    # initial centroid selection
    last_h = h-1
    last_w = w-1
    c1, c2, c3 = uniform(min_d, max_d), uniform(min_d, max_d), uniform(min_d, max_d)

    # k-means
    success = False
    for i in range(self.MAX_ITER):
      s1, s2, s3 = [], [], []
      s1pts, s2pts, s3pts = [], [], []
      for r in range(h): #rows
        for c in range(w): #cols
          depth = img[r, c]
          if isclose(depth, 0, rel_tol=1e-5) or abs(depth) == float('inf'): continue
          diff1 = abs(depth-c1)
          diff2 = abs(depth-c2)
          diff3 = abs(depth-c3)
          diffs = [diff1, diff2, diff3]
          mindiff = min(diffs)
          if mindiff == diff1:
            s1.append(depth)
            s1pts.append((r, c))
          elif mindiff == diff2:
            s2.append(depth)
            s2pts.append((r, c))
          else: 
            s3.append(depth)
            s3pts.append((r, c))
      newc1 = sum(s1) / len(s1)
      newc2 = sum(s2) / len(s2)
      newc3 = sum(s3) / len(s3)
      if isclose(newc1, c1, rel_tol=1e-5)\
          and isclose(newc2, c2, rel_tol=1e-5)\
          and isclose(newc3, c3, rel_tol=1e-5):
        success = True
        break
      c1, c2, c3 = newc1, newc2, newc3

    print("Iterations done: %d" % i)    
    if not success:
      rospy.logerr("Max iterations reached on k-means")
    
    newimg = img.copy() # type: np.ndarray
    for d, pts in [(c1, s1pts), (c2, s2pts), (c3, s3pts)]:
      for pt in pts:
        newimg[pt] = d

    print("publishing")
    self.pub_cluster_img.publish(self.cvBridge.cv2_to_imgmsg(newimg, encoding="passthrough"))

if __name__ == '__main__':

  rospy.init_node('clusterizer')
  node = ClusterizerNode()

  rospy.spin()