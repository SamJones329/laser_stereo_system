import rospy

import numpy as np
import typing
from random import randint
import math
from rospy import numpy_msg

from sensor_msgs.msg import Image

class ClusterizerNode(object):

  MAX_ITER = 5000

  def __init__(self):
    self.pub_cluster_img = rospy.Publisher("/stereo_kmeans/cluster_image", Image, queue_size=10)
    self.sub_points = rospy.Subscriber("/zedm/zed_node/depth/depth_registered", numpy_msg(Image), queue_size=10)

  def points_callback(self, msg: Image):
    pub_msg = Image()

    #convert to 2d array of pixels
    h = msg.height #rows
    w = msg.width #cols
    row_len_bytes = msg.step
    img = np.frombuffer(msg.data, dtype=np.uint8) # i think disparity pixels are 8-bit greyscale
    img: np.ndarray[typing.Any, np.dtype[np.uint8]] = img.reshape((h,w))

    # get depth range
    min_d = min(img)
    max_d = max(img)

    # initial centroid selection
    last_h = h-1
    last_w = w-1
    c1, c2, c3 = randint(min_d, max_d), randint(min_d, max_d), randint(min_d, max_d)

    # k-means
    success = False
    for i in range(self.MAX_ITER):
      s1, s2, s3 = [], [], []
      s1pts, s2pts, s3pts = [], [], []
      for r in range(h): #rows
        for c in range(w): #cols
          depth = img[r, c]
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
      if math.isclose(newc1, c1, rel_tol=1e-5)\
          and math.isclose(newc2, c2, rel_tol=1e-5)\
          and math.isclose(newc3, c3, rel_tol=1e-5):
        success = True
        break
      c1, c2, c3 = newc1, newc2, newc3
        
    if not success:
      rospy.logerr("Max iterations reached on k-means")
    
    newimg: np.ndarray = img.copy()
    for d, pts in [(c1, s1pts), (c2, s2pts), (c3, s3pts)]:
      for pt in pts:
        newimg[pt] = d

    pub_msg.data = newimg #check this
    pub_msg.height = h
    pub_msg.width = w
    self.pub_cluster_img.publish(pub_msg)

if __name__ == '__main__':

  rospy.init_node('clusterizer')
  node = ClusterizerNode()

  rospy.spin()