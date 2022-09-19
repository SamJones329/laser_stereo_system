import rospy

import numpy as np
import typing
from random import randint

from sensor_msgs.msg import Image

class ClusterizerNode(object):

  def __init__(self):
    self.pub_cluster_img = rospy.Publisher("/stereo_kmeans/cluster_image", Image, queue_size=10)
    self.sub_points = rospy.Subscriber("/zedm/zed_node/depth/depth_registered", Image, queue_size=10)

  def points_callback(self, msg: Image):
    pub_msg = Image()

    #convert to 2d array of pixels
    h = msg.height #rows
    w = msg.width #cols
    row_len_bytes = msg.step
    img = np.frombuffer(msg.data, dtype=np.uint8) # i think disparity pixels are 8-bit greyscale
    img: np.ndarray[typing.Any, np.dtype[np.uint8]] = img.reshape((h,w))

    # initial centroid selection
    last_h = h-1
    last_w = w-1
    c1, c2, c3 = (randint(0,last_h), randint(0,last_w)), (randint(0,last_h), randint(0,last_w)), (randint(0,last_h), randint(0,last_w))

    # k-means
    for r in range(h):
      for c in range(w):
        pt = (r,c)
        

    self.pub_cluster_img.publish(pub_msg)

if __name__ == '__main__':

  rospy.init_node('clusterizer')
  node = ClusterizerNode()

  rospy.spin()