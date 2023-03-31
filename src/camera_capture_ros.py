#!/usr/bin/python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

bridge = CvBridge()

imagecount = 0
def save_img_callback(msg):
    # type:(Image) -> None
    global imagecount
    imagecount += 1
    print("Received image %d" % imagecount)
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        cv2.imwrite('img%04d.png' % imagecount, cv2_img)
def save_stereo_img_callback(msg):
    pass

def main():
    rospy.init_node('zedm_recorder')
    rospy.Subscriber('/zedm/zed_node/left/image_rect_color', Image, save_img_callback, queue_size=10)
    rospy.spin()

if __name__ == "__main__":
    main()