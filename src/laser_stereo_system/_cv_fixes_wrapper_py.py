from StringIO import StringIO

import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import Int64, Float64
from cv_bridge import CvBridge

from laser_stereo_system._cv_fixes_wrapper_cpp import CvFixesWrapper
from laser_stereo_system.msg import HoughLinesResult

bridge = CvBridge()
class CvFixes(object):
    def __init__(self):
        self._cv_fixes = CvFixesWrapper()

    def _to_cpp(self, msg):
        """Return a serialized string from a ROS message

        Parameters
        ----------
        - msg: a ROS message instance.
        """
        buf = StringIO()
        msg.serialize(buf)
        return buf.getvalue()

    def _from_cpp(self, str_msg, cls):
        """Return a ROS message from a serialized string

        Parameters
        ----------
        - str_msg: str, serialized message
        - cls: ROS message class, e.g. sensor_msgs.msg.LaserScan.
        """
        msg = cls()
        return msg.deserialize(str_msg)

    def HoughLinesFix(self, image, rho, theta, threshold, lines=None):
        # type: (cv.Mat, float, float, int, np.ndarray) -> np.ndarray
        pass
        if not isinstance(image, cv.Mat):
            rospy.ROSException('Argument (image) 1 is not a cv.Mat')
        if not isinstance(rho, float):
            rospy.ROSException('Argument (rho) 2 is not a float')
        if not isinstance(theta, float):
            rospy.ROSException('Argument 3 (theta) is not a float')
        if not isinstance(threshold, int):
            rospy.ROSException('Argument 4 (threshold) is not an int')
        if lines is not None:
            if not isinstance(lines, np.ndarray):
                rospy.ROSException('Argument 5 (lines) is not a np.ndarray')
            linesshape = lines.shape
            if len(linesshape) != 3 or linesshape[1] != 1 or linesshape[2] != 3:
                rospy.ROSException('Invalid shape for lines array, shape should be (X, 1, 3)')
        
        image_msg = bridge.cv2_to_imgmsg(image, encoding="passthrough")
        rho_msg = Float64(data=rho)
        theta_msg = Float64(data=theta)
        threshold_msg = Int64(data=threshold)


        image = self._to_cpp(image_msg)
        rho = self._to_cpp(rho_msg)
        theta = self._to_cpp(theta_msg)
        threshold = self._to_cpp(threshold_msg)
    
        lines_res = self._cv_fixes.HoughLinesFix(image, rho, theta, threshold, lines)
        lines_res = self._from_cpp(lines_res, HoughLinesResult)

        if lines is not None:
            for i in range(linesshape[0]):
                lines[i,0,:] = (lines_res[i].one, lines_res[i].two, lines_res[i].three)
            return lines
        else: 
            lines = []
            for i in range(linesshape[0]):
                lines.append([(lines_res[i].one, lines_res[i].two, lines_res[i].three)])
            return np.array(lines)

