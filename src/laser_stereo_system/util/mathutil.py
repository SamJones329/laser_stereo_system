import numpy as np
import math

def angle_wrap(ang):
    """
    Return the angle normalized between [-pi, pi].

    Works with numbers and numpy arrays.

    :param ang: the input angle/s.
    :type ang: float, numpy.ndarray
    :returns: angle normalized between [-pi, pi].
    :rtype: float, numpy.ndarray
    """
    ang = ang % (2 * np.pi)
    if (isinstance(ang, int) or isinstance(ang, float)) and (ang > np.pi):
        ang -= 2 * np.pi
    elif isinstance(ang, np.ndarray):
        ang[ang > np.pi] -= 2 * np.pi
    return ang

def get_polar_line(line, odom=[0.0, 0.0, 0.0]):
    """
    Transform a line from cartesian to polar coordinates.

    Transforms a line from [x1 y1 x2 y2] from the world frame to the
    vehicle frame using odomotrey [x y ang].

    By default only transforms line to polar without translation.

    :param numpy.ndarray line: line as [x1 y1 x2 y2].
    :param list odom: the origin of the frame as [x y ang].
    :returns: the polar line as [range theta].
    :rtype: :py:obj:`numpy.ndarray`
    """
    # Line points
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    # Compute line (a, b, c) and range
    line = np.array([y1-y2, x2-x1, x1*y2-x2*y1])
    pt = np.array([odom[0], odom[1], 1])
    dist = np.dot(pt, line) / np.linalg.norm(line[:2])

    # print(np.shape(dist))
    # Compute angle
    if dist < 0:
        ang = np.arctan2(line[1], line[0])
    else:
        ang = np.arctan2(-line[1], -line[0])

    # Return in the vehicle frame
    return np.array([np.abs(dist), angle_wrap(ang - odom[2])])

def merge_polar_lines(lines, a_thresh, r_thresh, max_lines, debug=False):
    groups = [[[],[]] for _ in range(max_lines)]
    groupavgs = np.zeros((max_lines,2))
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
            if groupsmade == max_lines:
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
    if debug:
         print("Merge Polar Lines made %d groups and threw out %d lines" % (groupsmade, threwout))
    return groupavgs

def segments_distance(x11, y11, x12, y12, x21, y21, x22, y22):
  """ distance between two segments in the plane:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
  """
  if segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22): return 0
  # try each of the 4 vertices w/the other segment
  distances = []
  distances.append(point_segment_distance(x11, y11, x21, y21, x22, y22))
  distances.append(point_segment_distance(x12, y12, x21, y21, x22, y22))
  distances.append(point_segment_distance(x21, y21, x11, y11, x12, y12))
  distances.append(point_segment_distance(x22, y22, x11, y11, x12, y12))
  return min(distances)

def segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22):
  """ whether two segments in the plane intersect:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
  """
  dx1 = x12 - x11
  dy1 = y12 - y11
  dx2 = x22 - x21
  dy2 = y22 - y21
  delta = dx2 * dy1 - dy2 * dx1
  if delta == 0: return False  # parallel segments
  s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
  t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / (-delta)
  return (0 <= s <= 1) and (0 <= t <= 1)

def point_segment_distance(px, py, x1, y1, x2, y2):
  dx = x2 - x1
  dy = y2 - y1
  if dx == dy == 0:  # the segment's just a point
    return math.hypot(px - x1, py - y1)

  # Calculate the t that minimizes the distance.
  t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

  # See if this represents one of the segment's
  # end points or a point in the middle.
  if t < 0:
    dx = px - x1
    dy = py - y1
  elif t > 1:
    dx = px - x2
    dy = py - y2
  else:
    near_x = x1 + t * dx
    near_y = y1 + t * dy
    dx = px - near_x
    dy = py - near_y

  return math.hypot(dx, dy)

def px_2_3d(row, col, plane, K):
    '''Plane as (A,B,C,D). K as instrinsic camera matrix.'''
    a, b, c, d = plane
    c_x, c_y, f_x, f_y = K[0,2], K[1,2], K[0,0], K[1,1]
    if f_x == 0 or f_y == 0: print("px_2_3d warning: focal length of zero found")
    x = (col - c_x) / f_x
    y = (row - c_y) / f_y
    t_denom = (a * x + b * y + c)
    if t_denom == 0: print("px_2_3d warning: division by 0")
    t = - d / t_denom
    x *= t
    y *= t
    z = t
    return x, y, z