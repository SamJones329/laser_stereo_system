import numpy as np
import cv2 as cv
import math

DISP_COLORS = [ #BGR
    (255,0,0), # royal blue
    (0,255,0), # green
    (0,0,255), # brick red
    (255,255,0), # cyan
    (255,0,255), # magenta
    (0,255,255), # yellow
    (255,160,122), # light salmon
    (180,0,0), # dark blue
    (0,180,0), # forest green
    (0,0,180), # crimson
    (180,180,0), # turquoise
    (180,0,180), # purple
    (0,180,180), # wheat
    (180,180,180), # gray
    (255,180,100), # cerulean
]

DISP_COLORSf = [ #BGR
    (1,0,0), # royal blue
    (0,1,0), # green
    (0,0,1), # brick red
    (1,1,0), # cyan
    (1,0,1), # magenta
    (0,1,1), # yellow
    (1,.627,.478), # white
    (.7,0,0), # dark blue
    (0,.7,0), # forest green
    (0,0,.7), # crimson
    (.7,.7,0), # turquoise
    (.7,0,.7), # purple
    (0,.7,.7), # wheat
    (.7,.7,.7), # gray
    (1,.7,.4), # cerulean
]

# modifies in place and returns
def calc_chessboard_corners(board_size, square_size):
    # type:(tuple[int, int], float) -> list[tuple[float, float, float]]
    corners = []
    for i in range(board_size[0]): # height
        for j in range(board_size[1]): # width
            corners.append((j*square_size, i*square_size,0))
    return corners

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def recurse_patch(row, col, patch, img, onlyCheckImmediateNeighbors=True):
    # type:(int, int, set, cv.Mat, bool) -> None
    if onlyCheckImmediateNeighbors:
      # check neighbors
      up = row-1
      if up >= 0 and img[up, col] > 1e-6: # up
          patch.append((up, col, img[up,col]))
          img[up,col] = 0.
          recurse_patch(up, col, patch, img)

      down = row+1
      if down <= img.shape[0] and img[down, col] > 1e-6: # down
          patch.append((down, col, img[down,col]))
          img[down,col] = 0.
          recurse_patch(down, col, patch, img)

      left = col-1
      if left >= 0 and img[row, left] > 1e-6: # left
          patch.append((row, left, img[row,left]))
          img[row,left] = 0.
          recurse_patch(row, left, patch, img)

      right = col+1
      if right <= img.shape[1] and img[row, right] > 1e-6: # right
          patch.append((row, right, img[row,right]))
          img[row,right] = 0.
          recurse_patch(row, right, patch, img)
    else:
      # we define contiguity by being within 3 pixels of the source pixel
      # therefore there is a 7x7 box around the original pixel in which to search for pixels
      for i in range(-3,4): # [-3, -2, -1, 0, 2, 3]
          searchingrow = row + i
          for j in range(-3,4):
              searchingcol = col + j
              if searchingrow > 0 and searchingrow <= img.shape[0] and searchingcol > 0 and searchingcol <= img.shape[1]:   
                  if img[searchingrow, searchingcol] > 1e-6:
                      patch.append((searchingrow,searchingcol))
                      img[searchingrow, searchingcol] = 0
                      recurse_patch(searchingrow, searchingcol, patch, img, False)
         

    

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

def draw_polar_lines(img, lines):
    for i in range(0, len(lines)):
        try:
            print("line %s" % lines[i])
            rho = lines[i][0]
            theta = lines[i][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            cv.line(img, pt1, pt2, DISP_COLORS[i % len(DISP_COLORS)], 3, cv.LINE_AA)
        except:
            print("bad line (maybe vertical) %s" % lines[i])
    return img

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

################ MST Geeks For Geeks ######################
# Python program for the above algorithm
import sys
V = 5;

# Function to find index of max-weight
# vertex from set of unvisited vertices
def findMaxVertex(visited, weights):

	# Stores the index of max-weight vertex
	# from set of unvisited vertices
	index = -1;

	# Stores the maximum weight from
	# the set of unvisited vertices
	maxW = -sys.maxsize;

	# Iterate over all possible
	# Nodes of a graph
	for i in range(V):

		# If the current Node is unvisited
		# and weight of current vertex is
		# greater than maxW
		if (visited[i] == False and weights[i] > maxW):
		
			# Update maxW
			maxW = weights[i];

			# Update index
			index = i;
	return index;

# Utility function to find the maximum
# spanning tree of graph
def printMaximumSpanningTree(graph, parent):

	# Stores total weight of
	# maximum spanning tree
	# of a graph
	MST = 0;

	# Iterate over all possible Nodes
	# of a graph
	for i in range(1, V):
	
		# Update MST
		MST += graph[i][parent[i]];

	print("Weight of the maximum Spanning-tree ", MST);
	print();
	print("Edges \tWeight");

	# Print Edges and weight of
	# maximum spanning tree of a graph
	for i in range(1, V):
		print(parent[i] , " - " , i , " \t" , graph[i][parent[i]]);

# Function to find the maximum spanning tree
def maximumSpanningTree(graph):

	# visited[i]:Check if vertex i
	# is visited or not
	visited = [True]*V;

	# weights[i]: Stores maximum weight of
	# graph to connect an edge with i
	weights = [0]*V;

	# parent[i]: Stores the parent Node
	# of vertex i
	parent = [0]*V;

	# Initialize weights as -INFINITE,
	# and visited of a Node as False
	for i in range(V):
		visited[i] = False;
		weights[i] = -sys.maxsize;

	# Include 1st vertex in
	# maximum spanning tree
	weights[0] = sys.maxsize;
	parent[0] = -1;

	# Search for other (V-1) vertices
	# and build a tree
	for i in range(V - 1):

		# Stores index of max-weight vertex
		# from a set of unvisited vertex
		maxVertexIndex = findMaxVertex(visited, weights);

		# Mark that vertex as visited
		visited[maxVertexIndex] = True;

		# Update adjacent vertices of
		# the current visited vertex
		for j in range(V):

			# If there is an edge between j
			# and current visited vertex and
			# also j is unvisited vertex
			if (graph[j][maxVertexIndex] != 0 and visited[j] == False):

				# If graph[v][x] is
				# greater than weight[v]
				if (graph[j][maxVertexIndex] > weights[j]):
				
					# Update weights[j]
					weights[j] = graph[j][maxVertexIndex];

					# Update parent[j]
					parent[j] = maxVertexIndex;

	# Print maximum spanning tree
	printMaximumSpanningTree(graph, parent);

# Driver Code
if __name__ == '__main__':
	# Given graph
	graph = [[0, 2, 0, 6, 0], [2, 0, 3, 8, 5], [0, 3, 0, 0, 7], [6, 8, 0, 0, 9],
																[0, 5, 7, 9, 0]];

	# Function call
	maximumSpanningTree(graph);

	# This code is contributed by 29AjayKumar
