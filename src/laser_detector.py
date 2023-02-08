import cupy as cp

# Constants, should move these to param yaml file if gonna use with ROS
WINLEN = 5 # works for 1080p and 2.2k for Zed mini

def generate_laser_reward_image(img, color_weights):
    # type:(cp.ndarray, tuple[float,float,float]) -> cp.ndarray
    '''Creates a greyscale image where each pixel's value represents the likely of the pixel being part of a laser.

    :param img: (cv.Mat) MxNx3 RGB image, assumes Blue-Green-Red (BGR) color order
    :param color_weights: (tuple[float,float,float]) Values to weight colors of 
    pixels to detect laser lines, in the order (B,G,R). In the conversion to greyscale, 
    determines how much of a color's value is added to the greyscale value. e.g. if 
    (B,G,R)=(0.1, 0.5, 0.2), for a pixel with BGR(128,255,50), the resulting greyscale 
    value would be 0.1 * 128 + 0.5 * 255 + 0.2 * 50 = 150.3 ≈ 150. These values are 
    usually obtained by cross-referencing the color response charts of your camera to 
    the wavelength of your laser light.

    :return: MxN greyscale cv.Mat image or otherwise compatible arraylike
    '''
    return img[:,:,0] * color_weights[0] + img[:,:,1] * color_weights[1] + img[:,:,2] * color_weights[2]

def generate_candidate_laser_pt_img(img):
    '''Takes a greyscale laser reward image and thresholds it to a binary image

    :param img: MxN greyscale cv.Mat image or otherwise compatible arraylike

    :return: MxN binary cv.Mat image or otherwise compatible arraylike
    '''
    # need to parallelize this for loop
    gvals = cp.zeros(img, dtype=cp.float32)
    edge_width = WINLEN // 2
    for i in range(edge_width, x.shape[1] - edge_width):
        z[:,i] = cp.sum(x[:, i:i+WINLEN], axis=1)

    # threshold

def imagept_laserplane_assoc(img, planes):
    # type:(cp.ndarray, list[tuple(float, float, float)]) -> list[list[tuple(float,float,float)]]
    '''
    Associates each laser points in the image with one of the provided laser planes or throws it out.

    :param img: MxN binary cv.Mat image or otherwise compatible arraylike
    :param planes: (list[tuple(float,float,float)]) Planes of light projecting from a laser equiped with a diffractive optical element (DOE) described by their normal vectors in the camera reference frame.

    :return: (list[list[tuple(float,float,float)]]) Image points organized plane, with the index of the plane in the planes array corresponding to the index of its member points in the returned array.
    '''
    pass

def extract_laser_points(img):
    # type:(cp.ndarray) -> list[tuple(float,float,float)]
    '''
    Finds 3D coordinates of laser points in an image

    img - OpenCV Mat or otherwise compatible arraylike
    '''
    pass


if __name__ == "__main__":
    # example
    x = cp.arange(10, dtype=cp.float32).reshape(2,5)
    y = cp.arange(5, dtype=cp.float32)
    parallel_sum = cp.ElementwiseKernel(
        'float32 x, float32 y',
        'float32 z',
        'z = x + y',
        'parallel_sum'
    )
    z = cp.empty((2,5), dtype=cp.float32)
    parallel_sum(x,y,z)


    

