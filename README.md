# laser_stereo_system
Undergraduate thesis work on underwater active stereo.

# Calibration [WIP]
`calibrate_laser.py` in the `src` directory contains a script for extracting extrinsic parameters of the laser relative to a given camera with known instric parameters (camera matrix and distortion coefficients).

Currently, this file will extract the homography of a chessboard plane in frame and identify continguous laser patches in the image. 

Work needs to be done still to extract feature descriptions of the laser lines through triangulation and some tuning needs to be done on the Gaussian laser detection algorithm.

Here are some images from the current progress.

### Chessboard Pose Extraction
![chessboard axes](https://user-images.githubusercontent.com/55857337/195712009-8466d603-fd94-467d-88a2-4359af153c01.png)

### Laser Color Reward Image
![image that is brighter the closer the color is to the laser's color](https://user-images.githubusercontent.com/55857337/195712270-e16e98c5-410c-42f5-8688-a1411621eeaf.png)

### Potential Laser Pixels (Windows)
![image with circles drawn on potential laser pixels](https://user-images.githubusercontent.com/55857337/195712355-c1b27558-fcdb-41d2-9682-da058abe5582.png)

### Laser Subpixels
![image with pixels having a laser point in them white and everything else black](https://user-images.githubusercontent.com/55857337/195714877-d5315974-5103-4909-8a8e-3078c927a8d9.png)

### Laser Patches
![image with laser pixels grouped in minimum size patches](https://user-images.githubusercontent.com/55857337/195714826-8089c74b-2bdf-4534-b5c7-59cc35b1e0ec.png)
