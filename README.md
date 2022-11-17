# laser_stereo_system
Undergraduate thesis work on underwater active stereo.

# Calibration [WIP]
`calibrate_laser.py` in the `src` directory contains a script for extracting extrinsic parameters of the laser relative to a given rectified images and the camera matrix.

Currently, this file will extract the homography of a chessboard plane in frame and identify continguous laser patches in the image. 

Work needs to be done still to extract feature descriptions of the laser lines through triangulation and some tuning needs to be done on the Gaussian laser detection algorithm.

Here are some images from the current progress.

### Chessboard Pose Extraction
![Chessboard Axes](https://user-images.githubusercontent.com/55857337/202570815-65b65e35-d150-4a9f-b2eb-be9641a1fff8.png)

### Laser Color Reward Image
![image that is brighter the closer the color is to the laser's color](https://user-images.githubusercontent.com/55857337/202570885-d3198248-4ade-499e-b87d-61a8d5be409f.png)

### Potential Laser Pixels (Windows)
![image with circles drawn on potential laser pixels](https://user-images.githubusercontent.com/55857337/195712355-c1b27558-fcdb-41d2-9682-da058abe5582.png)

### Laser Subpixels
![image with pixels having a laser point in them white and everything else black](https://user-images.githubusercontent.com/55857337/202570931-caa357b4-30ac-4299-95b4-209fd29c1191.png)

### Laser Patches
![image with laser pixels grouped in minimum size patches](https://user-images.githubusercontent.com/55857337/202570976-59595cfa-fffc-4415-a0a8-2b54e5c6c7d0.png)

### Detected Lines
![detectedlines2022-11-17 16:07:24 017902](https://user-images.githubusercontent.com/55857337/202571093-aba2235c-7d83-438c-b663-c0fd3c1dc8a6.png)

### Merged Lines
![mergedlines2022-11-17 16:07:24 017902](https://user-images.githubusercontent.com/55857337/202571129-de154f95-9341-4b04-bdc1-7e4f0d635d39.png)

### Grouped Patches
![groupedpatches2022-11-17 16:07:24 017902](https://user-images.githubusercontent.com/55857337/202571166-2027324d-29e1-420f-9e68-a835c37f9612.png)

### Grouped Patches (clearer)
![groupedpatchesbig2022-11-17 16:07:24 017902](https://user-images.githubusercontent.com/55857337/202571199-9586f23b-940a-4bdc-8b73-c3743a4a0e87.png)
