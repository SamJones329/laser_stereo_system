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
![detectedlines2022-11-17 16:14:35 657407](https://user-images.githubusercontent.com/55857337/202571534-753a1f78-2a56-4ccb-a101-a28925296ea5.png)

### Merged Lines
![mergedlines2022-11-17 16:14:35 657407](https://user-images.githubusercontent.com/55857337/202571557-ef2118d8-c00d-4fb4-9761-91800cabc0b3.png)

### Grouped Patches
![groupedpatches2022-11-17 16:14:35 657407](https://user-images.githubusercontent.com/55857337/202571577-db02b1cd-79d7-4d0f-b989-f43b7f4f378d.png)

### Grouped Patches (clearer)
![groupedpatchesbig2022-11-17 16:14:35 657407](https://user-images.githubusercontent.com/55857337/202571603-1218fb86-bf1f-41b9-afa2-889512a19514.png)

### Plane Extraction
TODO - add images here

### Plane fitting [(source)](http://srv.uib.es/wp-content/uploads/2019/12/MassotCampos_PhD_v1.2.2_printer.pdf)
![Screenshot from 2022-11-17 16-18-18](https://user-images.githubusercontent.com/55857337/202572178-d41fa7e0-ad43-414c-8f86-a8abeecdd4c1.png)
