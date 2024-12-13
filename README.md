# SIPAL-KITTI
Spatial Image Projection Autoencoder for LiDAR data

![projections](readmefiles/projection_shrink.gif)

## Project Structure
### `autoencoder`
- **`main.py`**: Main script to train the autoencoder model.
- **`model.py`**: Defines the architecture of the autoencoder model used for compression.
- **`test.py`**: Contains testing functions for evaluating the model.
### `tools`
- **`LiDAR_image_visualizer.py`**: Script to visualize LiDAR data projections and compressed images.
- **`projection_generator.py`**: Converts 3D LiDAR data into 2D images by projecting point clouds.


## 1. LiDAR projection generator
The LiDAR data from KITTI dataset is projected to 2D image, in size of 720 x 16, considering the FOV of Velodyne HDL-64E, which was used for data collection in KITTI datset.
This code can be used for pointclouds from other LiDARs, with adjustment on FOV and number of points.

![projections](readmefiles/projection.gif)


## 2. Autoencoder
The autoencoder implemented here is strongly affected by the [delora repository](https://github.com/leggedrobotics/delora). You can checkout the details in the linked repository.

![autoencoder](readmefiles/autoencoder.gif)

## How to use
1. Run **`pip install requirements.txt`** for installation of required python libraries.
2. Change **`path/to/your/directories`** in the codes to your desired directories.
3. Run **`LiDAR_image_visualizer.py`** to visualize your LiDAR data and run **`projection_generator.py`** to save it as png files.
4. Run **`main.py`** to train your autoencoder.
5. Run **`test.py`** to compare the original projection image and reconstructed version of it.
