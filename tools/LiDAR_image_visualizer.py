import pykitti
import numpy as np
import matplotlib.pyplot as plt

# KITTI dataset path and sequence info
basedir = '/home/jinwoo/Projects/KITTI'
date = '2011_09_26'
drive = '0005'

# Load the dataset
dataset = pykitti.raw(basedir, date, drive)

def lidar_to_spherical_image(dataset, frame_idx, resolution=(16, 720), f_h=360.0, f_vu=2.0, f_vl=-24.9):
    f_h_rad = np.deg2rad(f_h)
    f_vu_rad = np.deg2rad(f_vu)
    f_vl_rad = np.deg2rad(f_vl)
    f_v_rad = f_vu_rad - f_vl_rad
    height, width = resolution
    delta_h = f_h_rad / width
    delta_v = f_v_rad / height
    lidar = dataset.get_velo(frame_idx)
    x, y, z = lidar[:, 0], lidar[:, 1], lidar[:, 2]
    ranges = np.sqrt(x**2 + y**2 + z**2)
    d = np.sqrt(x**2 + y**2)
    theta_h = np.arctan2(y, x)
    theta_v = np.arctan2(z, d)
    u = (f_h_rad / 2 - theta_h) / delta_h
    v = (f_vu_rad - theta_v) / delta_v
    u_idx = np.floor(u).astype(int)
    v_idx = np.floor(v).astype(int)
    u_idx = np.mod(u_idx, width)
    v_idx = np.clip(v_idx, 0, height - 1)
    depth_image = np.full((height, width), fill_value=np.inf)
    for i in range(len(ranges)):
        row = v_idx[i]
        col = u_idx[i]
        depth_image[row, col] = min(depth_image[row, col], ranges[i])
    depth_image[depth_image == np.inf] = 0
    depth_image_normalized = 255 * (1 - depth_image / np.max(depth_image))
    return depth_image_normalized.astype(np.uint8)

# Interactive visualization
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(np.zeros((16, 720)), cmap='magma', aspect='auto', vmin=0, vmax=255)
ax.axis('off')
ax.set_title("2D Spherical Image from KITTI LiDAR")

num_frames = len(dataset.velo_files)
for frame_idx in range(num_frames):
    depth_image = lidar_to_spherical_image(dataset, frame_idx, resolution=(64, 1024))
    print(f"Frame {frame_idx}, max: {np.max(depth_image)}, min: {np.min(depth_image)}")  # Debug
    im.set_data(depth_image)
    plt.pause(0.05)

plt.ioff()
plt.show()

