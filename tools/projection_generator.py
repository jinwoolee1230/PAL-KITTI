import pykitti
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# KITTI dataset path
basedir = '/home/jinwoo/Projects/KITTI'
date = '2011_10_03'
output_dir = os.path.join(basedir, 'scan_projection', date)

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Parameters for spherical projection
resolution = (16, 720)  # height, width
f_h = 360.0  # Horizontal field of view in degrees
f_vu = 2.0   # Vertical field of view upper bound in degrees
f_vl = -24.8  # Vertical field of view lower bound in degrees

def lidar_to_spherical_image(dataset, frame_idx, resolution=(16, 720), f_h=360.0, f_vu=2.0, f_vl=-24.8):
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

# Process each drive folder
drive_dirs = [d for d in os.listdir(os.path.join(basedir, date)) if d.endswith('_sync')]

for drive_dir in drive_dirs:
    drive = drive_dir.split('_')[-2]
    print(f"Processing drive {drive}...")
    dataset = pykitti.raw(basedir, date, drive)
    drive_output_dir = os.path.join(output_dir, drive)
    os.makedirs(drive_output_dir, exist_ok=True)
    
    num_frames = len(dataset.velo_files)
    for frame_idx in tqdm(range(num_frames), desc=f"Drive {drive}"):
        depth_image = lidar_to_spherical_image(dataset, frame_idx, resolution, f_h, f_vu, f_vl)
        output_path = os.path.join(drive_output_dir, f"frame_{frame_idx:04d}.png")
        plt.imsave(output_path, depth_image, cmap='magma', vmin=0, vmax=255)

print("All LiDAR projections have been processed and saved.")

