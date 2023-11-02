import open3d as o3d
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import compute_gaze_direction, compute_cube_center, compute_angle, index_1d_to_3d


# Gaze direction calculation
path = r"D:\User_Behavior\user_behavior_data\vv-ub-04.26\Sunqiran_News_interviewing.txt"
data = []
with open(path, "r", encoding='utf-8') as fr:
    columns = fr.readline().strip().split(', ')
    for line in fr.readlines()[1:]:
        data.append(list(map(lambda x: float(x), line.split(' '))))
data = np.array(data)

df = pd.DataFrame(data, columns=columns)
df.set_index('#Frame', inplace=True)

#gaze_directions = np.stack(df.apply(
#    lambda row: compute_gaze_direction(row[['HeadRX', 'HeadRZ', 'HeadRY']], row[['LEyeRX', 'LEyeRZ', 'LEyeRY']]),
#    axis=1).values)

gaze_directions = np.stack(df.apply(lambda row: [row['LEyeRX'], row['LEyeRZ'], row['LEyeRY']], axis=1).values)

# Load the point cloud from a .ply file
pcd = o3d.io.read_point_cloud(r"D:\User_Behavior\volumetric_data\News_interviewing.ply")

# Define the size of the cube blocks
cube_size = 0.15  # Change this value to adjust the cube size

# Get the axis-aligned bounding box of the point cloud
bbox = pcd.get_axis_aligned_bounding_box()

# Calculate the number of cubes needed to cover the bounding box
x_range = bbox.max_bound[0] - bbox.min_bound[0]
y_range = bbox.max_bound[1] - bbox.min_bound[1]
z_range = bbox.max_bound[2] - bbox.min_bound[2]
num_cubes_x = int(x_range / cube_size)
num_cubes_y = int(y_range / cube_size)
num_cubes_z = int(z_range / cube_size)

n_cubes = num_cubes_x * num_cubes_y * num_cubes_z

volumetric = np.ones((num_cubes_x, num_cubes_y, num_cubes_z))

colors = np.zeros((num_cubes_x, num_cubes_y, num_cubes_z, 4))

count_list = []
# Extract cube blocks from the point cloud
for idx in tqdm(range(n_cubes)):
    i, j, k = index_1d_to_3d(idx, num_cubes_x, num_cubes_y, num_cubes_z)
    # Calculate the bounds of the cube
    min_bound = [bbox.min_bound[0] + i * cube_size,
                 bbox.min_bound[1] + j * cube_size,
                 bbox.min_bound[2] + k * cube_size]
    max_bound = [bbox.min_bound[0] + (i + 1) * cube_size,
                 bbox.min_bound[1] + (j + 1) * cube_size,
                 bbox.min_bound[2] + (k + 1) * cube_size]
    # Crop the point cloud to the bounds of the cube
    cube = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound))

    # Store the number of points in the cube
    if len(cube.points) > 50:
        volumetric[i][j][k] = len(cube.points)
    else:
        volumetric[i][j][k] = 0

    cube_center = compute_cube_center(min_bound, max_bound)
    count = 0
    for g in range(len(gaze_directions)):
        gaze_vector = gaze_directions[g]
        headset_position = np.asarray(data[g][1: 4])
        distance_vector = headset_position - cube_center
        distance_current = np.linalg.norm(distance_vector)
        angle_current = compute_angle(gaze_vector, cube_center)

        if angle_current < 120:
            with open(r"D:\User_Behavior\user_behavior_data\angles\Sunqiran_News_interviewing.txt", "a") as f:
                f.write(str(angle_current) + "\n")
        #if angle_current < 90:
        #    count += 1

    #with open("hit_times.txt","a") as f:
    #    f.write(str(count) + "\n")


