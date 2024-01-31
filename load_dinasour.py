# %%
from scipy.io import loadmat

data_path = 'tsbb33-datasets/BAdino/BAdino2.mat'
dataset = loadmat(data_path)

# %%
print(type(dataset))
print(dataset.keys())
print(dataset['points2DMatrix'].shape)
print(dataset['newPoints3D'].shape)
print(dataset['newPs'].shape)
print(dataset['newPoints2D'].shape) #??
print(dataset['newPoints2D'][:, 0][0].shape)
print(dataset['newPoints2D'][:, 0][0])
print(dataset['newPoints2D'][:, 1][0])

# %%
# visualize 3D points

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(dataset['newPoints3D'])
o3d.io.write_point_cloud("sync.ply", pcd)

o3d.visualization.draw_geometries([pcd])


# %%
# load image
from PIL import Image
import matplotlib.pyplot as plt
img = Image.open("../images/dinasour/viff.000.ppm")
# plt.imshow(img)

# %%
# superimpose 2D points on the 1st image
plt.figure()
plt.imshow(img)
plt.plot(dataset['newPoints2D'][:, 0][0][0, :], dataset['newPoints2D'][:, 0][0][1, :], 'ro')
plt.savefig('..//images/img0_reproj.png')

plt.figure()
plt.imshow(img)
plt.plot(dataset['points2DMatrix'][:, 0, :][:, 0], dataset['points2DMatrix'][:, 0, :][:, 1], 'ro')
plt.savefig('..//images/img0_origin.png')


# %%
# superimpose 2D points on the 2nd image
img = Image.open("../images/dinasour/viff.001.ppm")

plt.figure()
plt.imshow(img)
plt.plot(dataset['newPoints2D'][:, 1][0][0, :], dataset['newPoints2D'][:, 1][0][1, :], 'ro')
plt.savefig('../images/img1.png')
# %%
# superimpose 2D points on the 2nd image
img = Image.open("../images/dinasour/viff.002.ppm")

plt.figure()
plt.imshow(img)
plt.plot(dataset['newPoints2D'][:, 2][0][0, :], dataset['newPoints2D'][:, 2][0][1, :], 'ro')
plt.savefig('../images/img2.png')

# %%
print(dataset['points2DMatrix'][:, 0, :].shape)
print(dataset['points2DMatrix'][:, 0, :])
print(dataset['newPoints2D'][:, 0][0])

# %%
