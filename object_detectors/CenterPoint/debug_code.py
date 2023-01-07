
import sys
import json

import cv2
import numpy as np
from PIL import Image
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image

sys.path.insert(0, "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/")
sys.path.insert(0, "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/det3d/datasets/nuscenes/")

sys.path.insert(0, "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/nuscenes-devkit/python-sdk/")
from talk2car import Talk2Car
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

def jaccard(a, b):
    # pairwise jaccard(IoU) botween boxes a and boxes b
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    inter = np.clip(rb - lt, 0, None)

    area_i = np.prod(inter, axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)

    area_u = area_a[:, np.newaxis] + area_b - area_i
    return area_i / np.clip(area_u, 1e-7, None)  # len(a) x len(b)


checkpoint = "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/latest.pth"
config = "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/configs/t2c/voxelnet/t2c_centerpoint_voxelnet_0075voxel_fix_bn_z.py"


# In[11]:


from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator
import torch


# In[12]:


cfg = Config.fromfile(config)
device = "cuda:0"


# In[13]:


net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
net.load_state_dict(torch.load(checkpoint)["state_dict"])
net = net.to(device).eval()

from det3d.datasets import build_dataloader, build_dataset

dataset = build_dataset(cfg.data.test)
sample = dataset[0]
inp = {}
for k, v in sample.items():
    if k == "coordinates":
        inp[k] = torch.tensor(
            np.pad(v, ((0, 0), (1, 0)), mode='constant', constant_values=0)).to(device)
    elif k == "shape":
        inp[k] = [v]
    elif k == "metadata":
        inp[k] = [v]
    else:
        inp[k] = torch.tensor(v).to(device)

with torch.no_grad():
    outputs = net(inp, return_loss=False)[0]

print(outputs)

# In[2]:


nusc = Talk2Car(version="test", dataroot="data/nuScenes", verbose=True)
predictions_file = "work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/infos_test_10sweeps_withvelo.json"
predictions_data = json.load(open(predictions_file, "r"))


# In[ ]:





# In[3]:


def load_cloud_from_nuscenes_file(pc_f, num_features = 5):
    #logging.info('loading cloud from: {}'.format(pc_f))
    cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
    # last dimension should be the timestamp.
    cloud[:, 4] = 0
    return cloud

def load_cloud_from_deecamp_file(pc_f):
    #logging.info('loading cloud from: {}'.format(pc_f))
    num_features = 4
    cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
    # last dimension should be the timestamp.
    cloud = np.hstack((cloud, np.zeros([cloud.shape[0], 1])))
    return cloud


# In[4]:


nusc.get("sample_data",nusc.commands[0].frame_token)


# In[5]:


sample_record = nusc.get("sample","6ba17bc380984319b41b0c97cd18c312")
sample_record


# In[6]:


lidar_data = nusc.get("sample_data",sample_record["data"]["LIDAR_TOP"])
lidar_data


# In[7]:


import os
root = "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/data/nuScenes"


# In[8]:


cloud = load_cloud_from_nuscenes_file(os.path.join(root, lidar_data["filename"]), 5)


# In[9]:


cloud.shape


# In[10]:

# In[ ]:





# In[14]:



voxel_range = cfg.voxel_generator.range
voxel_size = cfg.voxel_generator.voxel_size
max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
max_voxel_num = cfg.voxel_generator.max_voxel_num
voxel_generator = VoxelGenerator(
    voxel_size=voxel_size,
    point_cloud_range=voxel_range,
    max_num_points=max_points_in_voxel,
    max_voxels=max_voxel_num,
)


# In[ ]:





# In[15]:


# load sample from file
#self.points = self.load_cloud_from_nuscenes_file(cloud_file)
#self.points = self.load_cloud_from_deecamp_file(cloud_file)
points = cloud
points[:, 4] = 0
# prepare input
voxels, coords, num_points = voxel_generator.generate(points, max_points_in_voxel)
num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
grid_size = voxel_generator.grid_size
coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)


# In[16]:


a = torch.tensor(voxels, dtype=torch.float32)


# In[ ]:





# In[ ]:





# In[17]:



voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
coords = torch.tensor(coords, dtype=torch.int32, device=device)
num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=device)


# In[18]:



inputs = dict(
    voxels=voxels,
    num_points=num_points,
    num_voxels=num_voxels,
    coordinates=coords,
    shape=[grid_size]
)


# In[19]:


#torch.cuda.synchronize()
with torch.no_grad():
    outputs = net(inputs, return_loss=False)[0]

print(outputs)
