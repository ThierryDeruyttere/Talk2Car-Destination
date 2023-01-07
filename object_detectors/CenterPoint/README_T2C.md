
### Installation

Please refer to [INSTALL](docs/INSTALL.md) to set up libraries needed for distributed training and sparse convolution.


#### Download data and organise as follows

```
# For nuScenes Dataset + Talk2Car extension       
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
       ├── commands      <-- Talk2Car commands
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s DATA_ROOT nuScenes
```


#### Create data

Data creation should be under the gpu environment.

```
# nuScenes
python tools/create_data.py t2c_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="train" --nsweeps=10
```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── CenterPoint
       └── data    
              └── nuScenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── commands <-- Talk2Car commands
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database 
```

### Train & Evaluate in Command Line

**Now we only support training and evaluation with gpu. Cpu only mode is not supported.**

Use the following command to start a distributed training using 4 GPUs. The models and logs will be saved to ```work_dirs/CONFIG_NAME``` 

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/t2c/voxelnet/t2c_centerpoint_voxelnet_0075voxel_fix_bn_z.py
```

For distributed testing with 2 gpus,

```bash
python -m torch.distributed.launch --nproc_per_node=2 ./tools/t2c_dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 
```

For testing with one gpu and see the inference time,

```bash
python ./tools/t2c_dist_test.py CONFIG_PATH --work_dir work_dirs/t2c_centerpoint_voxelnet_0075voxel_fix_bn_z --checkpoint work_dirs/t2c_centerpoint_voxelnet_0075voxel_fix_bn_z/latest.pth --speed_test 
```
To get the results from Table B.1 and B.2 from the paper, please open `compute_theoretical_iou_top-min_distance_same_class.ipynb`.

As we did not use this model in the paper, we did not create a prediction extraction script.

# Checkpoints

The used checkpoint in the paper can be found [here](https://drive.google.com/file/d/1HGy8aM9KoD8YjLgG4oI3F-k9WR6qysLu/view?usp=share_link).
Unzip the weight and put it in `checkpoints/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z`

