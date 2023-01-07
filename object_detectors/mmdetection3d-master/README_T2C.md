# Installation

Please refer to [getting_started.md](docs/getting_started.md) for installation.

# Set up the data.
First, you need to download the complete nuScenes dataset.
On their download page, you will need to download all 10 parts.
You will need 300GB+ to download all data. 
Once this is done, make a symbol link to the directory where you downloaded the data in `data`.
Basically you want the file structure to look like this:

```
mmdetection3d-master
    |---   configs
    |---   data
            |----- lyft
            |----- s3dis
            |----- scannet
            |----- sunrgbd
            |----- nuscenes
                    |--- sweeps
                    |--- samples
                    |--- ...
            
```

# Create data

```
python tools/create_data.py talk2car --root-path ./data/nuscenes/ --out-dir ./data/nuscenes --extra-tag t2c
```

# Train

first 

```
 CUDA_VISIBLE_DEVICES=0,1,3 ./tools/dist_train.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py 3
```

then

```
 CUDA_VISIBLE_DEVICES=0,1,3 ./tools/dist_train.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py 3
```

# Test

```
python demo/t2c_det_demo.py --config configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py --checkpoint CHECKPOINT_PATH/latest.pth```
```

# Computing theoretical results

To compute the results from Table B.1 and B.2 in the paper, please open `compute_theoretical_acc_top.ipynb`.

# Extracting the predictions

`python extract_frontal_top_down_boxes_feats_fcos3d.py`

We also provide the extracted files in `../data_root/extracted_feats`


# Checkpoints

The used checkpoint in the paper can be found [here](https://drive.google.com/file/d/1hzHgja5moFTwJusHTro0DxkdgIo323rv/view?usp=share_link).
Unzip the weight and put it in `checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune`

