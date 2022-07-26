# Talk2Car-Destination

This is the dataset that accompanies the paper [Predicting Physical World Destinations for Commands Given to Self-Driving Cars
](https://arxiv.org/abs/2112.05419) accepted at AAAI 2022.

Talk2Car-Destination is an extension to [Talk2Car](https://github.com/talk2car/Talk2Car) which is built on [nuScenes](https://www.nuscenes.org/).

# Annotation format
Each json from the dataset is a dictionary where the key is the command token and the value is a dictionary of the following format.

```
{
 "image": "img name",
 "top-down": "top down image name"
 "command": "given command"
 "destinations": [[x,y]], #is a list of (x, y) pairs where each pair is a destination in the top-down image
 "egobbox_top": [ 4 x 2 list], # contains the corners of the ego vehicle bounding box in the top-down image.
 "all_detections_top": [64 x 4 x 2 list], # contains the corners of all detected objects in the top-down image.
 "detected_object_classes": [64 list], # contains the class of each detected object.
 "all_detections_front": [64 x 4 x 2 list], # contains the corners of all detected objects in the frontal image.
 "predicted_referred_obj_index":  [64 list], # contains the index of the predicted referred object.
 "detection_scores":  [64 list], # contains the confidence score of each detected object.
 "gt_referred_obj_top": [4 x 2 list], # contains the corners of the ground truth referred object in the top-down image.
}              
```

# How to use

1. Download top-down images [here](https://drive.google.com/file/d/1lrgghIVYPxCboZ77eTO8cdFcm_6mcZga/view?usp=sharing) and put the images in the data folder.
2. Download the frontal images [here](https://drive.google.com/file/d/1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek/view?usp=sharing) and put the images in the data folder.
3. Download the Talk2Car-Destination dataset [here](https://drive.google.com/file/d/1hZ-3OOAdpFUjkGyObi0xqg10U62UNOrj/view?usp=sharing) and put the jsons in the data folder.
4. Run `visualize.py` to visualize a sample of the dataset

# Integration with Talk2Car

Drag the Talk2Car-Destination dataset into the `data/commands` folder of Talk2Car.
Next, when calling the `get_talk2car_class`, set `load_talk2car_destination` to `True`.
Talk2Car-Destination will now be loaded.

# Citation
If you use this dataset, please consider using the following citation:
```
@inproceedings{grujicic2021predicting,
  title={Predicting Physical World Destinations for Commands Given to Self-Driving Cars},
  author={Grujicic, Dusan and Deruyttere, Thierry and Moens, Marie-Francine and Blaschko, Matthew},
  booktitle={Thirty-Sixth AAAI Conference on Artificial Intelligence},
  year={2021},
  organization={AAAI Press}
}
```