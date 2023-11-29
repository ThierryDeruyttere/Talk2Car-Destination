# Dataset creation
This directory is for the people that are interested in how we created the Talk2Car-Destination dataset.

The directory `annotation_tool` contains the annotation tool we used to annotate the dataset.
This is currently provided as-is to show how we have done it. This currently does not work as-is.

The directory `dataset_normalization` contains the code we used to normalize the dataset.
It contains the following files:

- `anno_tool_api.py`: This file contains the functionality to create the data for the annotation tool we had. i.e., rotate all the top-down views.
- `convert_top_down_images_to_rotated.py`: Use this file to rotate all the top-down images. 
You will need to download the Talk2Car dataset [here](https://github.com/talk2car/Talk2Car). Make sure you download the full Talk2Car dataset and not Talk2CarSlim. You will also need to download the [nuScenes map extension dataset](https://www.nuscenes.org/).
- `convert_t2c_data_to_rotated.py`: Use this file to rotate all the Talk2Car data (i.e., all the objects). For this, you will need to download the original data jsons that you can find [here](https://drive.google.com/file/d/1XhdEw3wagt8rQ21hNoAiYzMqXBhZGvah/view?usp=sharing).
- `unrotated.json`: This file contains the data of the Talk2Car-Destination dataset before we rotated the top-down images.
- `clean_data.py`: After you rotate the data, you can still clean the data based on rating, usability etc. This was not used in our paper.
