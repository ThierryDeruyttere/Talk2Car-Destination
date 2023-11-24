# This script loads a sample from Talk2Car-Destination and visualizes the following things:
# referred object in top and frontal view
# ego car in top down view
# Destination

import json
from PIL import Image, ImageEnhance
from PIL import ImageDraw
import numpy as np
dataset = "val"
sample_ix = 1
#train samples
ds = json.load(open(f"./data/talk2car_destination_{dataset}.json", "r"))

# command = [x for x in ds.values() if x["command"] == "take a left to follow the grey car"]
# Look for sample with frontal_img train_0.jpg
# sample = [x for x in ds.values() if x["image"] == f"{dataset}_{image_ix}.jpg"][0]
sample = list(ds.values())[sample_ix]
# sample = sample_annos

# Load images
img = Image.open(f"./data/imgs/img_{sample['image']}").convert("RGB")
top_down = Image.open(f"./data/top_down/{sample['top-down']}").convert("RGB")

drw = ImageDraw.Draw(img)
drw_top = ImageDraw.Draw(top_down)

print(sample["command"])

# Draw on frontal view
(x,y,w,h) = sample['gt_ref_obj_box_frontal']
drw.rectangle([x, y, x+w, y+h], outline="yellow")
img.show()

# Draw on top-down view
flatten = lambda t: [item for sublist in t for item in sublist]

drw_top = ImageDraw.Draw(top_down)
ego_car = np.array(sample["egobbox_top"])
egobbox_polygon = np.concatenate((ego_car, ego_car[0, :][None, :]), 0)

# Draw predictions
for det_obj in sample["all_detections_top"]:
    det_obj = np.array(det_obj)
    det_obj_polygon = np.concatenate((det_obj, det_obj[0, :][None, :]), 0)
    drw_top.polygon(
        flatten(det_obj_polygon.tolist()), fill="#808080", outline="#808080",
    )

ref_obj = np.array(sample["gt_referred_obj_top"])
ref_obj_polygon = np.concatenate((ref_obj, ref_obj[0, :][None, :]), 0)

drw_top.polygon(flatten(egobbox_polygon.tolist()), fill="#ff0000", outline="#ff0000")
drw_top.polygon(
    flatten(ref_obj_polygon.tolist()), fill="yellow", outline="yellow",
)

# Draw annotations on top down
for (x1, y1) in sample["destinations"]:
    color = (255, 0, 255)
    x2 = int(x1 + 3)
    y2 = int(y1 + 3)
    x1 = int(x1 - 3)
    y1 = int(y1 - 3)
    drw_top.ellipse([(x1, y1), (x2, y2)], color)

top_down.show()