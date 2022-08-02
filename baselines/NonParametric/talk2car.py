import random
from itertools import groupby
import copy
import h5py
import torch
from PIL import ImageDraw
from torch.utils.data import Dataset
import os
import json
from torchvision import transforms
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
import math


detector_classes = ["car", "truck", "trailer", "bus", "construction_vehicle", "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"]
detector_classes_mapping = {v: k for k, v in enumerate(detector_classes)}

def gauss(n=11, sigma=1):
    r = range(-int(n / 2), int(n / 2) + 1)
    return [
        1
        / (sigma * math.sqrt(2 * math.pi))
        * math.exp(-float(x) ** 2 / (2 * sigma ** 2))
        for x in r
    ]


class Talk2Car_Detector(Dataset):
    def __init__(self, dataset_root, width=1200, height=800, split="train", unrolled=False, use_ref_obj=True, gaussian_sigma=3, gaussian_size=11):

        self.orig_width = 1200
        self.orig_height = 800
        self.width_scaling = width / self.orig_width
        self.height_scaling = height / self.orig_height
        self.unrolled = unrolled
        self.use_ref_obj = use_ref_obj
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_size = gaussian_size

        self.split = split
        self.dataset_root = dataset_root
        self.width = width
        self.height = height
        self.data = list(
            json.load(open(os.path.join(self.dataset_root, f"talk2car_destination_{split}.json"), "r")).values()
        )

        if self.unrolled:
            self.data = [item for sublist in self.data for item in sublist]

        self.transforms = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor(),]
        )

        self.ego_car_value = 1
        self.referred_object_value = 1
        self.object_value = 1
        self.mapping = json.load(
            open(os.path.join(dataset_root, f"{split}_command_mapping.json"), "r")
        )
        self.embeddings = np.array(
            h5py.File(os.path.join(dataset_root, f"{split}_command_mapping.h5"), "r")[
                "embeddings"
            ]
        )

        self.gauss = torch.tensor(gauss(n=gaussian_size, sigma=self.gaussian_sigma))


    def __len__(self):
        return len(self.data)

    def get_mask(self, polygon, fill):
        mask = Image.new("RGB", (self.width, self.height))
        top_draw = ImageDraw.Draw(mask)
        top_draw.polygon(polygon, fill=fill)

        return torch.from_numpy(np.array(mask))

    def create_channel_for_all_objs(self, objects, classes):
        masks = torch.zeros(len(detector_classes), self.height, self.width)
        classes_groups = {
            key: [item[0] for item in group]
            for key, group in groupby(sorted(enumerate(classes), key=lambda x: x[1]), lambda x: x[1])
        }

        for class_ind, box_indices in classes_groups.items():
            mask = Image.new("L", (self.width, self.height))
            top_draw = ImageDraw.Draw(mask)
            boxes = [objects[box_index] for box_index in box_indices]
            for box in boxes:
                top_draw.polygon([(x * self.width_scaling, y * self.height_scaling) for (x, y) in box], fill=self.object_value)
            masks[class_ind] = torch.from_numpy(np.array(mask))
        return masks

    def __getitem__(self, ix):
        item = self.data[ix]

        if self.unrolled:
            item = [item]

        # load top down
        img_name = item["top-down"]
        img = Image.open(os.path.join(self.dataset_root, "top_down", img_name)).convert("RGB")

        # load detected top down
        detection_boxes = item["all_detections_top"]
        detection_boxes_type = item["detected_object_classes"]

        if self.split != "train":
            detection_pred_box_index = item["predicted_referred_obj_index"]
        else:
            referred_box = item["gt_referred_obj_top"]

            referred_poly = Polygon(referred_box)
            candidate_polys = [Polygon(item) for item in detection_boxes]
            ious = np.array([
                referred_poly.intersection(candidate_poly).area / referred_poly.union(candidate_poly).area for candidate_poly in candidate_polys
            ])
            if any([iou > 0.5 for iou in ious]):
                detection_pred_box_index = ious.argmax()
            else:
                detection_pred_box_index = np.random.randint(len(detection_boxes))
                detection_boxes[detection_pred_box_index] = referred_box

        if self.transforms:
            img = self.transforms(img)

        # make grid for car start point, referred object and end pos
        ego_car_top = item["egobbox_top"]
        ego_car_mask = self.get_mask(
            [
                (x * self.width_scaling, y * self.height_scaling)
                for (x, y) in ego_car_top
            ],
            self.ego_car_value,
        )[:, :, :1].permute([2, 0, 1])

        # Referred object top down
        referred_obj_top = detection_boxes[detection_pred_box_index]
        referred_obj_mask = self.get_mask(
            [
                (x * self.width_scaling, y * self.height_scaling)
                for (x, y) in referred_obj_top
            ],
            self.referred_object_value,
        )[:, :, :1].permute([2, 0, 1])

        all_objs = copy.deepcopy(detection_boxes)
        all_objs.pop(detection_pred_box_index)
        all_cls = copy.deepcopy(detection_boxes_type)

        all_cls.pop(detection_pred_box_index)
        all_objs_mask = self.create_channel_for_all_objs(all_objs, all_cls)

        # Get end pos
        end_pos = []
        for (x,y) in item["destinations"]:
            end_pos.append([x / self.orig_width, y / self.orig_height])

        if not self.unrolled:
            while len(end_pos) < 3:
                end_pos.append(random.sample(end_pos, 1)[0])

        x_one_hot = torch.zeros(self.width)
        y_one_hot = torch.zeros(self.height)

        # one_hot[int(end_pos[0][0]), int(end_pos[0][1])] = 1
        # one_hot = one_hot.view(-1)
        # ix = one_hot.argmax()
        for i, val in enumerate(self.gauss):
            for p in end_pos:
                ix = int(p[0]) - self.gaussian_size // 2 + i
                if ix < 0 or ix >= self.width - 1:
                    continue

                x_one_hot[ix] += val

        for i, val in enumerate(self.gauss):
            for p in end_pos:
                ix = int(p[1]) - self.gaussian_size // 2 + i
                if ix < 0 or ix >= self.height - 1:
                    continue

                y_one_hot[ix] += val

        if self.use_ref_obj:
            layout = torch.cat([img, ego_car_mask, referred_obj_mask, all_objs_mask])
        else:
            layout = torch.cat([img, ego_car_mask, all_objs_mask])

        return {
            "x": layout,
            # "command": item[0]["command_data"]["command"],
            "command_embedding": self.embeddings[
                self.mapping[item["command_token"]]
            ],
            "x_cls": torch.tensor(all_cls),
            "y": (x_one_hot / x_one_hot.sum(), y_one_hot / y_one_hot.sum()),
            "end_pos": torch.tensor(end_pos)
        }

    def get_obj_info(self, bidx):
        item = self.data[bidx]
        if self.unrolled:
            item = [item]

        img_name = item["top-down"]
        img_path = os.path.join(self.dataset_root, "top_down", img_name)

        frontal_img_name = item["image"]
        frontal_img_path = os.path.join(self.dataset_root, "frontal_imgs", frontal_img_name)

        ego_car = item["egobbox_top"]
        ref_obj = item["gt_referred_obj_top"]
        endpoint = [item["destinations"][0][0], item["destinations"][0][1]]
        command = item["command"]


        frame_data = json.load(
            open(
                os.path.join(
                    self.dataset_root,
                    "frame_data",
                    "rotated_" + item[0]["frame_data_url"].split("/")[-1],
                ),
                "r",
            )
        )

        return img_path, frontal_img_path, ego_car, ref_obj, endpoint, command, frame_data


def main():

    ds = Talk2Car_Detector(dataset_root="../../data/", split="val")
    ds[0]


if __name__ == "__main__":
    main()
