import argparse
import json
import os
import shutil
import sys
import math

import copy
import numpy as np
from PIL import Image, ImageDraw


def rotate_points(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = points[:, 0], points[:, 1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    q = np.concatenate((qx[:, None], qy[:, None]), axis=1)
    return q


def parse_args():
    parser = argparse.ArgumentParser(
        "Draw object annotations on top down and frontal view."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where the top down views, frontal views and infos are saved.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        help="Index of the sample.",
    )
    return parser.parse_args()


flatten = lambda t: [item for sublist in t for item in sublist]

object_color_map = {
    "human.pedestrian.adult": "#ff2200",
    "human.pedestrian.child": "#594c43",
    "human.pedestrian.wheelchair": "#ff8800",
    "human.pedestrian.stroller": "#4c2900",
    "human.pedestrian.personal_mobility": "#d9a66c",
    "human.pedestrian.police_officer": "#ffcc00",
    "human.pedestrian.construction_worker": "#736b00",
    "animal": "#ace639",
    "vehicle.car": "#27331a",
    "vehicle.motorcycle": "#8fbf8f",
    "vehicle.bicycle": "#00730f",
    "vehicle.bus.bendy": "#40ffbf",
    "vehicle.bus.rigid": "#00665f",
    "vehicle.truck": "#00c2f2",
    "vehicle.construction": "#acdae6",
    "vehicle.emergency.ambulance": "#2d4459",
    "vehicle.emergency.police": "#000f73",
    "vehicle.trailer": "#606cbf",
    "movable_object.barrier": "#5940ff",
    "movable_object.trafficcone": "#d26cd9",
    "movable_object.pushable_pullable": "#e6acd2",
    "movable_object.debris": "#802053",
    "static_object.bicycle_rack": "#7f0011",
}

def convert_points(data_root, command_data):
    limit_left = 7
    limit_right = 113
    limit_top = 40
    limit_bottom = 40

    name_frame_data = command_data['frame_data_url'].split("/")[-1]
    split = command_data["split"]
    ix = command_data["command_data"]["command_ix"]
    referred_obj = command_data["command_data"]["box_ix"]

    sample_data = json.load(open(os.path.join(data_root, "original_json", name_frame_data), "r"))

    # These are the already rotated top down views
    image = Image.open("../../data/top_down/top_down_{}_{}.png".format(split, ix))

    map_patch = sample_data["map_patch"]
    egobbox = sample_data["egobbox"]
    ego_translation = sample_data["ego_translation"]
    ego_rotation = sample_data["ego_rotation"]
    map_objects_bbox = sample_data["map_objects_bbox"]
    objects_type = sample_data["objects_type"]

    canvas_width, canvas_height = image.size

    map_patch_corner_x = ego_translation[0] - limit_left
    map_patch_corner_y = ego_translation[1] - limit_top
    map_patch_width = limit_left + limit_right
    map_patch_height = limit_top + limit_bottom

    yaw = -math.atan2(ego_rotation[1][0], ego_rotation[0][0])

    egobbox = np.array(egobbox)
    rotated_egobbox = rotate_points(ego_translation[:2], egobbox, yaw)
    copy_rotated_egobbox = copy.deepcopy(rotated_egobbox)
    x = rotated_egobbox[:, 0]
    y = rotated_egobbox[:, 1]
    x = (x - map_patch_corner_x) / map_patch_width * canvas_width
    y = (1 - (y - map_patch_corner_y) / map_patch_height) * canvas_height
    egobbox = np.concatenate((x[:, None], y[:, None]), 1)
    egobbox_polygon = np.concatenate((egobbox, egobbox[0, :][None, :]), 0)
    sample_data["egobbox"] = egobbox.tolist()


    map_objects_bbox = np.array(map_objects_bbox)
    map_objects_bbox_shape = map_objects_bbox.shape
    map_objects_bbox = map_objects_bbox.reshape(-1, map_objects_bbox_shape[2])
    map_objects_bbox = rotate_points(ego_translation[:2], map_objects_bbox, yaw)
    map_objects_bbox = map_objects_bbox.reshape(map_objects_bbox_shape)
    referred_obj_coords = copy.deepcopy(map_objects_bbox[referred_obj])

    x = map_objects_bbox[:, :, 0]
    y = map_objects_bbox[:, :, 1]

    x = (x - map_patch_corner_x) / map_patch_width * canvas_width
    y = (1 - (y - map_patch_corner_y) / map_patch_height) * canvas_height
    map_objects_bbox = np.concatenate((x[:, :, None], y[:, :, None]), 2)
    map_objects_polygon = np.concatenate(
        (map_objects_bbox, map_objects_bbox[:, 0, :][:, None, :]), 1
    )

    # if not (0 <= referred_obj_coords[:, 0]).all() or \
    #         not (referred_obj_coords[:, 0] <= canvas_width).all() or \
    #         not (0 <= referred_obj_coords[:, 1]).all() or \
    #         not (referred_obj_coords[:, 1] <= canvas_height).all():
    #     return None

    sample_data["map_objects_bbox"] = map_objects_bbox.tolist()

    path_points = command_data["points"]
    canvas_w, canvas_h = command_data["canvas"]["width"], command_data["canvas"]["height"]
    path = []

    orig_map_w = map_patch[2] - map_patch[0]
    orig_map_h = map_patch[3] - map_patch[1]

    for i in range(len(path_points)):
        o = [(path_points[i]["x"] / canvas_w) * orig_map_w + map_patch[0],
             (1 - (path_points[i]["y"] / canvas_h)) * orig_map_h + map_patch[1]]
        path.append(o)

    path = np.array(path)
    # unrotated_path = copy.deepcopy(path)
    rotated_path = rotate_points(ego_translation[:2], path, yaw)
    copy_rotated_path = copy.deepcopy(rotated_path)
    x = rotated_path[:, 0]
    y = rotated_path[:, 1]
    x = (x - map_patch_corner_x) / map_patch_width * canvas_width
    y = (1 - (y - map_patch_corner_y) / map_patch_height) * canvas_height
    path = np.concatenate((x[:, None], y[:, None]), 1)
    command_data["points"] = [{"x": x[0], "y": x[1]} for x in path]

    p = np.concatenate((copy_rotated_egobbox, referred_obj_coords, copy_rotated_path))
    p = p - np.mean(copy_rotated_egobbox, axis=0)

    if not (0 <= path[:, 0]).all() or \
            not (path[:, 0] <= canvas_width).all() or \
            not (0 <= path[:, 1]).all() or \
            not (path[:, 1] <= canvas_height).all():
        return None

    image_draw = ImageDraw.Draw(image)
    image_draw.polygon(
        flatten(egobbox_polygon.tolist()), fill="#ff0000", outline="#ff0000"
    )
    for object_type, map_object_polygon in zip(objects_type, map_objects_polygon):
        image_draw.polygon(
            flatten(map_object_polygon.tolist()),
            fill="#00ff00",
            outline="#00ff00",
        )

    for i in range(len(path) - 1):
        image_draw.line((path[i][0], path[i][1],
                       path[i + 1][0], path[i + 1][1]), fill="#ff0000", width=2)
    image.show()

    if (sum((-limit_left <= p[:, 0]) & (p[:, 0] <= limit_right) & (-limit_bottom <= p[:, 1]) & (p[:, 1] <= limit_top)) == len(p)):
        json.dump(sample_data, open("../data_root/normalized_jsons/rotated_{}".format(name_frame_data), "w"))
        return command_data
    else:
        return None


def main():

    command_data = []
    all_data = json.load(open("unrotated_data.json", "r"))
    print("previous size", len(all_data))
    #convert_points(all_data[0])
    #convert_points(all_data[100])
    data_root = "../data_root"

    for sample_data in all_data:
        new_command = convert_points(data_root, sample_data)
        if new_command:
            command_data.append(new_command)
    print("new size", len(command_data))

    json.dump(command_data, open("../data_root/rotated_data.json", "w"))
    print("Call clean_data.py to remove the commands that are not in the dataset anymore. Make sure that the path in clean_data.py is correct")

if __name__ == "__main__":
    main()
