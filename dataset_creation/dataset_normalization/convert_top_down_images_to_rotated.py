import argparse
import json
import os
from tqdm import tqdm

import cv2
import numpy as np
import tqdm
from nuscenes.utils.geometry_utils import view_points
from anno_tool_api import NuScenesMapT2C, NuScenesMapExplorerT2C
from nuscenes.nuscenes import NuScenes
import sys

from talk2car import Talk2Car
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, required=True)

def extract_data_for_t2c_per_video():
    args = parser.parse_args()
    # if not os.path.exists(os.path.join( "top_down_views")):
    #    os.makedirs(os.path.join( "top_down_views"))
    output_folder = "new_rotated_extracted"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    lazy_loading = {}
    nusc_root = (
        "/export/home2/NoCsBack/hci/thierry/datasets/nuScenes/data/sets/nuscenes"
    )
    # for split in ["train", "val", "test"]:
    split = args.split
    ds = Talk2Car(version=split, dataroot=nusc_root)

    for command_ix, command in enumerate(ds.commands):

        map_name = command.t2c.get(
            "log", command.t2c.get("scene", command.scene_token)["log_token"]
        )["location"]
        scene_name = command.t2c.get("scene", command.scene_token)["name"]
        print(scene_name)
        print(command.t2c.get("sample_data", command.frame_token)["sample_token"])

        if os.path.exists(os.path.join(output_folder, f"top_down_{split}_{command_ix}.png")):
            continue
        #frame_ix = 0
        #token = command.frame_token
        #while command.t2c.get("sample_data", token)["prev"]:
        #    token = command.t2c.get("sample_data", token)["prev"]
        #    frame_ix += 1

        # if os.path.exists(os.path.join(
        #         "extracted_data", scene_name, f"frame_{frame_ix}_data.json"
        #     )):
        #     continue

        #while nextFrame:
            # for frame_ix, frame_token in enumerate(all_frames[::-1]):

        if map_name not in lazy_loading:
            map_explorer = NuScenesMapExplorerT2C(
                NuScenesMapT2C(
                    dataroot=nusc_root,
                    map_name=map_name,
                )
            )

            lazy_loading[map_name] = map_explorer
        else:
            map_explorer = lazy_loading[map_name]
            # nuscenes_map = lazy_loading[map_name]["nuscenes_map"]

        sample_token = command.t2c.get("sample_data", command.frame_token)["sample_token"]

        # if not os.path.exists(os.path.join("extracted_data", scene_name)):
        #     os.makedirs(os.path.join("extracted_data", scene_name))

        # (
        #     egopose,
        #     egobbox,
        #     ego_translation,
        #     ego_rotation,
        #     cam_translation,
        #     cam_rotation,
        #     map_patch,
        #     map_objects,
        #     map_objects_bbox,
        #     map_objects_type,
        #     cam_intrinsic
        # ) = map_explorer.render_surroundings_on_fancy_map(
        #     ds,
        #     sample_token=sample_token,
        #     out_path=os.path.join(
        #          "extracted_data", scene_name, f"top_down_XXX.png"
        #     ),
        #     verbose=False,
        #     render_objects=False,
        #     render_car=False,
        #     only_visible_objects=True,
        # )

        # (
        #     cam_path,
        #     map_patch,
        #     map_patch_margin,
        #     egobbox,
        #     map_objects_center,
        #     map_objects_elevation,
        #     map_objects_bbox,
        #     image_objects_bbox,
        #     objects_token,
        #     objects_type,
        #     cam_intrinsic,
        #     ego_translation,
        #     ego_rotation,
        #     cam_translation,
        #     cam_rotation,
        # ) = map_explorer.generate_sample_data(ds, sample_token=sample_token)

        # Get boxes and image
        sd_rec = ds.get("sample_data", command.frame_token)
        #impath, boxes, camera_intrinsic = ds.get_sample_data(sd_rec["token"])
        # boxes_2d = [ds._transform_3d_to_2d_bbox(x.corners(), camera_intrinsic) for x in boxes]
        # gt_box = ds._transform_3d_to_2d_bbox(command.box.corners(), camera_intrinsic)

        # dists = []
        # for ix, b in enumerate(boxes_2d):
        #    dists.append(np.sqrt(((np.array(gt_box) - np.array(b)) ** 2).sum()))

        # sel_box_ix = np.argmin(dists)

        # Check that we have the referred object
        # if frame_ix == len(all_frames) - 1 :
        #    assert sum([int(x.token==command.box_token) for x in boxes]) == 1
        #
        # out_dict = {
        #     "sample_token": sample_token,
        #     "map_patch": map_patch_margin,
        #     "egobbox": egobbox,
        #     "map_objects_center": map_objects_center,
        #     "map_objects_elevation": map_objects_elevation,
        #     "map_objects_bbox": map_objects_bbox,
        #     "image_objects_bbox": image_objects_bbox,
        #     "objects_token": objects_token,
        #     "objects_type": objects_type,
        #     "cam_intrinsic": cam_intrinsic,
        #     "ego_translation": ego_translation,
        #     "ego_rotation": ego_rotation,
        #     "cam_translation": cam_translation,
        #     "cam_rotation": cam_rotation,
        # }

        if not os.path.exists(output_folder):
            os.makedirs(
                output_folder
            )

        sample_record = ds.get("sample", sd_rec["sample_token"])
        sample_data_token = sample_record["data"]["LIDAR_TOP"]

        fig, ax = map_explorer.render_map_patch_custom(
            ds,
            sample_data_token,
            limit_left=7,
            limit_right=113,
            limit_top=40,
            limit_bottom=40,
            layer_names=[
                "drivable_area",
                "road_segment",
                "road_block",
                "lane",
                "ped_crossing",
                "walkway",
                "stop_line",
                "carpark_area",
                "road_divider",
                "lane_divider",
                "traffic_light",
            ],
            alpha=0.5,
            figsize=(12, 8),
        )

        # fig, ax = map_explorer.render_map_patch(
        #     map_patch,
        #     layer_names=[
        #         "drivable_area",
        #         "road_segment",
        #         # "road_block",
        #         "lane",
        #         "ped_crossing",
        #         "walkway",
        #         "stop_line",
        #         "carpark_area",
        #         # "road_divider",
        #         # "lane_divider",
        #         # "traffic_light",
        #     ],
        #     figsize=(10, 10),
        #     render_egoposes_range=False,
        #     render_legend=False,
        # )
        plt.axis("off")
        plt.savefig(
            os.path.join(
                output_folder, f"top_down_{split}_{command_ix}.png"
            ),
            #bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        # Copy frontal view
        # copyfile(
        #     impath,
        #     os.path.join(
        #         "extracted_data", "{}_{}".format(split, command_ix), f"frontal_{split}_{command_ix}.jpg"
        #     ),
        # )

        # Store json
        # with open(
        #     os.path.join(
        #         "extracted_data", "{}_{}".format(split, command_ix), f"rotated_frame_{split}_{command_ix}_data.json"
        #     ),
        #     "w",
        # ) as f:
        #     json.dump(out_dict, f)
        #
        #     #nextFrame = command.t2c.get("sample_data", nextFrame)["next"]
        #     #print("going to ", nextFrame, "for video", scene_name)
        #     #frame_ix += 1
        #
        # token = command.frame_token  # command.t2c.get("sample_data", command.frame_token)
        #
        # #sd_rec = ds.get("sample_data", command.frame_token)
        # #impath, boxes, camera_intrinsic = ds.get_sample_data(sd_rec["token"])
        #
        # frame_data = json.load(open(os.path.join(
        #                 "extracted_data", "{}_{}".format(split, command_ix), f"frame_{split}_{command_ix}_data.json"
        #             ),
        #             "r"))
        #
        #
        # to_dump = {
        #             "split": args.split,
        #             "scenes": scene_name,
        #             "command": command.text,
        #             "command_ix": command_ix,
        #             "command_token": command.command_token,
        #             "box_ix": frame_data["objects_token"].index(command.box_token),
        #             #
        #             #  "":
        #             #     [
        #             #     {
        #             #         "box": view_points(
        #             #             x.corners(), camera_intrinsic, normalize=True
        #             #         )[:2, :].tolist(),
        #             #         "is_referred": (x.token == command.box_token),
        #             #         "box_token": x.token,
        #             #     }
        #             #     for x in boxes
        #             # ],
        #         }
        # with open(
        #     os.path.join(
        #         "extracted_data", "{}_{}".format(split, command_ix), f"video_data.json"
        #     ),
        #     "w",
        # ) as f:
        #     json.dump(to_dump,f)
        #
        # print(f"{command_ix}/{len(ds.commands)}")

        # with open(os.path.join( "top_down_sample_data.json"), "w") as f:
        #    json.dump(sample_data, f)


if __name__ == "__main__":
    extract_data_for_t2c_per_video()
