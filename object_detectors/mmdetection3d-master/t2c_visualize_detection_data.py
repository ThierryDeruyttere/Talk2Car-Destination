from argparse import ArgumentParser
import numpy as np
import json

from PIL import Image, ImageDraw
from os import path as osp
from mmdet3d.datasets.talk2car import Talk2Car


flatten = lambda t: [item for sublist in t for item in sublist]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--num_images', type=int, default=-1, help='Number of images to show per split.')
    args = parser.parse_args()

    data_root = "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root/"
    nuscenes_dbase = "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/mmdetection3d-master/data/nuscenes"
    save_dir = '/home2/NoCsBack/hci/dusan/Results/fcos3d_extracted'
    img_save_dir = '/home2/NoCsBack/hci/dusan/Results/fcos3d_extracted/top_down_vis'

    missing_commands_top = []
    test = Talk2Car(version="train", dataroot=nuscenes_dbase)
    mapping_dict = json.load(open(osp.join(save_dir, 'fcos3d_t2c_mapping.json'), "r"))["feats_mapping"]
    for split in ["train", "val", "test"]:
        top_down_data = json.load(open(osp.join(data_root, f"{split}.json"), "r"))
        box_dicts = json.load(open(osp.join(save_dir, f'fcos3d_t2c_{split}.json'), "r"))
        test.change_version(split)

        num_images = args.num_images if args.num_images > 0 else len(test.commands)
        for ix, cmd in enumerate(test.commands[:num_images]):
            command_token = cmd.command_token
            if not command_token in top_down_data:
                print(command_token, "not in top_down_data")
                missing_commands_top.append(command_token)
                continue
            box_dict = box_dicts[mapping_dict[command_token]]
            pred_boxes_top = np.array(box_dict["2d_boxes_top"])
            pred_boxes_class = np.array(box_dict["classes"])

            """TOP VIEW BOXES"""
            top_down_command_data = top_down_data[command_token][0]

            item = top_down_data[command_token][0]
            img_name = item["top-down"].split("/")[-1]
            img_path = osp.join(data_root, "top_down", img_name)
            image = Image.open(img_path)
            image = image.convert("RGBA")
            referred_obj_top_down_ix = top_down_command_data["command_data"]["box_ix"]
            frame_data = json.load(
                open(
                    osp.join(
                        data_root,
                        "normalized_jsons",
                        "rotated_" + item["frame_data_url"].split("/")[-1],
                    ),
                    "r",
                )
            )
            egobbox = frame_data["egobbox"]
            gt_top = frame_data["map_objects_bbox"]
            ################################################################################################################
            ################################################################################################################

            print("t2c {} {}/{}".format(split, ix, len(test.commands)))
            image_draw = ImageDraw.Draw(image, "RGBA")
            image_draw.polygon(
                flatten(egobbox), fill=(255, 0, 0, 255), outline=(255, 0, 0, 255)
            )
            for ix, map_object_polygon in enumerate(gt_top):
                if ix != referred_obj_top_down_ix:
                    image_draw.polygon(
                        flatten(map_object_polygon),
                        fill=(0, 0, 255, 255),
                        outline=(0, 0, 255, 255),
                    )
                else:
                    image_draw.polygon(
                        flatten(map_object_polygon),
                        fill=(255, 0, 255, 255),
                        outline=(255, 0, 255, 255),
                    )

            for ix, map_object_polygon in enumerate(pred_boxes_top):
                image_draw.polygon(
                    flatten(map_object_polygon),
                    fill=(0, 255, 0, 255),
                    outline=(0, 255, 0, 255),
                )
            image.save(osp.join(img_save_dir, f"top_down_{split}_{cmd.command_token}.png"))

if __name__ == "__main__":
    main()