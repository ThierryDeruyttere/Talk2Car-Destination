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

nusc = Talk2Car(version="test", dataroot="data/nuScenes", verbose=True)
predictions_file = "work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/infos_test_10sweeps_withvelo.json"
predictions_data = json.load(open(predictions_file, "r"))

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

def convert_to_2d(box, cam):
    # Get translated corners
    b = np.zeros((900, 1600, 3))

    box.render_cv2(
        b,
        view=cam,
        normalize=True,
        colors=((0, 0, 255), (0, 0, 255), (0, 0, 255)),
    )
    y, x = np.nonzero(b[:, :, 0])

    x1, y1, x2, y2 = map(int, (x.min(), y.min(), x.max(), y.max()))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(1600, x2)
    y2 = min(900, y2)
    return (x1, y1, x2 - x1, y2 - y1)

TACC = 0
for command in nusc.commands:
    sample_data = nusc.get("sample_data", command.frame_token)
    #sample_data["sample_token"] in predictions_data["results"]
    preds = predictions_data["results"][sample_data["sample_token"]]
    impath, boxes, cam_intrinsic = nusc.get_sample_data(sample_data["token"])
    im = Image.open(impath)
    im_size = im.size
    #cam_token = sample_record["data"][camera_channel]
    #cam_record = nusc.get("sample_data", cam_token)
    #cam_path = nusc.get_sample_data_path(cam_token)
    sample_record = nusc.get("sample", sample_data["sample_token"])
    cs_record = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

    sample_data_record = nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
    pose_record = nusc.get("ego_pose", sample_data_record["ego_pose_token"])
    gt = convert_to_2d(command.box, cam_intrinsic)
    gt_x0_y0_x1_y1 = np.array([gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]).reshape(
            1, -1
        )
    keep_objs = []
    for obj in preds:

        box = Box(
            obj["translation"],
            obj["size"],
            Quaternion(obj["rotation"]),
            name=obj["detection_name"],
        )

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        if box_in_image(
                    box,
                    intrinsic=cam_intrinsic,
                    imsize=im_size,
                    vis_level=BoxVisibility.ANY,
            ) and obj["detection_score"] > .0:
            keep_objs.append((obj,box, convert_to_2d(box, cam_intrinsic)))
    

    #print(len(keep_objs))

    # Draw objects
    # im = cv2.imread(impath)
    # for (jsn, box) in keep_objs:
    #     c = nusc.explorer.get_color(box.name)
    #     box.render_cv2(
    #         im, view=cam_intrinsic, normalize=True, colors=(c, c, c)
    #     )
    #
    # # Render
    # im = cv2.resize(im, (640, 360))
    # cv2.imwrite(f"test.png", im)
    # print("")


    # find theoretical accuracy
    iou_list = []
    for (_, _, pred) in keep_objs:
        pred_x0_y0_x1_y1 = np.array(
            [pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3]]
        ).reshape(1, -1)
        iou_list.append(jaccard(gt_x0_y0_x1_y1, pred_x0_y0_x1_y1).squeeze().item())
    if np.sum(np.array(iou_list) > .5) > 0:
        TACC += 1

print("Theoretical acc: ", TACC/len(nusc.commands))