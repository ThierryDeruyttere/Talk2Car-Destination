import os
from argparse import ArgumentParser
import numpy as np
import json
import torch
import copy
import math
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from copy import deepcopy
from mmcv.parallel import collate, scatter
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image
from pyquaternion import Quaternion

from PIL import Image, ImageDraw
from os import path as osp
from mmdet3d.core.bbox import mono_cam_box2vis
from mmdet3d.core.bbox import points_cam2img
from mmdet3d.core import Box3DMode
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox.structures.cam_box3d import CameraInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose
import h5py
from mmdet3d.apis import init_model
from mmdet3d.datasets.talk2car import Talk2Car
import iou_3d

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


def inference_mono_3d_detector(model, image, cam_intr):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (str): Image files.
        cam_intr: Camera intrinsics.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    # find the info corresponding to this image
    data = dict(
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(cam_intrinsic=cam_intr))

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data

def output_to_nusc_box(detection, chosen_indices=None):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
            - attrs_3d (torch.Tensor, optional): Predicted attributes.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    attrs = None
    if 'attrs_3d' in detection:
        attrs = detection['attrs_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    box_list = []
    for i in range(len(box3d)):
        q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        quat = q2 * q1
        velocity = (box3d.tensor[i, 7], 0.0, box3d.tensor[i, 8])
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list, attrs


def cam_nusc_box_to_global(info, boxes, attrs):
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    attr_list = []
    b = copy.deepcopy(boxes)

    for box in b:
        # Move box to ego vehicle coord system
        # cam_rot = np.zeros((4,4))
        # cam_rot[:3, :3] = info['cam_rotation']
        # cam_rot[3,3] = 1
        # print(cam_rot)
        box.rotate(pyquaternion.Quaternion._from_matrix(np.array(info['cam_rotation'])))
        box.translate(np.array(info['cam_translation']))

        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion._from_matrix(np.array(info['ego_rotation'])))
        box.translate(np.array(info['ego_translation']))
        box_list.append(box)
    return box_list


flatten = lambda t: [item for sublist in t for item in sublist]


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
            - attrs_3d (torch.Tensor, optional): Predicted attributes.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["img_bbox"]['boxes_3d']
    scores = detection["img_bbox"]['scores_3d'].numpy()
    labels = detection["img_bbox"]['labels_3d'].numpy()
    attrs = None
    if 'attrs_3d' in detection:
        attrs = detection["img_bbox"]['attrs_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    box_list = []
    for i in range(len(box3d)):
        q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        quat = q2 * q1
        velocity = (box3d.tensor[i, 7], 0.0, box3d.tensor[i, 8])
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list, attrs


def inference_mono_3d_detector(model, image, cam_intr):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    #print(cfg)
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    # find the info corresponding to this image
    data = dict(
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(cam_intrinsic=cam_intr))

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference'
    )
    parser.add_argument(
        '--top-k', type=int, default=64, help='number of chosen top boxes'
    )
    parser.add_argument(
        '--visualize', action='store_true', help='show online visuliaztion results'
    )
    parser.add_argument(
        "--extract_feats", action='store_true', help='extract features'
    )
    args = parser.parse_args()

    data_root = "../data_root/"
    checkpoint = "checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/latest.pth"
    config = "configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py"
    nuscenes_dbase = "data/nuscenes"

    save_dir = '../data_root/fcos3d_extracted'

    visualize = args.visualize

    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=args.device)
    #model.test_cfg["max_per_img"] = 100
    #model.bbox_head.test_cfg.nms_pre = 1000

    #model.bbox_head.test_cfg.nms_thr = 0.1
    #model.bbox_head.test_cfg.score_thr = 0
    model.bbox_head.test_cfg.nms_pre = 4096
    model.bbox_head.test_cfg.nms_thr = 0.05
    model.bbox_head.test_cfg.score_thr = 0.0
    model.bbox_head.test_cfg.max_per_frame = 500

    # test a single image
    if args.extract_feats:
        f = h5py.File(osp.join(save_dir, 'fcos3d_t2c_feats.h5'), 'w')
        feats_list = []
    mapping = {}
    missing_commands_top = []
    talk2car = Talk2Car(version="train", dataroot=nuscenes_dbase)
    root = "../data_root"

    for split in ["train", "val", "test"]:
        out_dict = []
        top_down_data = json.load(open(osp.join(data_root, f"{split}.json"), "r"))
        #h5_feats = f.create_dataset("feats", dtype=np.float)
        talk2car.change_version(split)
        top_down = json.load(open("../data_root/{}.json".format(split), "r"))

        for ix, cmd in enumerate(talk2car.commands):
            command_token = cmd.command_token
            if not command_token in top_down_data:
                print(command_token, "not in top_down_data")
                missing_commands_top.append(command_token)
                continue
            path, intr = cmd.get_image_path_and_cam_intr()
            result, data = inference_mono_3d_detector(model, path, intr)

            boxes_3d = result[0]["img_bbox"]["boxes_3d"].tensor.numpy()
            top_scores_3d, top_indices = result[0]["img_bbox"]["scores_3d"].topk(args.top_k)
            top_classes = result[0]["img_bbox"]["labels_3d"][top_indices]
            boxes_3d = boxes_3d[top_indices]

            if args.extract_feats:
                top_feats = result[0]["img_bbox"]["feats"][top_indices]
                feats_list.append(top_feats.unsqueeze(0).tolist())

            boxes_3d = CameraInstance3DBoxes(boxes_3d, box_dim=boxes_3d.shape[-1], origin=(0.5, 1.0, 0.5))

            #### NEW

            referred_object_3d_info, predicted_3d_boxes_info = get_referred_and_predicted_boxes_in_same_3d_frame(cmd, result, root,
                                                                                                talk2car, top_down,
                                                                                                top_indices)

            # box_center = boxes_3d.center #xyz
            # #box_center_transformed = list(zip(box_center[:, 2].tolist(), box_center[:, 0].tolist(), box_center[:, 1].tolist()))
            # box_whl = boxes_3d.dims
            # box_rot = boxes_3d.yaw
            # pred_box_params = list(zip(box_whl[:, 2].tolist(), box_whl[:, 0].tolist(), box_whl[:, 1].tolist(),
            #                            box_rot.tolist(), box_center.tolist()))
            # #pred_box_params_transformed = list(zip(box_whl[:, 2].tolist(), box_whl[:, 0].tolist(), box_whl[:, 1].tolist(),
            # #                           box_rot.tolist(), box_center_transformed))
            # pred_boxes_3d = [iou_3d.get_3d_box(x[:3], x[3], x[4]) for x in pred_box_params]
            # #pred_boxes_3d_transformed = [iou_3d.get_3d_box(x[:3], x[3], x[4]) for x in pred_box_params_transformed]
            #
            # gt_box_lwh = [cmd.box.wlh[1], cmd.box.wlh[0], cmd.box.wlh[2]]
            # referred_3d = iou_3d.get_3d_box(referred_object_3d_info["wlh"], referred_object_3d_info["rad"],
            #                                 referred_object_3d_info["center"])
            # #
            # pred_boxes_3d = [iou_3d.get_3d_box(x["wlh"], x["rad"],
            #                                 x["center"]) for x in predicted_3d_boxes_info]
            # r1 = [iou_3d.box3d_iou(x, referred_3d) for x in pred_boxes_3d]
            # #r2 = [iou_3d.box3d_iou(x, referred_3d) for x in pred_boxes_3d_transformed]
            ### END NEW

            """FRONT VIEW BOXES"""
            boxes_front = mono_cam_box2vis(boxes_3d)

            cam_intrinsic = deepcopy(intr)
            if len(boxes_front) == 0: continue

            corners_3d_front = boxes_front.corners
            num_bbox = corners_3d_front.shape[0]
            points_3d_front = corners_3d_front.reshape(-1, 3)
            if not isinstance(cam_intrinsic, torch.Tensor):
                cam_intrinsic = torch.from_numpy(np.array(cam_intrinsic))
            cam_intrinsic = cam_intrinsic.reshape(3, 3).float().cpu()

            # project to 2d to get image coords (uv)
            uv_origin = points_cam2img(points_3d_front, cam_intrinsic)
            uv_origin = (uv_origin - 1).round()
            imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

            xy_max = imgfov_pts_2d.max(1)
            x_max, y_max = xy_max[:, 0].clip(0, 1600), xy_max[:, 1].clip(0, 900)
            xy_min = imgfov_pts_2d.min(1)
            x_min, y_min = xy_min[:, 0].clip(0, 1600), xy_min[:, 1].clip(0, 900)
            gt = convert_to_2d(cmd.box, cam_intrinsic)
            #gt_x0_y0_x1_y1 = np.array([gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]).reshape(1, -1)
            pred_2d_front = np.array([x_min, y_min, x_max, y_max]).transpose(1,0)

            ################################################################################################################
            ################################################################################################################
            ################################################################################################################
            ################################################################################################################

            """TOP VIEW BOXES"""
            frame_token = cmd.frame_token
            sample_data = talk2car.get("sample_data", frame_token)
            sample_record = talk2car.get("sample", sample_data["sample_token"])
            cs_record = talk2car.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            sample_data_record = talk2car.get("sample_data", sample_record["data"]["CAM_FRONT"])
            pose_record = talk2car.get("ego_pose", sample_data_record["ego_pose_token"])

            top_down_command_data = top_down_data[command_token][0]

            item = top_down_data[command_token][0]
            img_name = item["top-down"].split("/")[-1]
            img_path = osp.join(data_root, "top_down", img_name)
            image = Image.open(img_path)

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

            map_patch = frame_data["map_patch"]
            egobbox = frame_data["egobbox"]
            ego_translation = frame_data["ego_translation"]
            ego_rotation = frame_data["ego_rotation"]
            yaw = -math.atan2(ego_rotation[1][0], ego_rotation[0][0])

            limit_left = 7
            limit_right = 113
            limit_top = 40
            limit_bottom = 40

            map_patch_corner_x = ego_translation[0] - limit_left
            map_patch_corner_y = ego_translation[1] - limit_top
            map_patch_width = limit_left + limit_right
            map_patch_height = limit_top + limit_bottom
            ################################################################################################################
            ################################################################################################################
            # GT TOP DOWN
            gt_top = frame_data["map_objects_bbox"]
            ################################################################################################################
            ################################################################################################################
            # Predicted TOP DOWN
            #if 'pts_bbox' in result[0].keys():
            #    result[0] = result[0]['pts_bbox']
            #elif 'img_bbox' in result[0].keys():
            #    result[0] = result[0]['img_bbox']
            boxes, attrs = output_to_nusc_box(result[0])
            global_objs = cam_nusc_box_to_global(frame_data, boxes, attrs)
            canvas_width, canvas_height = image.size

            pred_map_objects_bbox = []
            for pred in global_objs:
                map_object_bbox = pred.bottom_corners()[:2, :].transpose().tolist()
                pred_map_objects_bbox.append(map_object_bbox)

            map_objects_bbox = np.array(pred_map_objects_bbox)
            map_objects_bbox_shape = map_objects_bbox.shape
            map_objects_bbox = map_objects_bbox.reshape(-1, map_objects_bbox_shape[2])
            map_objects_bbox = rotate_points(ego_translation[:2], map_objects_bbox, yaw)
            map_objects_bbox = map_objects_bbox.reshape(map_objects_bbox_shape)
            # referred_obj_coords = copy.deepcopy(map_objects_bbox[referred_obj])

            x = map_objects_bbox[:, :, 0]
            y = map_objects_bbox[:, :, 1]

            x = (x - map_patch_corner_x) / map_patch_width * canvas_width
            y = (1 - (y - map_patch_corner_y) / map_patch_height) * canvas_height
            pred2d_top = np.concatenate((x[:, :, None], y[:, :, None]), 2)

            #Pick the best 64
            pred2d_top = pred2d_top[top_indices]

            out_dict.append({"command_token": cmd.command_token,
                             "referred_box_3d": referred_object_3d_info,
                             "predicted_boxes_3d": predicted_3d_boxes_info,
                             "3d_boxes_corners_front": corners_3d_front.tolist(),
                             "2d_boxes_front": pred_2d_front.tolist(),
                             "2d_boxes_top": pred2d_top.tolist(),
                             "classes": top_classes.tolist(),
                             "boxes_scores": top_scores_3d.tolist()})

            print("t2c {} {}/{}".format(split, ix, len(talk2car.commands)))
            mapping[cmd.command_token] = len(mapping)

            if visualize:
                image_draw = ImageDraw.Draw(image)
                image_draw.polygon(
                    flatten(egobbox), fill="#ff0000", outline="#ff0000"
                )
                for ix, map_object_polygon in enumerate(gt_top):
                    if ix != referred_obj_top_down_ix:
                        image_draw.polygon(
                            flatten(map_object_polygon),
                            fill="#0000ff",
                            outline="#0000ff",
                        )
                    else:
                        image_draw.polygon(
                            flatten(map_object_polygon),
                            fill="#ff00ff",
                            outline="#ff00ff",
                        )

                for ix, map_object_polygon in enumerate(pred2d_top):
                    image_draw.polygon(
                        flatten(map_object_polygon),
                        fill="#00ff00",
                        outline="#00ff00",
                    )
                image.save(osp.join(save_dir, f"top_down_{cmd.command_token}.png"))
        json.dump(out_dict, open(osp.join(save_dir, f'fcos3d_t2c_{split}_with_3d_info.json'), "w"))
    if args.extract_feats:
        f.create_dataset("feats", data=np.concatenate(feats_list, axis=0))
    json.dump({"feats_mapping": mapping, "class_mapping": model.CLASSES}, open(osp.join(save_dir, 'fcos3d_t2c_mapping_with_3d_info.json'), "w"))
    json.dump(missing_commands_top, open(osp.join(save_dir, 'fcos3d_t2c_missing_commands_with_3d_info.json'), "w"))


def get_referred_and_predicted_boxes_in_same_3d_frame(cmd, result, root, talk2car, top_down, top_indices):
    frame_token = cmd.frame_token
    frame_data = json.load(
        open(
            os.path.join(
                root,
                "normalized_jsons",
                "rotated_" + top_down[cmd.command_token][0]["frame_data_url"].split("/")[-1],
            ),
            "r",
        )
    )
    sample_data = talk2car.get("sample_data", frame_token)
    sample_record = talk2car.get("sample", sample_data["sample_token"])
    cs_record = talk2car.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    sample_data_record = talk2car.get("sample_data", sample_record["data"]["CAM_FRONT"])
    pose_record = talk2car.get("ego_pose", sample_data_record["ego_pose_token"])
    # ref_ix = top_down[cmd.command_token][0]["command_data"]["box_ix"]
    # _, frontal_gt_boxes, _ = talk2car.get_sample_data(frame_token, use_flat_vehicle_coordinates=False)
    # ref_box = frontal_gt_boxes[ref_ix]
    # box_list = []
    gt_box = cmd.box.copy()
    boxes, attrs = output_to_nusc_box(result[0])
    boxes_est_frontal = cam_nusc_box_to_global(frame_data, boxes[:len(top_indices)], attrs)
    predicted_3d_boxes_info = []
    for obj in boxes_est_frontal:
        obj.translate(-np.array(pose_record['translation']))
        obj.rotate(pyquaternion.Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        obj.translate(-np.array(cs_record['translation']))
        obj.rotate(pyquaternion.Quaternion(cs_record['rotation']).inverse)
        predicted_3d_boxes_info.append({"center": obj.center.tolist(),
                                        "wlh": obj.wlh.tolist(),
                                        "rad": obj.orientation.radians,
                                        "corners": obj.corners().tolist()})

    referred_object_3d_info = {"center": gt_box.center.tolist(),
                                                 "wlh": gt_box.wlh.tolist(),
                                                 "rad": gt_box.orientation.radians,
                                                 "corners": gt_box.corners().tolist()}
    return referred_object_3d_info, predicted_3d_boxes_info


if __name__ == "__main__":
    main()
