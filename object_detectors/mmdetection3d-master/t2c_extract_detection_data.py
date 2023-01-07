import os
from argparse import ArgumentParser
import numpy as np
import json
import torch
import copy
import math
from tqdm import tqdm
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from copy import deepcopy
from mmcv.parallel import collate, scatter

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
        ann_file (str): Annotation files.

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

    for (box, attr) in zip(b, attrs):
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
        attr_list.append(attr)
    return box_list, attr_list


flatten = lambda t: [item for sublist in t for item in sublist]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:3', help='Device used for inference'
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

    data_root = "/cw/liir_code/NoCsBack/thierry/PathProjection/data_root/"
    checkpoint = "/cw/liir_code/NoCsBack/thierry/PathProjection/3d_object_detection/mmdetection3d-master/work_dirs/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/latest.pth"
    config = "/cw/liir_code/NoCsBack/thierry/PathProjection/3d_object_detection/mmdetection3d-master/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py"
    nuscenes_dbase = "/cw/liir_code/NoCsBack/thierry/PathProjection/3d_object_detection/mmdetection3d-master/data/nuscenes"

    save_dir = '/cw/liir_code/NoCsBack/thierry/PathProjection/data_root/fcos3d_extracted'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
    test = Talk2Car(version="train", dataroot=nuscenes_dbase)
    for split in ["train", "val", "test"]:
        print(f"Processing split: {split}")
        out_dict = []
        top_down_data = json.load(open(osp.join(data_root, f"{split}.json"), "r"))
        #h5_feats = f.create_dataset("feats", dtype=np.float)
        test.change_version(split)
        for ix, cmd in enumerate(tqdm(test.commands)):
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
            sample_data = test.get("sample_data", frame_token)
            sample_record = test.get("sample", sample_data["sample_token"])
            cs_record = test.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            sample_data_record = test.get("sample_data", sample_record["data"]["CAM_FRONT"])
            pose_record = test.get("ego_pose", sample_data_record["ego_pose_token"])

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
            if 'pts_bbox' in result[0].keys():
                result[0] = result[0]['pts_bbox']
            elif 'img_bbox' in result[0].keys():
                result[0] = result[0]['img_bbox']
            boxes, attrs = output_to_nusc_box(result[0])
            global_objs, _ = cam_nusc_box_to_global(frame_data, boxes, attrs)
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
                             "3d_boxes_corners_front": corners_3d_front.tolist(),
                             "2d_boxes_front": pred_2d_front.tolist(),
                             "2d_boxes_top": pred2d_top.tolist(),
                             "classes": top_classes.tolist(),
                             "boxes_scores": top_scores_3d.tolist()})

            print("t2c {} {}/{}".format(split, ix, len(test.commands)))
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
        json.dump(out_dict, open(osp.join(save_dir, f'fcos3d_t2c_{split}.json'), "w"))
    if args.extract_feats:
        f.create_dataset("feats", data=np.concatenate(feats_list, axis=0))
    json.dump({"feats_mapping": mapping, "class_mapping": model.CLASSES}, open(osp.join(save_dir, 'fcos3d_t2c_mapping.json'), "w"))
    json.dump(missing_commands_top, open(osp.join(save_dir, 'fcos3d_t2c_missing_commands.json'), "w"))

if __name__ == "__main__":
    main()