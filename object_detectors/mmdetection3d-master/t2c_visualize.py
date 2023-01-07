"""

Example command: python demo/t2c_det_demo.py --config configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py --checkpoint work_dirs/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/latest.pth

"""
import copy
from argparse import ArgumentParser
import mmcv
import numpy as np
import re
import torch
from copy import deepcopy
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from os import path as osp
from mmdet3d.core.bbox import mono_cam_box2vis
from mmdet3d.core.bbox import points_cam2img
from mmdet3d.core import (Box3DMode, DepthInstance3DBoxes,
                          LiDARInstance3DBoxes,
                          show_result, show_seg_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox.structures.cam_box3d import CameraInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model

from mmdet3d.apis import (init_model,
                          show_result_meshlab)
from mmdet3d.datasets.talk2car import Talk2Car
import json
import copy
import cv2
import os
import copy
import mmcv
import numpy as np
import pyquaternion
import tempfile
import torch
import warnings
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp


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


def cam_nusc_box_to_global(info,
                           boxes,
                           attrs):
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
        box_list.append(box.bottom_corners()[:2, :].transpose().tolist())
        attr_list.append(attr)
    return box_list, attr_list


import math


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


flatten = lambda t: [item for sublist in t for item in sublist]

root = "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"


def draw_predicted_objects_with_gt_on_top_down(command_token, result):
    item = test_top_down[command_token][0]
    img_name = item["top-down"].split("/")[-1]

    image = Image.open(os.path.join(root, "top_down", img_name)).convert("RGB")
    # (ratio_w, ratio_h) = self.width / orig_w, self.height / orig_h
    # tmp_img = img.copy()
    # Load json
    frame_data = json.load(
        open(
            os.path.join(
                root,
                "normalized_jsons",
                "rotated_" + item["frame_data_url"].split("/")[-1],
            ),
            "r",
        )
    )

    # path, intr = cmd.get_image_path_and_cam_intr()

    # result, data = inference_mono_3d_detector(model, path, intr)
    # print(result)

    # boxes_3d = result[0]["img_bbox"]["boxes_3d"].tensor.numpy()
    # scores_3d = result[0]["img_bbox"]["scores_3d"]
    # labels_3d = result[0]["img_bbox"]["labels_3d"]
    # show_bboxes = CameraInstance3DBoxes(boxes_3d, box_dim=boxes_3d.shape[-1], origin=(0.5, 1.0, 0.5))
    boxes, attrs = output_to_nusc_box(result[0])
    global_objs, _ = cam_nusc_box_to_global(frame_data, boxes, attrs)

    # Convert global to top down
    sample_data = copy.deepcopy(frame_data)
    map_objects_bbox = copy.deepcopy(global_objs)
    egobbox = sample_data["egobbox"]
    ego_translation = sample_data["ego_translation"]
    ego_rotation = sample_data["ego_rotation"]
    # map_objects_bbox = sample_data["map_objects_bbox"]
    # objects_type = sample_data["objects_type"]
    limit_left = 7
    limit_right = 113
    limit_top = 40
    limit_bottom = 40

    canvas_width, canvas_height = image.size

    map_patch_corner_x = ego_translation[0] - limit_left
    map_patch_corner_y = ego_translation[1] - limit_top
    map_patch_width = limit_left + limit_right
    map_patch_height = limit_top + limit_bottom

    yaw = -math.atan2(ego_rotation[1][0], ego_rotation[0][0])

    map_objects_bbox = np.array(map_objects_bbox)
    map_objects_bbox_shape = map_objects_bbox.shape
    map_objects_bbox = map_objects_bbox.reshape(-1, map_objects_bbox_shape[2])
    map_objects_bbox = rotate_points(ego_translation[:2], map_objects_bbox, yaw)
    map_objects_bbox = map_objects_bbox.reshape(map_objects_bbox_shape)

    x = map_objects_bbox[:, :, 0]
    y = map_objects_bbox[:, :, 1]

    x = (x - map_patch_corner_x) / map_patch_width * canvas_width
    y = (1 - (y - map_patch_corner_y) / map_patch_height) * canvas_height
    map_objects_bbox = np.concatenate((x[:, :, None], y[:, :, None]), 2)
    map_objects_polygon = np.concatenate(
        (map_objects_bbox, map_objects_bbox[:, 0, :][:, None, :]), 1
    )

    image_draw = ImageDraw.Draw(image)

    image_draw.polygon(
        flatten(egobbox), fill="#ff0000", outline="#ff0000"
    )

    for map_object_polygon in map_objects_polygon:
        image_draw.polygon(
            flatten(map_object_polygon.tolist()),
            fill="#FFC0CB",
            outline="#FFC0CB",
        )

    for map_object_polygon in sample_data["map_objects_bbox"]:
        image_draw.polygon(
            flatten(map_object_polygon),
            fill="#00ff00",
            outline="#00ff00",
        )

    return image


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
    # print(cfg)
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


def draw_camera_bbox3d_on_img(bboxes3d,
                              raw_img,
                              cam_intrinsic,
                              img_metas,
                              color=(0, 255, 0),
                              thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]):
            3d bbox in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam_intrinsic (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.bbox import points_cam2img

    img = raw_img.copy()
    cam_intrinsic = copy.deepcopy(cam_intrinsic)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam_intrinsic, torch.Tensor):
        cam_intrinsic = torch.from_numpy(np.array(cam_intrinsic))
    cam_intrinsic = cam_intrinsic.reshape(3, 3).float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam_intrinsic)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)


def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               # out_dir,
                               filename,
                               box_mode,
                               img_metas=None,
                               show=False,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str): Coordinate system the boxes are in.
            Should be one of 'depth', 'lidar' and 'camera'.
        img_metas (dict): Used in projecting depth bbox.
        show (bool): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
        pred_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    elif box_mode == 'lidar':
        draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    # result_path = osp.join(out_dir, filename)
    # mmcv.mkdir_or_exist(result_path)
    show_img = img.copy()
    if gt_bboxes is not None:
        show_img = draw_bbox(
            gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
    if pred_bboxes is not None:
        show_img = draw_bbox(
            pred_bboxes,
            show_img,
            proj_mat,
            img_metas,
            color=pred_bbox_color)
    return show_img


def imshow(img):
    import cv2
    import IPython
    _, ret = cv2.imencode('.png', img)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


from mmdet3d.core.bbox import mono_cam_box2vis
from PIL import Image, ImageDraw


def visualize(cmd, thresh):
    path, intr = cmd.get_image_path_and_cam_intr()
    result, data = inference_mono_3d_detector(model, path, intr)
    # print(result)

    mask = result[0]["img_bbox"]["scores_3d"] > thresh
    result[0]["img_bbox"]["boxes_3d"] = result[0]["img_bbox"]["boxes_3d"][mask]
    result[0]["img_bbox"]["scores_3d"] = result[0]["img_bbox"]["scores_3d"][mask]
    result[0]["img_bbox"]["labels_3d"] = result[0]["img_bbox"]["labels_3d"][mask]
    boxes_3d = result[0]["img_bbox"]["boxes_3d"].tensor.numpy()

    assert len(boxes_3d) == mask.sum()

    if len(result[0]["img_bbox"]["boxes_3d"]) == 0:
        print("No bounding boxes with threshold {}".format(thresh))
        return

    # print(scores_3d)
    # print(boxes_3d)
    show_bboxes = CameraInstance3DBoxes(boxes_3d, box_dim=boxes_3d.shape[-1], origin=(0.5, 1.0, 0.5))
    # print(show_bboxes)
    # TODO: remove the hack of box from NuScenesMonoDataset
    show_bboxes = mono_cam_box2vis(show_bboxes)

    cam_intrinsic = deepcopy(intr)

    corners_3d = show_bboxes.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam_intrinsic, torch.Tensor):
        cam_intrinsic = torch.from_numpy(np.array(cam_intrinsic))
    cam_intrinsic = cam_intrinsic.reshape(3, 3).float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam_intrinsic)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    xy_max = imgfov_pts_2d.max(1)
    x_max, y_max = xy_max[:, 0].clip(0, 1600), xy_max[:, 1].clip(0, 900)
    xy_min = imgfov_pts_2d.min(1)
    x_min, y_min = xy_min[:, 0].clip(0, 1600), xy_min[:, 1].clip(0, 900)
    gt = convert_to_2d(cmd.box, cam_intrinsic)
    gt_x0_y0_x1_y1 = np.array([gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]).reshape(1, -1)
    pred_2d = np.array([x_min, y_min, x_max, y_max]).transpose(1, 0)

    img_filename = data['img_metas'][0][0]['filename']
    file_name = osp.split(img_filename)[-1].split('.')[0]

    # read from file because img in data_dict has undergone pipeline transform
    img = mmcv.imread(img_filename)

    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    # pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    # pred_scores = result[0]['scores_3d'].numpy()

    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if 'cam_intrinsic' not in data['img_metas'][0][0]:
        raise NotImplementedError(
            'camera intrinsic matrix is not provided')

    # show_bboxes = CameraInstance3DBoxes(
    #    pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))
    # TODO: remove the hack of box from NuScenesMonoDataset
    # show_bboxes = mono_cam_box2vis(show_bboxes)

    drawn_img = show_multi_modality_result(
        img,
        None,
        show_bboxes,
        data['img_metas'][0][0]['cam_intrinsic'],
        # out_dir,
        file_name,
        box_mode='camera',
        show=True)
    # print(pred_2d)

    # Draw 2d boxes
    # im = Image.fromarray(drawn_img).convert('RGB')
    # drw = ImageDraw.Draw(im)
    # for box in pred_2d:
    #
    #    drw.rectangle(box, outline="blue", width=2)
    #
    # drawn_img = np.array(im)
    imshow(drawn_img)

    # Top down
    # print(result)
    return draw_predicted_objects_with_gt_on_top_down(cmd.command_token, result)


if __name__ == "__main__":

    test_top_down = json.load(open("/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root/test.json", "r"))
    config = "configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py"
    checkpoint = "work_dirs/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/latest.pth"

    model = init_model(config, checkpoint, device="cuda:0")

    test = Talk2Car(version="test", dataroot="data/nuscenes/", verbose=True)

    # model.test_cfg["use_rotate_nms"] = True
    # model.test_cfg["max_per_img"] = 1000
    # model.test_cfg["score_thr"] = 0.1
    # This is from nuscenes_mono_dataset.py
    # test_cfg = dict(
    #     use_rotate_nms=True,
    #     nms_across_levels=False,
    #     nms_pre=10,
    #     nms_thr=0.9,
    #     score_thr=0.9,
    #     min_bbox_size=0,
    #     max_per_frame=1)

    model.bbox_head.test_cfg.nms_pre = 1000
    model.bbox_head.test_cfg.nms_thr = 0.1
    model.bbox_head.test_cfg.score_thr = 0.1

    visualize(test.commands[100], thresh=0)
