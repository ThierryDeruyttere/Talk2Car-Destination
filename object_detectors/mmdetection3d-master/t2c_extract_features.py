
from argparse import ArgumentParser
import mmcv
import numpy as np
import re
import json
import torch
from copy import deepcopy
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from os import path as osp
from mmdet3d.core.bbox import mono_cam_box2vis
from mmdet3d.core.bbox import points_cam2img
from mmdet3d.core import (Box3DMode, DepthInstance3DBoxes,
                          LiDARInstance3DBoxes, show_multi_modality_result,
                          show_result, show_seg_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox.structures.cam_box3d import CameraInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
import h5py
from mmdet3d.apis import (init_model,
                          show_result_meshlab)
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


def main():
    parser = ArgumentParser()
    #parser.add_argument('image', help='image file')
    #parser.add_argument('ann', help='ann file')
    #parser.add_argument('--config', help='Config file')
    #parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:3', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0, help='bbox score threshold')
    parser.add_argument('--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show', action='store_true', help='show online visuliaztion results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visuliaztion results')
    args = parser.parse_args()

    config = "configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py"
    checkpoint = "work_dirs/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/latest.pth"
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
    f = h5py.File('fcos3d_t2c_feats.h5', 'w')
    feats_list = []
    mapping = {}
    test = Talk2Car(version="train", dataroot="data/nuscenes/")
    for split in ["train", "val", "test"]:
        out_dict = []
        #h5_feats = f.create_dataset("feats", dtype=np.float)
        test.change_version(split)
        for ix, cmd in enumerate(test.commands):
            path, intr = cmd.get_image_path_and_cam_intr()
            result, data = inference_mono_3d_detector(model, path, intr)

            boxes_3d = result[0]["img_bbox"]["boxes_3d"].tensor.numpy()
            top_scores_3d, top_indices = result[0]["img_bbox"]["scores_3d"].topk(64)
            top_feats = result[0]["img_bbox"]["feats"][top_indices]
            top_classes = result[0]["img_bbox"]["labels_3d"][top_indices]
            boxes_3d = boxes_3d[top_indices]

            feats_list.append(top_feats.unsqueeze(0).tolist())
            show_bboxes = CameraInstance3DBoxes(boxes_3d, box_dim=boxes_3d.shape[-1], origin=(0.5, 1.0, 0.5))
            # TODO: remove the hack of box from NuScenesMonoDataset
            show_bboxes = mono_cam_box2vis(show_bboxes)

            cam_intrinsic = deepcopy(intr)
            if len(show_bboxes) == 0: continue

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
            #gt_x0_y0_x1_y1 = np.array([gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]).reshape(1, -1)
            pred_2d = np.array([x_min, y_min, x_max, y_max]).transpose(1,0)

            out_dict.append({"command_token": cmd.command_token,
                             "3d_boxes_corners": corners_3d.tolist(),
                             "2d_boxes": pred_2d.tolist(),
                             "classes": top_classes.tolist(),
                             "boxes_scores": top_scores_3d.tolist()})

            print("t2c {} {}/{}".format(split, ix, len(test.commands)))
            mapping[cmd.command_token] = len(mapping)

        json.dump(out_dict, open('fcos3d_t2c_{}.json'.format(split), "w"))

    f.create_dataset("feats", data=np.concatenate(feats_list, axis=0))
    json.dump({"feats_mapping": mapping,
               "class_mapping": model.CLASSES},
              open('fcos3d_t2c_mapping.json', "w"))


main()