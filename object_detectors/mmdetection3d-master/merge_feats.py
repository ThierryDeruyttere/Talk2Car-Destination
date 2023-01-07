
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
import os

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



    with h5py.File('/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root/fcos3d/fcos3d_t2c_feats_v2.h5', 'w') as f:

        feats = np.array(
            h5py.File("/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root/fcos3d/fcos3d_t2c_feats.h5",
                      "r")["feats"])

        f.create_dataset("feats", data=feats.reshape((11959, 64, 1536)))
main()