import itertools
from pathlib import Path
import sys
from nuscenes.utils.data_classes import Box
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image
from pyquaternion import Quaternion

sys.path.insert(0, "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/")
sys.path.insert(0, "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/det3d/datasets/nuscenes/")

sys.path.insert(0, "/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/nuscenes-devkit/python-sdk/")
from talk2car import Talk2Car

from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
)

# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import os
import random
import time
from typing import Tuple, Dict, Any

import json
from typing import Dict, Tuple

import numpy as np
import tqdm
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.splits import create_splits_scenes
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample


def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False, only_frontal=False, nusc=None) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'
    if only_frontal:
        new_data = {}
        frontal_samples = {nusc.get("sample_data", x.frame_token)["sample_token"] for x in nusc.commands}
        for sample_token in frontal_samples:
            # Get data from DB
            sample = nusc.get("sample", sample_token)
            cam_token = sample["data"]["CAM_FRONT"]
            cam_record = nusc.get("sample_data", cam_token)
            #cam_path = nusc.get_sample_data_path(cam_token)
            cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            pose_record = nusc.get("ego_pose", cam_record["ego_pose_token"])

            kept_detections = []
            for detections in data["results"][sample_token]:
                    bbox = Box(
                        center=detections["translation"],
                        size=detections["size"],
                        orientation=Quaternion(detections["rotation"]),
                    )

                    bbox_view = bbox.copy()
                    # Move box to ego vehicle coord system.
                    bbox_view.translate(-np.array(pose_record["translation"]))
                    bbox_view.rotate(Quaternion(pose_record["rotation"]).inverse)

                    #  Move box to sensor coord system.
                    bbox_view.translate(-np.array(cs_record["translation"]))
                    bbox_view.rotate(Quaternion(cs_record["rotation"]).inverse)

                    # if sensor_record['modality'] == 'camera' and not \
                    #        box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                    #    continue

                    if box_in_image(
                            bbox_view,
                            intrinsic=cam_intrinsic,
                            imsize=(1600, 900),
                            vis_level=BoxVisibility.ANY,
                    ):
                        kept_detections.append(detections)
            new_data[sample_token] = kept_detections

        data = {"results": new_data, "meta": data["meta"]}

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta


def load_gt(nusc: NuScenes, eval_split: str, box_cls, cleaned_samples, verbose: bool = False, only_frontal=False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    if cleaned_samples:
        sample_tokens_all = [s['token'] for s in cleaned_samples]
    else:
        sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()
    #splits = {"test": list(set([x["scene_token"] for x in cleaned_samples]))}
    # Check compatibility of split with nusc_version.
    version = nusc.version

    if cleaned_samples:
        sample_tokens = sample_tokens_all
    else:
        sample_tokens = []
        for sample_token in sample_tokens_all:
            scene_token = nusc.get('sample', sample_token)['scene_token']
            scene_record = nusc.get('scene', scene_token)
            if scene_record['name'] in splits[eval_split]:
                sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()
    frontal_samples = {nusc.get("sample_data", x.frame_token)["sample_token"] for x in nusc.commands}
    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        if only_frontal and sample_token not in frontal_samples: continue

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        if only_frontal:
            # Get data from DB
            cam_token = sample["data"]["CAM_FRONT"]
            cam_record = nusc.get("sample_data", cam_token)
            cam_path = nusc.get_sample_data_path(cam_token)
            cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            pose_record = nusc.get("ego_pose", cam_record["ego_pose_token"])

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                box = box_cls(
                            sample_token=sample_token,
                            translation=sample_annotation['translation'],
                            size=sample_annotation['size'],
                            rotation=sample_annotation['rotation'],
                            velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                            num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                            detection_name=detection_name,
                            detection_score=-1.0,  # GT samples do not have a score.
                            attribute_name=attribute_name
                        )
                if only_frontal:
                    bbox = Box(
                        center=sample_annotation["translation"],
                        size=sample_annotation["size"],
                        orientation=Quaternion(sample_annotation["rotation"]),
                    )

                    bbox_view = bbox.copy()
                    # Move box to ego vehicle coord system.
                    bbox_view.translate(-np.array(pose_record["translation"]))
                    bbox_view.rotate(Quaternion(pose_record["rotation"]).inverse)

                    #  Move box to sensor coord system.
                    bbox_view.translate(-np.array(cs_record["translation"]))
                    bbox_view.rotate(Quaternion(cs_record["rotation"]).inverse)

                    # if sensor_record['modality'] == 'camera' and not \
                    #        box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                    #    continue

                    if not box_in_image(
                            bbox_view,
                            intrinsic=cam_intrinsic,
                            imsize=(1600, 900),
                            vis_level=BoxVisibility.ANY,
                    ): continue

                sample_boxes.append(box)
            elif box_cls == TrackingBox:
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                from nuscenes.eval.tracking.utils import category_to_tracking_name
                tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    )
                )
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

        return all_annotations


class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 cleaned_samples=None,
                 only_frontal=False):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.only_frontal = only_frontal


        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose, only_frontal=only_frontal, nusc=nusc)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox,
                                cleaned_samples, verbose=verbose, only_frontal=only_frontal)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

        return metrics_summary


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """

def eval_main(nusc, eval_version, res_path, eval_set, output_dir, cleaned_samples):
    # nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        cleaned_samples=cleaned_samples,
        only_frontal=True
    )
    metrics_summary = nusc_eval.main(plot_examples=10,)


tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

mapped_class_names = []
for n in class_names:
    if n in general_to_detection:
        mapped_class_names.append(general_to_detection[n])
    else:
        mapped_class_names.append(n)

nusc = Talk2Car(version="test", dataroot="data/nuScenes", verbose=True)

output_dir = "work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/"
scene_tokens = {c.scene_token: True for c in nusc.commands}
cleaned_samples = [x for x in nusc.sample if x["scene_token"] in scene_tokens]
#nusc.sample = cleaned_samples
#nusc.filter_samples(cleaned_samples, True)

eval_main(
    nusc,
    "detection_cvpr_2019",
    "work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/infos_test_10sweeps_withvelo.json",
    #eval_set_map[self.version],
    "test",
    output_dir,
    cleaned_samples,
)

with open(Path(output_dir) / "metrics_summary.json", "r") as f:
    metrics = json.load(f)


detail = {}
result = f"Talk2car Test Evaluation\n"
for name in mapped_class_names:
    detail[name] = {}
    for k, v in metrics["label_aps"][name].items():
        detail[name][f"dist@{k}"] = v
    threshs = ", ".join(list(metrics["label_aps"][name].keys()))
    scores = list(metrics["label_aps"][name].values())
    mean = sum(scores) / len(scores)
    scores = ", ".join([f"{s * 100:.2f}" for s in scores])
    result += f"{name} Nusc dist AP@{threshs}\n"
    result += scores
    result += f" mean AP: {mean}"
    result += "\n"

res_nusc = {
    "results": {"nusc": result},
    "detail": {"nusc": detail},
}