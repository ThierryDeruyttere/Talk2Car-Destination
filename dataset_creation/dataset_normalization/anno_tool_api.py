# nuScenes dev-kit.
# Code written by Sergi Adipraja Widjaja, 2019.
# + Map mask by Kiwoo Shin, 2019.
# + Methods operating on NuScenesMap and NuScenes by Holger Caesar, 2019.

from typing import List, Tuple, Union, Dict, Any

import math
import descartes
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Arrow
from matplotlib import transforms
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from shapely.geometry import Polygon, LineString
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image

# Recommended style to use as the plots will show grids.
plt.style.use("seaborn-whitegrid")

# Define a map geometry type for polygons and lines.
Geometry = Union[Polygon, LineString]


def return_side_points(
    cur_point: Union[Tuple, List],
    prev_point: Union[Tuple, List, None] = None,
    thickness=2.0,
):
    if prev_point == None:
        return cur_point, cur_point
    else:
        line = LineString([cur_point, prev_point])
        left = line.parallel_offset(thickness / 2, "left")
        right = line.parallel_offset(thickness / 2, "right")
        return left.boundary[1], right.boundary[0]


def compute_polygon_from_path(path_nodes: List[List], thickness=2.0):
    prev_point = None
    forward = []
    backward = []
    for cur_point in path_nodes:
        left, right = return_side_points(cur_point, prev_point, thickness)
        forward.append(left)
        backward.append(right)
        prev_point = cur_point
    forward = forward + [path_nodes[-1]]
    backward = backward + [path_nodes[-1]]
    backward = backward[::-1]
    return Polygon(forward + backward)


class NuScenesMapT2C(NuScenesMap):
    """
    NuScenesMap database class for querying and retrieving information from the semantic maps.
    Before using this class please use the provided tutorial `map_expansion_tutorial.ipynb`.

    Below you can find the map origins (south western corner, in [lat, lon]) for each of the 4 maps in nuScenes:
    boston-seaport: [42.336849169438615, -71.05785369873047]
    singapore-onenorth: [1.2882100868743724, 103.78475189208984]
    singapore-hollandvillage: [1.2993652317780957, 103.78217697143555]
    singapore-queenstown: [1.2782562240223188, 103.76741409301758]

    The dimensions of the maps are as follows ([width, height] in meters):
    singapore-onenorth: [1585.6, 2025.0]
    singapore-hollandvillage: [2808.3, 2922.9]
    singapore-queenstown: [3228.6, 3687.1]
    boston-seaport: [2979.5, 2118.1]
    The rasterized semantic maps (e.g. singapore-onenorth.png) published with nuScenes v1.0 have a scale of 10px/m,
    hence the above numbers are the image dimensions divided by 10.

    We use the same WGS 84 Web Mercator (EPSG:3857) projection as Google Maps/Earth.
    """

    def __init__(
        self,
        dataroot: str = "/data/sets/nuscenes",
        map_name: str = "singapore-onenorth",
    ):
        """
        Loads the layers, create reverse indices and shortcuts, initializes the explorer class.
        :param dataroot: Path to the layers in the form of a .json file.
        :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown`,
        `boston-seaport` that we want to load.
        """
        super().__init__(dataroot, map_name)
        assert map_name in [
            "singapore-onenorth",
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
        ]
        self.explorer = NuScenesMapExplorerT2C(self)


class NuScenesMapExplorerT2C(NuScenesMapExplorer):
    """ Helper class to explore the nuScenes map data. """

    def __init__(
        self,
        map_api: NuScenesMap,
        representative_layers: Tuple[str] = ("drivable_area", "lane", "walkway"),
        color_map: dict = None,
        object_color_map: dict = None,
    ):
        """
        :param map_api: NuScenesMap database class.
        :param representative_layers: These are the layers that we feel are representative of the whole mapping data.
        :param color_map: Color map.
        :param object_color_map: Color mapping from object types.
        """
        super().__init__(map_api, representative_layers, color_map)

        if object_color_map is None:
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

        self.object_color_map = object_color_map

    def render_surroundings_in_image(
        self,
        nusc: NuScenes,
        sample_token: str,
        path_nodes: List[List] = None,
        path_thickness: float = 2.0,
        camera_channel: str = "CAM_FRONT",
        alpha: float = 0.3,
        patch_radius: float = 10000,
        render_objects: bool = True,
        verbose: bool = True,
        out_path: str = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Render a nuScenes camera image and overlay the polygons for the specified map layers.
        Note that the projections are not always accurate as the localization is in 2d.
        :param nusc: The NuScenes instance to load the image from.
        :param sample_token: The image's corresponding sample_token.
        :param path_nodes: Set of path nodes, points in (x,y) relative to ego pose (ego pose is (0,0))
        :param path_thickness: Thickness of the path polygon.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param alpha: The transparency value of the layers to render in [0, 1].
        :param patch_radius: The radius in meters around the ego car in which to select map records.
        :param render_objects: Whether to render bounding boxes of all objects in the image.
        :param verbose: Whether to print to stdout.
        :param out_path: Optional path to save the rendered figure to disk.
        """
        near_plane = 1e-8

        if verbose:
            print(
                "Warning: Note that the projections are not always accurate as the localization is in 2d."
            )

        # Check that NuScenesMap was loaded for the correct location.
        sample_record = nusc.get("sample", sample_token)
        scene_record = nusc.get("scene", sample_record["scene_token"])
        log_record = nusc.get("log", scene_record["log_token"])
        log_location = log_record["location"]
        assert (
            self.map_api.map_name == log_location
        ), "Error: NuScenesMap loaded for location %s, should be %s!" % (
            self.map_api.map_name,
            log_location,
        )

        # Grab the front camera image and intrinsics.
        cam_token = sample_record["data"][camera_channel]
        cam_record = nusc.get("sample_data", cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)
        im = Image.open(cam_path)
        im_size = im.size
        cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])

        # Retrieve the current map.
        pose_record = nusc.get("ego_pose", cam_record["ego_pose_token"])
        egopose = pose_record["translation"]

        box_coords = (
            egopose[0] - patch_radius,
            egopose[1] - patch_radius,
            egopose[0] + patch_radius,
            egopose[1] + patch_radius,
        )

        # Init axes.
        fig = plt.figure(figsize=(9, 16))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, im_size[0])
        ax.set_ylim(0, im_size[1])
        ax.imshow(im)

        if path_nodes:
            # Compute polygon from path
            polygon = compute_polygon_from_path(path_nodes, thickness=path_thickness)

            # Convert polygon nodes to pointcloud with 0 height.
            points = np.array(polygon.exterior.xy)
            points = np.vstack((points, np.zeros((1, points.shape[1]))))

            # Transform into the ego vehicle frame for the timestamp of the image.
            points = points - np.array(pose_record["translation"]).reshape((-1, 1))
            points = np.dot(
                Quaternion(pose_record["rotation"]).rotation_matrix.T, points
            )

            # Transform into the camera.
            points = points - np.array(cs_record["translation"]).reshape((-1, 1))
            points = np.dot(Quaternion(cs_record["rotation"]).rotation_matrix.T, points)
            # Remove points that are partially behind the camera.
            depths = points[2, :]
            behind = depths < near_plane
            if np.all(behind):
                print("Path is completely behind the camera view...")
            else:
                # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                points = view_points(points, cam_intrinsic, normalize=True)

                # Skip polygons where all points are outside the image.
                # Leave a margin of 1 pixel for aesthetic reasons.
                inside = np.ones(points.shape[1], dtype=bool)
                inside = np.logical_and(inside, points[0, :] > 1)
                inside = np.logical_and(inside, points[0, :] < im.size[0] - 1)
                inside = np.logical_and(inside, points[1, :] > 1)
                inside = np.logical_and(inside, points[1, :] < im.size[1] - 1)
                if np.all(np.logical_not(inside)):
                    print("Path is completely outside the image...")
                else:
                    # Drawing path
                    points = points[:2, :]
                    points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                    polygon_proj = Polygon(points)
                    ax.add_patch(
                        descartes.PolygonPatch(
                            polygon_proj,
                            fc="#00ff00",
                            alpha=alpha,
                            label="path",
                        )
                    )

        image_objects = []
        image_objects_bbox = []
        image_objects_color = []
        image_objects_type = []
        if verbose:
            print("Adding ego pose and object poses to map...")
        # Poses are associated with the sample_data. Here we use the lidar sample_data.
        sample_data_record = nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
        pose_record = nusc.get("ego_pose", sample_data_record["ego_pose_token"])

        annotation_tokens = sample_record["anns"]
        for annotation_token in annotation_tokens:
            object_record = nusc.get("sample_annotation", annotation_token)

            category_name = object_record["category_name"]
            color = self.object_color_map[category_name]

            image_objects.append(object_record["translation"])

            # Get 3D box
            bbox = Box(
                center=object_record["translation"],
                size=object_record["size"],
                orientation=Quaternion(object_record["rotation"]),
            )
            # Move box to ego vehicle coord system.
            bbox.translate(-np.array(pose_record["translation"]))
            bbox.rotate(Quaternion(pose_record["rotation"]).inverse)

            #  Move box to sensor coord system.
            bbox.translate(-np.array(cs_record["translation"]))
            bbox.rotate(Quaternion(cs_record["rotation"]).inverse)

            if box_in_image(
                bbox,
                intrinsic=cam_intrinsic,
                imsize=im_size,
                vis_level=BoxVisibility.ANY,
            ):
                image_objects_bbox.append(bbox)
                image_objects_color.append(color)
                image_objects_type.append(category_name)
                if render_objects:
                    bbox.render(
                        axis=ax,
                        view=cam_intrinsic,
                        normalize=True,
                        colors=(
                            color,
                            color,
                            color,
                        ),
                        linewidth=1,
                    )

        # Display the image.
        plt.axis("off")
        ax.invert_yaxis()

        if out_path is not None:
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0)

        ego_translation = np.array(pose_record["translation"])
        ego_rotation = np.array(Quaternion(pose_record["rotation"]).rotation_matrix)
        cam_translation = np.array(cs_record["translation"])
        cam_rotation = np.array(Quaternion(cs_record["rotation"]).rotation_matrix)
        image_objects = np.array(image_objects)
        image_objects_bbox = np.array(image_objects_bbox)
        image_objects_type = np.array(image_objects_type)

        return (
            ego_translation,
            ego_rotation,
            cam_translation,
            cam_rotation,
            image_objects,
            image_objects_bbox,
            image_objects_type,
            cam_intrinsic,
        )


    def render_surroundings_on_fancy_map(
        self,
        nusc: NuScenes,
        sample_token: str,
        verbose: bool = True,
        out_path: str = None,
        patch_margin: int = 10,
        render_car: bool = True,
        render_objects: bool = True,
        only_visible_objects: bool = True,
        camera_channel: str = "CAM_FRONT",
        path_nodes: List[List] = None,
        path_thickness: float = 2.0,
    ):
        """
        Renders each ego pose and positions of other objects of a list of scenes on the map (around 40 poses per scene).
        This method is heavily inspired by NuScenes.render_egoposes_on_map(), but uses the map expansion pack maps.
        Note that the maps are constantly evolving, whereas we only released a single snapshot of the data.
        Therefore for some scenes there is a bad fit between ego poses and maps.
        :param nusc: The NuScenes instance to load the ego poses from.
        :param sample_token: Sample token.
        :param verbose: Whether to show status messages and progress bar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param patch_margin: Additional margin for map display on top of object locations.
        :param render_car: Whether to render the car.
        :param render_objects: Whether to render objects.
        :param only_visible_objects: Whether to render only visible objects.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param path_nodes: List of points of the path.
        :param path_thickness: Thickness of generated path polygon.
        """
        # Settings
        min_diff_patch = 30

        # Ids of scenes with a bad match between localization and map.
        scene_blacklist = [499, 515, 517]

        # Get scene info for this sample
        sample_record = nusc.get("sample", sample_token)
        scene_record = nusc.get("scene", sample_record["scene_token"])
        log_record = nusc.get("log", scene_record["log_token"])
        log_location = log_record["location"]
        assert (
            self.map_api.map_name == log_location
        ), "Error: NuScenesMap loaded for location %s, should be %s!" % (
            self.map_api.map_name,
            log_location,
        )
        scene_name = scene_record["name"]
        scene_id = int(scene_name.replace("scene-", ""))
        log_record = nusc.get("log", scene_record["log_token"])
        assert (
            log_record["location"] == log_location
        ), "Error: The provided scene_tokens do not correspond to the provided map location!"

        # Print a warning if the localization is known to be bad.
        if verbose and scene_id in scene_blacklist:
            print(
                "Warning: %s is known to have a bad fit between ego pose and map."
                % scene_name
            )

        # Grab the camera info for rendering only visible objects.
        im_size = (1600, 900)
        cam_token = sample_record["data"][camera_channel]
        cam_record = nusc.get("sample_data", cam_token)
        cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        #self.get("sensor", cs_record["sensor_token"])
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])

        # Calculate the pose on the map.
        # Poses are associated with the sample_data. Here we use the lidar sample_data.
        sample_data_record = nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
        pose_record = nusc.get("ego_pose", sample_data_record["ego_pose_token"])
        egopose = pose_record["translation"]

        map_objects = []
        map_objects_bbox = []
        map_objects_color = []
        map_objects_type = []
        objects_frontal_view = []
        if verbose:
            print("Adding ego pose and object poses to map...")

        annotation_tokens = sample_record["anns"]
        for annotation_token in annotation_tokens:
            object_record = nusc.get("sample_annotation", annotation_token)

            # Get 2D box
            bbox = Box(
                center=object_record["translation"],
                size=object_record["size"],
                orientation=Quaternion(object_record["rotation"]),
            )

            if only_visible_objects:
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
                    imsize=im_size,
                    vis_level=BoxVisibility.ANY,
                ):
                    render_object = True
                else:
                    render_object = False
            else:
                render_object = True

            if render_object:
                box_corners = bbox.bottom_corners()[:2, :].transpose()
                box_polygon = Polygon(box_corners)
                x, y = np.array(box_polygon.exterior.xy)

                map_objects.append(object_record["translation"])
                map_objects_bbox.append(
                    np.concatenate((x[:, None], y[:, None]), axis=1)
                )
                map_objects_color.append(
                    self.object_color_map[object_record["category_name"]]
                )
                map_objects_type.append(object_record["category_name"])

        if verbose:
            print("Creating plot...")

        # No Rotation
        egopose = np.array(egopose)[:2]
        map_objects = np.vstack(map_objects)[:, :2]
        map_objects_bbox = np.stack(map_objects_bbox)
        map_objects_color = np.vstack(map_objects_color)[:, 0]
        map_objects_type = np.vstack(map_objects_type)[:, 0]

        # Render the map patch with the current objects.
        min_patch = np.floor(
            np.concatenate((map_objects, egopose[None, :]), axis=0).min(axis=0)
            - patch_margin
        )
        max_patch = np.ceil(
            np.concatenate((map_objects, egopose[None, :]), axis=0).max(axis=0)
            + patch_margin
        )
        diff_patch = max_patch - min_patch
        if any(diff_patch < min_diff_patch):
            center_patch = (min_patch + max_patch) / 2
            diff_patch = np.maximum(diff_patch, min_diff_patch)
            min_patch = center_patch - diff_patch / 2
            max_patch = center_patch + diff_patch / 2
        map_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])

        fig, ax = self.render_map_patch(
            map_patch,
            self.map_api.non_geometric_layers,
            figsize=(10, 10),
            render_egoposes_range=False,
            render_legend=False,
        )

        # Compensate for the margin they used in plotting
        x_margin = np.minimum((map_patch[2] - map_patch[0]) / 4, 50)
        y_margin = np.minimum((map_patch[3] - map_patch[1]) / 4, 10)
        map_patch = (
            map_patch[0] - x_margin,
            map_patch[1] - y_margin,
            map_patch[2] + x_margin,
            map_patch[3] + y_margin,
        )

        # Plot in the same axis as the map.
        # Make sure these are plotted "on top".

        # Render self driving car
        bbox = Box(
            center=pose_record["translation"],
            size=[1.730, 4.084, 1.562],
            orientation=Quaternion(pose_record["rotation"]),
        )
        box_corners = bbox.bottom_corners()[:2, :].transpose()
        box_polygon = Polygon(box_corners)
        x, y = np.array(box_polygon.exterior.xy)
        egobbox = np.concatenate((x[:, None], y[:, None]), axis=1)

        if render_car:
            plt.plot(egobbox[:, 0], egobbox[:, 1], c="#f10c1a", linewidth=1)

        if render_objects:
            for bbox, color in zip(map_objects_bbox, map_objects_color):
                plt.plot(bbox[:, 0], bbox[:, 1], c=color)

        if path_nodes:
            # Create path polygon and draw it on the map
            path_polygon = compute_polygon_from_path(
                path_nodes, thickness=path_thickness
            )
            ax.scatter(
                np.array(path_nodes)[:, 0],
                np.array(path_nodes)[:, 1],
                s=10,
                c="k",
                alpha=1.0,
                zorder=2,
                marker="*",
            )

            ax.add_patch(
                descartes.PolygonPatch(
                    path_polygon,
                    fc="#00ff00",
                    alpha=0.3,
                    label="path",
                )
            )

        plt.axis("off")

        if out_path is not None:
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0)

        egopose = np.array(egopose)
        egobbox = np.array(egobbox)
        ego_translation = np.array(pose_record["translation"])
        ego_rotation = np.array(Quaternion(pose_record["rotation"]).rotation_matrix)
        cam_translation = np.array(cs_record["translation"])
        cam_rotation = np.array(Quaternion(cs_record["rotation"]).rotation_matrix)
        map_patch = np.array(map_patch)
        map_objects = np.array(map_objects)
        map_objects_bbox = np.array(map_objects_bbox)
        map_objects_type = np.array(map_objects_type)

        return (
            egopose,
            egobbox,
            ego_translation,
            ego_rotation,
            cam_translation,
            cam_rotation,
            map_patch,
            map_objects,
            map_objects_bbox,
            map_objects_type,
            cam_intrinsic,
        )

    def generate_sample_data(
        self,
        nusc: NuScenes,
        sample_token: str,
        patch_margin: int = 10,
    ):
        """
        Renders each ego pose and positions of other objects of a list of scenes on the map (around 40 poses per scene).
        This method is heavily inspired by NuScenes.render_egoposes_on_map(), but uses the map expansion pack maps.
        Note that the maps are constantly evolving, whereas we only released a single snapshot of the data.
        Therefore for some scenes there is a bad fit between ego poses and maps.
        :param nusc: The NuScenes instance to load the ego poses from.
        :param sample_token: Sample token.
        :param patch_margin: Additional margin for map display on top of object locations.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        """
        # Settings
        min_diff_patch = 30

        # Ids of scenes with a bad match between localization and map.
        scene_blacklist = [499, 515, 517]

        sample_cam_token = nusc.get("sample", sample_token)["data"]["CAM_FRONT"]

        # Get scene info for this sample
        sample_cam_data_record = nusc.get(
            "sample_data", sample_cam_token
        )  # sample_record["data"]["CAM_FRONT"])
        sample_record = nusc.get("sample", sample_cam_data_record["sample_token"])

        # tmp = sample_record["data"]["CAM_FRONT"]
        # cnt = 0
        # while tmp:
        #     tmp = nusc.get("sample_data", tmp)["prev"]
        #     if tmp == sample_token:
        #         print("found", cnt)
        #     cnt += 1

        scene_record = nusc.get("scene", sample_record["scene_token"])
        log_record = nusc.get("log", scene_record["log_token"])
        log_location = log_record["location"]
        assert (
            self.map_api.map_name == log_location
        ), "Error: NuScenesMap loaded for location %s, should be %s!" % (
            self.map_api.map_name,
            log_location,
        )
        scene_name = scene_record["name"]
        scene_id = int(scene_name.replace("scene-", ""))
        log_record = nusc.get("log", scene_record["log_token"])
        assert (
            log_record["location"] == log_location
        ), "Error: The provided scene_tokens do not correspond to the provided map location!"

        # Print a warning if the localization is known to be bad.
        if scene_id in scene_blacklist:
            print(
                "Warning: %s is known to have a bad fit between ego pose and map."
                % scene_name
            )

        # Grab the camera info for rendering only visible objects.
        im_size = (1600, 900)
        cam_record = nusc.get("sample_data", sample_cam_token)
        cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        # sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        # pose_record = nusc.get('ego_pose', cam_record['ego_pose_token'])

        # cam_token = sample_record["data"]["CAM_FRONT"]
        cam_path = nusc.get_sample_data_path(sample_cam_token)
        # cam_record = nusc.get("sample_data", cam_token)
        # cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        cam_translation = cs_record["translation"]
        cam_rotation = Quaternion(cs_record["rotation"]).rotation_matrix
        # _, boxes, _ = nusc.get_sample_data(sample_token, BoxVisibility.ANY)
        # cam_sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        cam_pose_record = nusc.get("ego_pose", sample_cam_data_record["ego_pose_token"])
        cam_egopose = cam_pose_record["translation"]
        cam_ego_translation = cam_pose_record["translation"]
        cam_ego_rotation = Quaternion(cam_pose_record["rotation"]).rotation_matrix

        # Calculate the pose on the map.
        # Poses are associated with the sample_data. Here we use the lidar sample_data.
        sample_data_record = nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
        lidar_pose_record = nusc.get("ego_pose", sample_data_record["ego_pose_token"])
        lidar_egopose = lidar_pose_record["translation"]
        lidar_ego_translation = lidar_pose_record["translation"]
        lidar_ego_rotation = Quaternion(lidar_pose_record["rotation"]).rotation_matrix

        # Car
        bbox = Box(
            center=cam_pose_record["translation"],
            size=[1.730, 4.084, 1.562],
            orientation=Quaternion(cam_pose_record["rotation"]),
        )
        egobbox = bbox.bottom_corners()[:2, :].transpose().tolist()

        # Objects
        objects_type = []
        objects_token = []
        map_objects_center = []
        map_objects_elevation = []
        map_objects_bbox = []
        image_objects_bbox = []
        annotation_tokens = sample_record["anns"]
        # impath, boxes, camera_intrinsic = nusc.get_sample_data(sample_cam_data_record['token'],
        #                                                            box_vis_level=BoxVisibility.ANY)

        # anns_boxes = nusc.get_boxes(sample_data_token=sample_cam_data_record["token"])
        boxes = nusc.get_boxes(sample_cam_token)

        for annotation_tokens, box in zip(annotation_tokens, boxes):
            object_record = nusc.get("sample_annotation", box.token)

            bbox_image = box.copy()
            bbox_image.translate(-np.array(cam_pose_record["translation"]))
            bbox_image.rotate(Quaternion(cam_pose_record["rotation"]).inverse)

            #  Move box to sensor coord system.
            bbox_image.translate(-np.array(cs_record["translation"]))
            bbox_image.rotate(Quaternion(cs_record["rotation"]).inverse)

            if box_in_image(
                bbox_image,
                intrinsic=cam_intrinsic,
                imsize=im_size,
                vis_level=BoxVisibility.ANY,
            ):
                # map_objects_center.append(object_record["translation"][:2])
                map_objects_center.append(box.center[:2])
                objects_type.append(box.name)
                objects_token.append(box.token)

                map_object_bbox = box.bottom_corners()[:2, :].transpose().tolist()
                image_object_bbox = (
                    view_points(
                        bbox_image.corners(), view=cam_intrinsic, normalize=True
                    )[:2, :]
                    .transpose()
                    .tolist()
                )
                map_objects_bbox.append(map_object_bbox)
                # map_objects_elevation.append(box.bottom_corners()[2, 0])
                map_objects_elevation.append(box.bottom_corners()[2, 0])
                image_objects_bbox.append(image_object_bbox)

        # for annotation_token in annotation_tokens:
        #     object_record = nusc.get("sample_annotation", annotation_token)
        #
        #     bbox = Box(
        #         center=object_record["translation"],
        #         size=object_record["size"],
        #         orientation=Quaternion(object_record["rotation"]),
        #     )
        #
        #     bbox_image = bbox.copy()
        #     # Move box to ego vehicle coord system.
        #     bbox_image.translate(-np.array(pose_record["translation"]))
        #     bbox_image.rotate(Quaternion(pose_record["rotation"]).inverse)
        #
        #     #  Move box to sensor coord system.
        #     bbox_image.translate(-np.array(cs_record["translation"]))
        #     bbox_image.rotate(Quaternion(cs_record["rotation"]).inverse)
        #     if annotation_token == box_token:
        #         print("here")
        #
        #     if box_in_image(
        #         bbox_image,
        #         intrinsic=cam_intrinsic,
        #         imsize=im_size,
        #         vis_level=BoxVisibility.ANY,
        #     ):
        #
        #         # Just save heights, but not the height of the object center, but rather it's floor
        #         map_objects_center.append(object_record["translation"][:2])
        #         objects_type.append(object_record["category_name"])
        #         objects_token.append(object_record["token"])
        #         map_object_bbox = bbox.bottom_corners()[:2, :].transpose().tolist()
        #         image_object_bbox = (
        #             view_points(
        #                 bbox_image.corners(), view=cam_intrinsic, normalize=True
        #             )[:2, :]
        #             .transpose()
        #             .tolist()
        #         )
        #         map_objects_bbox.append(map_object_bbox)
        #         map_objects_elevation.append(bbox.bottom_corners()[2, 0])
        #         image_objects_bbox.append(image_object_bbox)

        # Render the map patch with the current objects.
        egopose = np.array(cam_egopose)[:2]
        # egopose = np.array(lidar_egopose)[:2]
        if len(map_objects_center):
            map_objects_center = np.vstack(map_objects_center)[:, :2]
        else:
            map_objects_center = np.zeros((0, 2))

        min_patch = np.floor(
            np.concatenate((map_objects_center, egopose[None, :]), axis=0).min(axis=0)
            - patch_margin
        )
        max_patch = np.ceil(
            np.concatenate((map_objects_center, egopose[None, :]), axis=0).max(axis=0)
            + patch_margin
        )
        diff_patch = max_patch - min_patch
        if any(diff_patch < min_diff_patch):
            center_patch = (min_patch + max_patch) / 2
            diff_patch = np.maximum(diff_patch, min_diff_patch)
            min_patch = center_patch - diff_patch / 2
            max_patch = center_patch + diff_patch / 2
        map_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])

        # Compensate for the margin they used in plotting
        x_margin = np.minimum((map_patch[2] - map_patch[0]) / 4, 50)
        y_margin = np.minimum((map_patch[3] - map_patch[1]) / 4, 10)
        map_patch_margin = (
            map_patch[0] - x_margin,
            map_patch[1] - y_margin,
            map_patch[2] + x_margin,
            map_patch[3] + y_margin,
        )

        # Return to list
        map_objects_center = map_objects_center.tolist()
        cam_intrinsic = cam_intrinsic.tolist()
        ego_rotation = cam_ego_rotation.tolist()
        # ego_rotation = lidar_ego_rotation.tolist()
        cam_rotation = cam_rotation.tolist()

        return (
            cam_path,
            map_patch,
            map_patch_margin,
            egobbox,
            map_objects_center,
            map_objects_elevation,
            map_objects_bbox,
            image_objects_bbox,
            objects_token,
            objects_type,
            cam_intrinsic,
            cam_ego_translation,
            # lidar_ego_translation,
            ego_rotation,
            cam_translation,
            cam_rotation,
        )

        # return (
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
        # )

    # Works fairly ok
    def generate_sample_data_lidar_top_down_camera_frontal(
        self,
        nusc: NuScenes,
        sample_token: str,
        box_token: str,
        patch_margin: int = 10,
    ):
        """
        Renders each ego pose and positions of other objects of a list of scenes on the map (around 40 poses per scene).
        This method is heavily inspired by NuScenes.render_egoposes_on_map(), but uses the map expansion pack maps.
        Note that the maps are constantly evolving, whereas we only released a single snapshot of the data.
        Therefore for some scenes there is a bad fit between ego poses and maps.
        :param nusc: The NuScenes instance to load the ego poses from.
        :param sample_token: Sample token.
        :param patch_margin: Additional margin for map display on top of object locations.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        """
        # Settings
        min_diff_patch = 30

        # Ids of scenes with a bad match between localization and map.
        scene_blacklist = [499, 515, 517]

        # Get scene info for this sample
        sample_cam_data_record = nusc.get(
            "sample_data", sample_token
        )  # sample_record["data"]["CAM_FRONT"])
        sample_record = nusc.get("sample", sample_cam_data_record["sample_token"])

        # tmp = sample_record["data"]["CAM_FRONT"]
        # cnt = 0
        # while tmp:
        #     tmp = nusc.get("sample_data", tmp)["prev"]
        #     if tmp == sample_token:
        #         print("found", cnt)
        #     cnt += 1

        scene_record = nusc.get("scene", sample_record["scene_token"])
        log_record = nusc.get("log", scene_record["log_token"])
        log_location = log_record["location"]
        assert (
            self.map_api.map_name == log_location
        ), "Error: NuScenesMap loaded for location %s, should be %s!" % (
            self.map_api.map_name,
            log_location,
        )
        scene_name = scene_record["name"]
        scene_id = int(scene_name.replace("scene-", ""))
        log_record = nusc.get("log", scene_record["log_token"])
        assert (
            log_record["location"] == log_location
        ), "Error: The provided scene_tokens do not correspond to the provided map location!"

        # Print a warning if the localization is known to be bad.
        if scene_id in scene_blacklist:
            print(
                "Warning: %s is known to have a bad fit between ego pose and map."
                % scene_name
            )

        # Grab the camera info for rendering only visible objects.
        im_size = (1600, 900)
        cam_record = nusc.get("sample_data", sample_token)
        cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        # sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        # pose_record = nusc.get('ego_pose', cam_record['ego_pose_token'])

        # cam_token = sample_record["data"]["CAM_FRONT"]
        cam_path = nusc.get_sample_data_path(sample_token)
        # cam_record = nusc.get("sample_data", cam_token)
        # cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        cam_translation = cs_record["translation"]
        cam_rotation = Quaternion(cs_record["rotation"]).rotation_matrix
        # _, boxes, _ = nusc.get_sample_data(sample_token, BoxVisibility.ANY)
        # cam_sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        cam_pose_record = nusc.get("ego_pose", sample_cam_data_record["ego_pose_token"])
        cam_egopose = cam_pose_record["translation"]
        cam_ego_translation = cam_pose_record["translation"]
        cam_ego_rotation = Quaternion(cam_pose_record["rotation"]).rotation_matrix

        # Calculate the pose on the map.
        # Poses are associated with the sample_data. Here we use the lidar sample_data.
        sample_data_record = nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
        lidar_pose_record = nusc.get("ego_pose", sample_data_record["ego_pose_token"])
        lidar_egopose = lidar_pose_record["translation"]
        lidar_ego_translation = lidar_pose_record["translation"]
        lidar_ego_rotation = Quaternion(lidar_pose_record["rotation"]).rotation_matrix

        # Car
        bbox = Box(
            # center=cam_pose_record["translation"],
            center=lidar_pose_record["translation"],
            size=[1.730, 4.084, 1.562],
            # orientation=Quaternion(cam_pose_record["rotation"]),
            orientation=Quaternion(lidar_pose_record["rotation"]),
        )
        egobbox = bbox.bottom_corners()[:2, :].transpose().tolist()

        # Objects
        objects_type = []
        objects_token = []
        map_objects_center = []
        map_objects_elevation = []
        map_objects_bbox = []
        image_objects_bbox = []
        annotation_tokens = sample_record["anns"]
        # impath, boxes, camera_intrinsic = nusc.get_sample_data(sample_cam_data_record['token'],
        #                                                            box_vis_level=BoxVisibility.ANY)

        # anns_boxes = nusc.get_boxes(sample_data_token=sample_cam_data_record["token"])
        boxes = nusc.get_boxes(sample_token)

        for annotation_tokens, box in zip(annotation_tokens, boxes):
            object_record = nusc.get("sample_annotation", box.token)

            bbox_image = box.copy()
            bbox_image.translate(-np.array(cam_pose_record["translation"]))
            bbox_image.rotate(Quaternion(cam_pose_record["rotation"]).inverse)

            #  Move box to sensor coord system.
            bbox_image.translate(-np.array(cs_record["translation"]))
            bbox_image.rotate(Quaternion(cs_record["rotation"]).inverse)

            if box_in_image(
                bbox_image,
                intrinsic=cam_intrinsic,
                imsize=im_size,
                vis_level=BoxVisibility.ANY,
            ):
                map_objects_center.append(object_record["translation"][:2])
                # map_objects_center.append(box.center[:2])
                objects_type.append(box.name)
                objects_token.append(box.token)

                top_down_obj_bbox = Box(
                    center=object_record["translation"],
                    size=object_record["size"],
                    orientation=Quaternion(object_record["rotation"]),
                )

                map_object_bbox = (
                    top_down_obj_bbox.bottom_corners()[:2, :].transpose().tolist()
                )
                image_object_bbox = (
                    view_points(
                        bbox_image.corners(), view=cam_intrinsic, normalize=True
                    )[:2, :]
                    .transpose()
                    .tolist()
                )
                map_objects_bbox.append(map_object_bbox)
                # map_objects_elevation.append(box.bottom_corners()[2, 0])
                map_objects_elevation.append(top_down_obj_bbox.bottom_corners()[2, 0])
                image_objects_bbox.append(image_object_bbox)

        # for annotation_token in annotation_tokens:
        #     object_record = nusc.get("sample_annotation", annotation_token)
        #
        #     bbox = Box(
        #         center=object_record["translation"],
        #         size=object_record["size"],
        #         orientation=Quaternion(object_record["rotation"]),
        #     )
        #
        #     bbox_image = bbox.copy()
        #     # Move box to ego vehicle coord system.
        #     bbox_image.translate(-np.array(pose_record["translation"]))
        #     bbox_image.rotate(Quaternion(pose_record["rotation"]).inverse)
        #
        #     #  Move box to sensor coord system.
        #     bbox_image.translate(-np.array(cs_record["translation"]))
        #     bbox_image.rotate(Quaternion(cs_record["rotation"]).inverse)
        #     if annotation_token == box_token:
        #         print("here")
        #
        #     if box_in_image(
        #         bbox_image,
        #         intrinsic=cam_intrinsic,
        #         imsize=im_size,
        #         vis_level=BoxVisibility.ANY,
        #     ):
        #
        #         # Just save heights, but not the height of the object center, but rather it's floor
        #         map_objects_center.append(object_record["translation"][:2])
        #         objects_type.append(object_record["category_name"])
        #         objects_token.append(object_record["token"])
        #         map_object_bbox = bbox.bottom_corners()[:2, :].transpose().tolist()
        #         image_object_bbox = (
        #             view_points(
        #                 bbox_image.corners(), view=cam_intrinsic, normalize=True
        #             )[:2, :]
        #             .transpose()
        #             .tolist()
        #         )
        #         map_objects_bbox.append(map_object_bbox)
        #         map_objects_elevation.append(bbox.bottom_corners()[2, 0])
        #         image_objects_bbox.append(image_object_bbox)

        # Render the map patch with the current objects.
        egopose = np.array(cam_egopose)[:2]
        # egopose = np.array(lidar_egopose)[:2]
        if len(map_objects_center):
            map_objects_center = np.vstack(map_objects_center)[:, :2]
        else:
            map_objects_center = np.zeros((0, 2))

        min_patch = np.floor(
            np.concatenate((map_objects_center, egopose[None, :]), axis=0).min(axis=0)
            - patch_margin
        )
        max_patch = np.ceil(
            np.concatenate((map_objects_center, egopose[None, :]), axis=0).max(axis=0)
            + patch_margin
        )
        diff_patch = max_patch - min_patch
        if any(diff_patch < min_diff_patch):
            center_patch = (min_patch + max_patch) / 2
            diff_patch = np.maximum(diff_patch, min_diff_patch)
            min_patch = center_patch - diff_patch / 2
            max_patch = center_patch + diff_patch / 2
        map_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])

        # Compensate for the margin they used in plotting
        x_margin = np.minimum((map_patch[2] - map_patch[0]) / 4, 50)
        y_margin = np.minimum((map_patch[3] - map_patch[1]) / 4, 10)
        map_patch_margin = (
            map_patch[0] - x_margin,
            map_patch[1] - y_margin,
            map_patch[2] + x_margin,
            map_patch[3] + y_margin,
        )

        # Return to list
        map_objects_center = map_objects_center.tolist()
        cam_intrinsic = cam_intrinsic.tolist()
        ego_rotation = cam_ego_rotation.tolist()
        # ego_rotation = lidar_ego_rotation.tolist()
        cam_rotation = cam_rotation.tolist()

        return (
            cam_path,
            map_patch,
            map_patch_margin,
            egobbox,
            map_objects_center,
            map_objects_elevation,
            map_objects_bbox,
            image_objects_bbox,
            objects_token,
            objects_type,
            cam_intrinsic,
            cam_ego_translation,
            # lidar_ego_translation,
            ego_rotation,
            cam_translation,
            cam_rotation,
        )

        # return (
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
        # )

    def generate_sample_data_old(
        self,
        nusc: NuScenes,
        sample_token: str,
        box_token: str,
        patch_margin: int = 10,
    ):
        """
        Renders each ego pose and positions of other objects of a list of scenes on the map (around 40 poses per scene).
        This method is heavily inspired by NuScenes.render_egoposes_on_map(), but uses the map expansion pack maps.
        Note that the maps are constantly evolving, whereas we only released a single snapshot of the data.
        Therefore for some scenes there is a bad fit between ego poses and maps.
        :param nusc: The NuScenes instance to load the ego poses from.
        :param sample_token: Sample token.
        :param patch_margin: Additional margin for map display on top of object locations.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        """
        # Settings
        min_diff_patch = 30

        # Ids of scenes with a bad match between localization and map.
        scene_blacklist = [499, 515, 517]

        # Get scene info for this sample
        sample_cam_data_record = nusc.get(
            "sample_data", sample_token
        )  # sample_record["data"]["CAM_FRONT"])
        sample_record = nusc.get("sample", sample_cam_data_record["sample_token"])

        scene_record = nusc.get("scene", sample_record["scene_token"])
        log_record = nusc.get("log", scene_record["log_token"])
        log_location = log_record["location"]
        assert (
            self.map_api.map_name == log_location
        ), "Error: NuScenesMap loaded for location %s, should be %s!" % (
            self.map_api.map_name,
            log_location,
        )
        scene_name = scene_record["name"]
        scene_id = int(scene_name.replace("scene-", ""))
        log_record = nusc.get("log", scene_record["log_token"])
        assert (
            log_record["location"] == log_location
        ), "Error: The provided scene_tokens do not correspond to the provided map location!"

        # Print a warning if the localization is known to be bad.
        if scene_id in scene_blacklist:
            print(
                "Warning: %s is known to have a bad fit between ego pose and map."
                % scene_name
            )

        # Grab the camera info for rendering only visible objects.
        im_size = (1600, 900)
        cam_record = nusc.get("sample_data", sample_token)
        cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        # sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        # pose_record = nusc.get('ego_pose', cam_record['ego_pose_token'])

        # cam_token = sample_record["data"]["CAM_FRONT"]
        cam_path = nusc.get_sample_data_path(sample_token)
        # cam_record = nusc.get("sample_data", cam_token)
        # cs_record = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        cam_translation = cs_record["translation"]
        cam_rotation = Quaternion(cs_record["rotation"]).rotation_matrix
        _, boxes, _ = nusc.get_sample_data(sample_token, BoxVisibility.ANY)

        # Calculate the pose on the map.
        # Poses are associated with the sample_data. Here we use the lidar sample_data.
        sample_data_record = nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
        pose_record = nusc.get("ego_pose", sample_data_record["ego_pose_token"])
        egopose = pose_record["translation"]
        ego_translation = pose_record["translation"]
        ego_rotation = Quaternion(pose_record["rotation"]).rotation_matrix

        # Car
        bbox = Box(
            center=pose_record["translation"],
            size=[1.730, 4.084, 1.562],
            orientation=Quaternion(pose_record["rotation"]),
        )
        egobbox = bbox.bottom_corners()[:2, :].transpose().tolist()

        # Objects
        objects_type = []
        objects_token = []
        map_objects_center = []
        map_objects_elevation = []
        map_objects_bbox = []
        image_objects_bbox = []
        annotation_tokens = sample_record["anns"]
        # impath, boxes, camera_intrinsic = nusc.get_sample_data(sample_cam_data_record['token'],
        #                                                            box_vis_level=BoxVisibility.ANY)

        # anns_boxes = nusc.get_boxes(sample_data_token=sample_cam_data_record["token"])

        for box in boxes:
            object_record = nusc.get("sample_annotation", box.token)

            # Move box to ego vehicle coord system
            # box.translate(-np.array(pose_record['translation']))
            # box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system
            # box.translate(-np.array(cs_record['translation']))
            # box.rotate(Quaternion(cs_record['rotation']).inverse)
            # if box.token == box_token:
            #     print("mhh")
            # if box_in_image(
            #     box,
            #     intrinsic=cam_intrinsic,
            #     imsize=im_size,
            #     vis_level=BoxVisibility.ANY,
            # ):
            map_objects_center.append(object_record["translation"][:2])
            objects_type.append(object_record["category_name"])
            objects_token.append(object_record["token"])
            map_object_bbox = box.bottom_corners()[:2, :].transpose().tolist()
            image_object_bbox = (
                view_points(box.corners(), view=cam_intrinsic, normalize=True)[:2, :]
                .transpose()
                .tolist()
            )
            map_objects_bbox.append(map_object_bbox)
            map_objects_elevation.append(box.bottom_corners()[2, 0])
            image_objects_bbox.append(image_object_bbox)

        # for annotation_token in annotation_tokens:
        #     object_record = nusc.get("sample_annotation", annotation_token)
        #
        #     bbox = Box(
        #         center=object_record["translation"],
        #         size=object_record["size"],
        #         orientation=Quaternion(object_record["rotation"]),
        #     )
        #
        #     bbox_image = bbox.copy()
        #     # Move box to ego vehicle coord system.
        #     bbox_image.translate(-np.array(pose_record["translation"]))
        #     bbox_image.rotate(Quaternion(pose_record["rotation"]).inverse)
        #
        #     #  Move box to sensor coord system.
        #     bbox_image.translate(-np.array(cs_record["translation"]))
        #     bbox_image.rotate(Quaternion(cs_record["rotation"]).inverse)
        #     if annotation_token == box_token:
        #         print("here")
        #
        #     if box_in_image(
        #         bbox_image,
        #         intrinsic=cam_intrinsic,
        #         imsize=im_size,
        #         vis_level=BoxVisibility.ANY,
        #     ):
        #
        #         # Just save heights, but not the height of the object center, but rather it's floor
        #         map_objects_center.append(object_record["translation"][:2])
        #         objects_type.append(object_record["category_name"])
        #         objects_token.append(object_record["token"])
        #         map_object_bbox = bbox.bottom_corners()[:2, :].transpose().tolist()
        #         image_object_bbox = (
        #             view_points(
        #                 bbox_image.corners(), view=cam_intrinsic, normalize=True
        #             )[:2, :]
        #             .transpose()
        #             .tolist()
        #         )
        #         map_objects_bbox.append(map_object_bbox)
        #         map_objects_elevation.append(bbox.bottom_corners()[2, 0])
        #         image_objects_bbox.append(image_object_bbox)

        # Render the map patch with the current objects.
        egopose = np.array(egopose)[:2]
        if len(map_objects_center):
            map_objects_center = np.vstack(map_objects_center)[:, :2]
        else:
            map_objects_center = np.zeros((0, 2))

        min_patch = np.floor(
            np.concatenate((map_objects_center, egopose[None, :]), axis=0).min(axis=0)
            - patch_margin
        )
        max_patch = np.ceil(
            np.concatenate((map_objects_center, egopose[None, :]), axis=0).max(axis=0)
            + patch_margin
        )
        diff_patch = max_patch - min_patch
        if any(diff_patch < min_diff_patch):
            center_patch = (min_patch + max_patch) / 2
            diff_patch = np.maximum(diff_patch, min_diff_patch)
            min_patch = center_patch - diff_patch / 2
            max_patch = center_patch + diff_patch / 2
        map_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])

        # Compensate for the margin they used in plotting
        x_margin = np.minimum((map_patch[2] - map_patch[0]) / 4, 50)
        y_margin = np.minimum((map_patch[3] - map_patch[1]) / 4, 10)
        map_patch_margin = (
            map_patch[0] - x_margin,
            map_patch[1] - y_margin,
            map_patch[2] + x_margin,
            map_patch[3] + y_margin,
        )

        # Return to list
        map_objects_center = map_objects_center.tolist()
        cam_intrinsic = cam_intrinsic.tolist()
        ego_rotation = ego_rotation.tolist()
        cam_rotation = cam_rotation.tolist()

        return (
            cam_path,
            map_patch,
            map_patch_margin,
            egobbox,
            map_objects_center,
            map_objects_elevation,
            map_objects_bbox,
            image_objects_bbox,
            objects_token,
            objects_type,
            cam_intrinsic,
            ego_translation,
            ego_rotation,
            cam_translation,
            cam_rotation,
        )
        
        
    def _render_layer(self, ax: Axes, layer_name: str, alpha: float, tokens: List[str] = None, transform: transforms.Transform = None) -> None:
        """
        Wrapper method that renders individual layers on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name in self.map_api.non_geometric_polygon_layers:
            self._render_polygon_layer(ax, layer_name, alpha, tokens, transform)
        elif layer_name in self.map_api.non_geometric_line_layers:
            self._render_line_layer(ax, layer_name, alpha, tokens, transform)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _render_polygon_layer(self, ax: Axes, layer_name: str, alpha: float, tokens: List[str] = None, transform: transforms.Transform = None) -> None:
        """
        Renders an individual non-geometric polygon layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name not in self.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        first_time = True
        records = getattr(self.map_api, layer_name)
        if tokens is not None:
            records = [r for r in records if r['token'] in tokens]
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    if first_time:
                        label = layer_name
                        first_time = False
                    else:
                        label = None
                    patch = descartes.PolygonPatch(polygon, fc=self.color_map[layer_name], alpha=alpha, label=label)
                    if transform:
                        patch.set_transform(transform)
                    ax.add_patch(patch)
        else:
            for record in records:
                polygon = self.map_api.extract_polygon(record['polygon_token'])

                if first_time:
                    label = layer_name
                    first_time = False
                else:
                    label = None
                patch = descartes.PolygonPatch(polygon, fc=self.color_map[layer_name], alpha=alpha, label=label)
                if transform:
                    patch.set_transform(transform)
                ax.add_patch(patch)

    def _render_line_layer(self, ax: Axes, layer_name: str, alpha: float, tokens: List[str] = None, transform: transforms.Transform = None) -> None:
        """
        Renders an individual non-geometric line layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name not in self.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        first_time = True
        records = getattr(self.map_api, layer_name)
        if tokens is not None:
            records = [r for r in records if r['token'] in tokens]
        for record in records:
            if first_time:
                label = layer_name
                first_time = False
            else:
                label = None
            line = self.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            if layer_name == 'traffic_light':
                # Draws an arrow with the physical traffic light as the starting point, pointing to the direction on
                # where the traffic light points.
                patch = Arrow(xs[0], ys[0], xs[1]-xs[0], ys[1]-ys[0], color=self.color_map[layer_name], label=label)
                if transform:
                    patch.set_transform(transform)
                ax.add_patch(patch)
            else:
                if transform:
                    ax.plot(xs, ys, color=self.color_map[layer_name], alpha=alpha, label=label, transform=transform)
                else:
                    ax.plot(xs, ys, color=self.color_map[layer_name], alpha=alpha, label=label)

    def render_map_patch_custom(
        self,
        nusc: NuScenes,
        sample_data_token: str,
        limit_left: float = 10,
        limit_right: float = 70,
        limit_top: float = 40,
        limit_bottom: float = 40,
        layer_names: List[str] = None,
        alpha: float = 0.5,
        figsize: Tuple[float, float] = (15, 15),
    ):
        """
        Renders a rectangular patch specified by `box_coords`. By default renders all layers.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: All the non geometric layers that we want to render.
        :param alpha: The opacity of each layer.
        :param figsize: Size of the whole figure.
        :param bitmap: Optional BitMap object to render below the other map layers.
        :return: The matplotlib figure and axes of the rendered layers.
        """

        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        fig = plt.figure(figsize=figsize)

        #local_aspect_ratio = (limit_left + limit_right) / (limit_top + limit_bottom)

        ax = fig.add_axes([0, 0, 1, 1])# / local_aspect_ratio])
        ax.grid(False)
        
        sd_record = nusc.get("sample_data", sample_data_token)
        pose = nusc.get("ego_pose", sd_record["ego_pose_token"])
        pixel_coords = (pose["translation"][0], pose["translation"][1])
        ypr_rad = Quaternion(pose["rotation"]).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])
        
        print(f"Yaw degrees: {yaw_deg}")
        
        tr = transforms.Affine2D().rotate_deg_around(
            pixel_coords[0], pixel_coords[1], yaw_deg
        )
        tr = tr + ax.transData

        for layer_name in layer_names:
            self._render_layer(ax, layer_name, alpha, transform=tr)

        # plt.plot()
        # plt.savefig("kobaja.png")
        # exit()
        ax.set_xlim(pixel_coords[0] - limit_left, pixel_coords[0] + limit_right)
        ax.set_ylim(pixel_coords[1] - limit_bottom, pixel_coords[1] + limit_top)

        return fig, ax