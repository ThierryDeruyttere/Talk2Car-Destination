from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .talk2car import Talk2CarDataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "T2C": Talk2CarDataset
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
