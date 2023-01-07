import sys
sys.path.append("/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/det3d/datasets/nuscenes")
from talk2car import Talk2Car
import json

t2c = Talk2Car(version="train", dataroot="/cw/liir/NoCsBack/testliir/thierry/PathProjection/3d_object_detection/CenterPoint/data/nuscenes")

day_night = {}
for cmd in t2c.commands:
    sample_token = t2c.get("sample_data", cmd.frame_token)["sample_token"]
    scene = t2c.get("sample", sample_token)["scene_token"]
    day_night[cmd.command_token] = "night" if 'night' in t2c.get("scene",scene)['description'].lower() else "day"

json.dump(day_night, open("t2c_day_night_train.json","w"))