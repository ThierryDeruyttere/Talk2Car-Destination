import json
import os

fcos_root = "/cw/liir_code/NoCsBack/thierry/PathProjection/data_root/fcos3d_extracted"
filter = {}

for split in ["train", "val", "test"]:
    with open(os.path.join(fcos_root, 'fcos3d_t2c_{}.json'.format(split)), 'r') as f:
        data = json.load(f)
        correct = []
        for x in data:
            if x["command_token"] not in filter:
                correct.append(x)
                filter[x["command_token"]] = True
    print(split, len(correct))
    json.dump(correct, open(os.path.join(fcos_root, 'filtered_fcos3d_t2c_{}.json'.format(split)), "w"))
