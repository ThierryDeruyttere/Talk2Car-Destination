import json
import os.path as osp

save_dir = '/home2/NoCsBack/hci/dusan/Results/fcos3d_extracted'
comm_list = json.load(open(osp.join(save_dir, "commands_with_missing_top_down_data.json"), "r"))
print(len(comm_list))
print(len(list(set(comm_list))))