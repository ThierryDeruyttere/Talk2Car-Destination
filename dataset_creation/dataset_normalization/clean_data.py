""""
This script prepares the talk2car data into train/val/test jsons with some options

::options

    - only reviewed samples
    - only legal samples
    - only highest ranking sample
"""
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--only_reviewed", action="store_true", default=False, required=False)
parser.add_argument("--only_legal", action="store_true", default=False, required=False)
parser.add_argument("--only_highest_ranking", action="store_true", default=False, required=False)
parser.add_argument("--discard_unusable", action="store_true", default=False, required=False)

DATA_ROOT = "rotated_data.json"

def split_data(args):
    all_data = json.load(open(DATA_ROOT, "r"))

    out = {"train": {}, "val": {}, "test": {}}
    counter = 0
    unusable = 0
    for command in all_data:
        if args.only_reviewed:
            if "rating" not in command: continue
        if args.discard_unusable and command["unusable"]:
            unusable += 1
            continue

        if args.only_legal:
            if command["legality"]["legalityIllegal"]: continue

        selected = out[command["split"]]
        token = command["command_token"]
        if token not in selected:
            selected[token] = []

        selected[token].append(command)
        counter += 1
    print("unusable", unusable)
    if args.only_highest_ranking:
        new_out = {}
        for split, data in out.items():
            new_out[split] = {}
            for token, annotations in data.items():
                sorted_annos = sorted(annotations,
                                      key=lambda x: x["rating"] if "rating" in x else -1,
                                      reverse=True)
                if "rating" in sorted_annos[0] and sorted_annos[0]["rating"] > -1:
                    new_out[split][token] = [sorted_annos[0]]
        out = new_out

    return counter, out


if __name__ == "__main__":
    args = parser.parse_args()
    args.discard_unusable = True
    #args.only_legal = True
    all_data, out = split_data(args)
    # args.only_reviewed = True
    # only_reviewed, b = split_data(args)
    # args.only_reviewed = False
    #
    # args.only_legal = True
    # only_legal, c = split_data(args)
    # args.only_legal = False
    #
    # args.only_highest_ranking = True
    # only_highest_ranking, d = split_data(args)
    # args.only_highest_ranking = False

    for split, data in out.items():
        with open("{}.json".format(split), "w") as f:
            json.dump(data, f)
