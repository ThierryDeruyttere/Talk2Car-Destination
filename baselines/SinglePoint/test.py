import argparse
import json
import os
import pathlib

import torch
from torch.utils.data import DataLoader

from model import SinglePoint
from talk2car import Talk2Car_Detector
from utils import draw_hyps_t2c
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Talk2Car", required=False)
parser.add_argument(
    "--data_dir",
    required=False,
    default="/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"
)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument("--seed", default=42, required=False)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Path to the checkpoint to potentially continue training from"
)
parser.add_argument(
    "--draw",
    action="store_true",
    help="Whether to draw the hypotheses and the heatmaps.",
)
parser.add_argument(
    "--num_heatmaps_drawn", type=int, default=5, help="Number of drawn images.",
)
parser.add_argument(
    "--thresholds",
    nargs="*",
    type=float,
    default=[2.0, 4.0],
    help="Thresholds for distance (in meters) below which the prediction is considered to be correct.",
)

# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
# parser = SinglePoint.add_model_specific_args(parser)
args = parser.parse_args()
torch.manual_seed(args.seed)

@torch.no_grad()
def main(args):
    device = torch.device(
        'cuda', index=args.gpu_index
    ) if torch.cuda.is_available() else torch.device('cpu')

    checkpoint_path = args.checkpoint_path
    print(f"Checkpoint Path: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_serialize = checkpoint["state_dict"]
    save_path = os.path.join(checkpoint_path[:-5], "results")
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    hparams = checkpoint["hyper_parameters"]
    model = SinglePoint(hparams)
    model = model.to(device)
    model.load_state_dict(model_serialize)
    model.eval()

    data_test = Talk2Car_Detector(
        split="test",
        dataset_root=args.data_dir,
        height=hparams["height"],
        width=hparams["width"],
        unrolled=hparams["unrolled"],
        use_ref_obj=hparams["use_ref_obj"]
    )

    criterion = torch.nn.MSELoss(reduction="none")

    pa_sums = [0.0 for _ in range(len(args.thresholds))]
    ade_sum = 0.0

    all_pases = [[] for _ in range(len(args.thresholds))]
    all_ades = []

    counter = 0
    to_meters = torch.tensor([120.0, 80.0]).to(device)

    results = []
    test_loader = DataLoader(
        dataset=data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    for bidx, data in enumerate(tqdm(test_loader)):

        inp, gt_coord, command = data["x"], data["y"], data["command_embedding"]
        inp = inp.float().to(device)
        gt_coord = gt_coord.float().to(device)
        [B, N, _] = gt_coord.shape
        gt_coord = gt_coord.view(-1, 2)
        command = command.to(device)

        pred_coord = model.forward(inp, command)

        loss = criterion(
            pred_coord.repeat_interleave(N, dim=0),
            gt_coord
        ).mean()

        gt_coord = gt_coord.view(B, N, -1)
        pred_coord = pred_coord.view(B, 1, -1)

        distances = (
                (gt_coord.unsqueeze(1) - pred_coord.unsqueeze(2))
                * to_meters.to(pred_coord)
        ).norm(2, dim=-1).min(dim=-1)[0]

        pas = [0.0 for _ in range(len(args.thresholds))]
        for i, threshold in enumerate(args.thresholds):
            corrects = distances < threshold
            p = corrects.sum(dim=-1) / corrects.shape[1]
            pas[i] = p.item()
            all_pases[i].extend(p.cpu().tolist())
            pa_sums[i] += pas[i]

        ade = distances.mean(dim=-1)
        all_ades.extend(ade.cpu().tolist())
        ade = ade.mean()
        ade = ade.item()
        ade_sum += ade

        result_row = {
            "bidx": bidx,
            "pa": pas,
            "ade": float(ade),
        }
        results.append(result_row)

        pred_coord = pred_coord.view(B, -1).cpu().numpy()
        gt_coord = gt_coord.view(B * N, -1).cpu().numpy()

        if args.draw and counter < args.num_heatmaps_drawn:
            # Draw some predictions
            img_path, frontal_img_path, ego_car, ref_obj, _, command_text = data_test.get_obj_info(
                bidx
            )
            drawn_img_hyps = draw_hyps_t2c(
                img_path, pred_coord, ego_car, ref_obj, gt_coord, normalize=True
            )
            drawn_img_hyps.save(
                os.path.join(save_path, "test" + "-" + str(bidx) + "-hyps.png")
            )
            with open(
                os.path.join(save_path, "test" + "-" + str(bidx) + "-command.txt"), "w"
            ) as f:
                f.write(command_text)
        else:
            break

        counter = counter + 1

    all_ades = torch.tensor(all_ades)
    all_ades_eb = torch.std(all_ades, dim=0).item() / (counter ** 0.5)
    all_pases = torch.tensor(all_pases)
    all_pases_eb = (torch.std(all_pases, dim=1) / (counter ** 0.5)).tolist()

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(args)
