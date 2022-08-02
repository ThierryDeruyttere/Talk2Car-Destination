import argparse
import json
import os
import pathlib

from utils import mmfp_utils
import torch
from tqdm import tqdm

from dataset.talk2car import Talk2Car, Talk2Car_Detector
from utils.utils import draw_hyps_t2c

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Talk2Car", required=False)
parser.add_argument(
    "--data_dir",
    required=False,
    default="/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"
)
parser.add_argument(
    "--detector_data_dir",
    required=False,
    default="/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root/fcos3d_extracted"
)
parser.add_argument(
    "--save_path",
    required=True,
    help="Save directory."
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
parser.add_argument("--seed", default=42, required=False)
parser.add_argument("--width", type=int, default=300, help="Image width.")
parser.add_argument("--height", type=int, default=200, help="Image height")
parser.add_argument('--gpu_index', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device(
		'cuda', index=args.gpu_index
) if torch.cuda.is_available() else torch.device('cpu')

@torch.no_grad()
def main(args):
    save_path = args.save_path
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    if args.dataset == "Talk2Car_Detector":
        data_test = Talk2Car_Detector(
            split="test",
            dataset_root=args.data_dir,
            height=args.height,
            width=args.width,
            unrolled=False,
        )
    else:
        data_test = Talk2Car(
            split="test",
            root=args.data_dir,
            height=args.height,
            width=args.width,
            unrolled=False,
        )

    pa_sums = [0.0 for _ in range(len(args.thresholds))]
    ade_sum = 0.0
    demd_sum = 0.0

    all_demds = []
    all_pases = [[] for _ in range(len(args.thresholds))]
    all_ades = []

    counter = 0
    to_meters = torch.tensor([120.0, 80.0]).to(device)

    results = []
    test_loader = torch.utils.data.DataLoader(
        dataset=data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    for bidx, data in enumerate(tqdm(test_loader)):
        inp, gt_coord, command = data["x"], data["y"], data["command_embedding"]
        inp = inp.float().to(device)
        gt_coord = gt_coord.float().to(device)
        command = command.to(device)

        pred_coord = torch.rand((1, 1000, 2)).to(gt_coord)

        distances = (
            (gt_coord.unsqueeze(1) - pred_coord.unsqueeze(2))
            * to_meters
        ).norm(2, dim=-1).min(dim=-1)[0]

        pas = [0.0 for _ in range(len(args.thresholds))]
        for i, threshold in enumerate(args.thresholds):
            corrects = distances < threshold
            p = corrects.sum(dim=-1).item() / corrects.shape[-1]
            pas[i] = p
            all_pases[i].extend([p])
            pa_sums[i] += p

        ade = distances.mean()
        ade = ade.item()
        all_ades.extend([ade])
        ade_sum += ade

        pred_coord = pred_coord.squeeze()
        demd = mmfp_utils.wemd_from_pred_samples(pred_coord.cpu().numpy())
        all_demds.extend([demd])
        demd_sum += demd

        result_row = {
            "bidx": bidx,
            "pa": pas,
            "ade": float(ade),
            "demd": float(demd),
        }
        results.append(result_row)

        if args.draw and counter < args.num_heatmaps_drawn:
            # Draw some predictions
            img_path, frontal_img_path, ego_car, ref_obj, endpoint, command_text, frame_data = data_test.get_obj_info(
                bidx
            )
            drawn_img_hyps = draw_hyps_t2c(
                img_path, pred_coord, ego_car, ref_obj, endpoint, normalize=True
            )
            drawn_img_hyps.save(
                os.path.join(save_path, "test" + "-" + str(bidx) + "-hyps.png")
            )
            with open(
                os.path.join(save_path, "test" + "-" + str(bidx) + "-command.txt"), "w"
            ) as f:
                f.write(command_text)

        counter = counter + 1

    all_ades = torch.tensor(all_ades)
    all_ades_eb = torch.std(all_ades, dim=0).item() / (counter ** 0.5)
    all_pases = torch.tensor(all_pases)
    all_pases_eb = (torch.std(all_pases, dim=1) / (counter ** 0.5)).tolist()
    all_demds = torch.tensor(all_demds)
    all_demds_eb = torch.std(all_demds, dim=0).item() / (counter ** 0.5)

    print(f"Mean DEMD: {all_demds.mean().item()} +/- {all_demds_eb}")
    for i, threshold in enumerate(args.thresholds):
        print(f"Mean PA @ {threshold} : {all_pases[i].mean().item() * 100:.2f} +/- {all_pases_eb[i] * 100:.2f} %")
    print(f"Mean ADE: {all_ades.mean().item():.2f} +/- {all_ades_eb:.2f} m")
    print(f"Median ADE: {all_ades.median().item():.2f} m")

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(args)
