import argparse
import json
import os
import pathlib

import mmfp_utils
import torch
import torch.nn.functional as F
import torch.distributions as D

from model import NonParametric
from talk2car import Talk2Car_Detector
from utils import draw_heatmap_t2c, draw_heatmap_frontal_t2c
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


def get_grid_axes(height, width):
    x = torch.linspace(0, width - 1, width // 1)
    y = torch.linspace(0, height - 1, height // 1)
    X, Y = torch.meshgrid(x, y)
    return X, Y

# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
# parser = NonParametric.add_model_specific_args(parser)
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
    model = NonParametric(hparams)
    model = model.to(device)
    model.load_state_dict(model_serialize)
    model.eval()

    data_test = Talk2Car_Detector(
        split="test",
        dataset_root=args.data_dir,
        height=hparams["height"],
        width=hparams["width"],
        unrolled=hparams["unrolled"],
        use_ref_obj=hparams["use_ref_obj"],
        gaussian_sigma=hparams["gaussian_sigma"],
        gaussian_size=hparams["gaussian_size"],
    )

    input_width = hparams["width"]
    input_height = hparams["height"]

    nll_sum = 0.0
    demd_sum = 0.0
    pa_sums = [0.0 for _ in range(len(args.thresholds))]
    ade_sum = 0.0

    all_nlls = []
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

        inp, (gt_x_axis, gt_y_axis), command = (
            data["x"],
            data["y"],
            data["command_embedding"],
        )
        gt_end_pos = data["end_pos"]
        [B, N, _] = gt_end_pos.shape

        inp = inp.float().to(device)
        gt_x_axis = gt_x_axis.float().to(device)
        gt_y_axis = gt_y_axis.float().to(device)
        gt_end_pos = gt_end_pos.to(device)
        command = command.to(device)
        x_probs, y_probs = model.forward(inp, command)
        loss = F.kl_div(
            torch.log_softmax(x_probs, dim=-1), gt_x_axis, size_average="batchmean"
        ) + F.kl_div(
            torch.log_softmax(y_probs, dim=-1), gt_y_axis, size_average="batchmean"
        )
        x_probs = F.softmax(x_probs, dim=1)
        y_probs = F.softmax(y_probs, dim=1)

        gt_x = gt_end_pos[:, :, 0].long()
        gt_y = gt_end_pos[:, :, 1].long()
        gt_coord = torch.cat((gt_x.unsqueeze(2), gt_y.unsqueeze(2)), dim=2)
        prob_grid = x_probs.unsqueeze(2).bmm(y_probs.unsqueeze(1))
        log_prob_grid = torch.log(prob_grid)
        log_py = torch.gather(
            log_prob_grid.unsqueeze(1).repeat(1, N, 1, 1).view(B, N, -1),
            2,
            (gt_coord[:, :, 0] * log_prob_grid.shape[-1] + gt_coord[:, :, 1]).unsqueeze(2)
        ).mean(dim=-1).mean(dim=-1)
        nll = -1.0 * log_py.mean().item()
        all_nlls.extend((-1.0 * log_py).cpu().tolist())
        nll_sum += nll

        x_distr = D.Categorical(probs=x_probs)
        y_distr = D.Categorical(probs=y_probs)
        pred_coord = torch.cat(
            (
                x_distr.sample(sample_shape=(1000,)).unsqueeze(2),
                y_distr.sample(sample_shape=(1000,)).unsqueeze(2),
            ),
            dim=2,
        ).float()
        pred_coord = pred_coord.permute(1, 0, 2)
        img_input_size = torch.tensor([input_width, input_height]).to(pred_coord)

        # Take smallest distance to one of the three end points
        distances = (
            (gt_end_pos.unsqueeze(1) - pred_coord.unsqueeze(2))
        )
        distances = distances / img_input_size * to_meters.to(distances)
        distances = distances.norm(2, dim=-1).min(-1)[0]

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

        demd = 0.0
        pred_coord = (pred_coord / img_input_size).cpu().numpy()
        for i in range(B):
            d = mmfp_utils.wemd_from_pred_samples(pred_coord[i])
            all_demds.extend([d])
            demd += d
        demd /= B
        demd_sum += demd

        result_row = {
            "bidx": bidx,
            "nll": float(nll),
            "demd": float(demd),
            "pa": pas,
            "ade": float(ade),
        }
        results.append(result_row)

        log_prob_grid = log_prob_grid.squeeze().cpu().numpy()
        gt_end_pos = gt_end_pos / img_input_size
        gt_end_pos = gt_end_pos.squeeze().cpu().numpy()

        if args.draw and counter < args.num_heatmaps_drawn:
            # Draw some predictions
            img_path, frontal_img_path, ego_car, ref_obj, _, command_text, frame_data = data_test.get_obj_info(
                bidx
            )
            with open(
                os.path.join(save_path, "test" + "-" + str(bidx) + "-command.txt"), "w"
            ) as f:
                f.write(command_text)

            # Overlay probability map over image
            _, _, height, width = inp.shape
            X, Y = get_grid_axes(height, width)
            X = X.cpu().numpy()
            Y = Y.cpu().numpy()

            draw_heatmap_t2c(
                ref_obj,
                ego_car,
                img_path,
                gt_end_pos,
                log_prob_grid,
                X,
                Y,
                save_path=os.path.join(
                    save_path, "test" + "-" + str(bidx) + "-heatmap_top_down.png"
                ),
            )

            draw_heatmap_frontal_t2c(
                frame_data,
                frontal_img_path,
                log_prob_grid,
                X,
                Y,
                save_path=os.path.join(
                    save_path, "test" + "-" + str(bidx) + "-heatmap_frontal.png"
                ),
            )
        else:
            break

        counter = counter + 1

    all_nlls = torch.tensor(all_nlls)
    all_nlls_eb = torch.std(all_nlls, dim=0) .item() / (counter ** 0.5)
    all_ades = torch.tensor(all_ades)
    all_ades_eb = torch.std(all_ades, dim=0).item() / (counter ** 0.5)
    all_pases = torch.tensor(all_pases)
    all_pases_eb = (torch.std(all_pases, dim=1) / (counter ** 0.5)).tolist()
    all_demds = torch.tensor(all_demds)
    all_demds_eb = torch.std(all_demds, dim=0).item() / (counter ** 0.5)


    print(f"Mean NLL: {all_nlls.mean().item():.2f} +/- {all_nlls_eb:.2f}")
    print(f"Mean DEMD: {all_demds.mean().item()} +/- {all_demds_eb}")
    for i, threshold in enumerate(args.thresholds):
        print(f"Mean PA @ {threshold} : {all_pases[i].mean().item() * 100:.2f} +/- {all_pases_eb[i] * 100:.2f} %")
    print(f"Mean ADE: {all_ades.mean().item():.2f} +/- {all_ades_eb:.2f} m")
    print(f"Median ADE: {all_ades.median().item():.2f} m")

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(args)
