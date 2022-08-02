import argparse
import json
import os
import pathlib

import mmfp_utils
import torch
import torch.distributions as D
from torch.utils.data import DataLoader

from model import MDN
import mdn_dependent, mdn_independent
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

def get_grid_logprob(height, width, mix, device):
    x = torch.linspace(0, width - 1, width // 1)
    y = torch.linspace(0, height - 1, height // 1)
    X, Y = torch.meshgrid(x, y)
    XY = torch.cat([X.contiguous().view(-1, 1), Y.contiguous().view(-1, 1)], dim=1).to(
        device
    )
    XY = XY / torch.tensor([width, height]).to(XY)
    log_prob_grid = mix.log_prob(XY)
    if len(log_prob_grid.shape) > 1:
        log_prob_grid = log_prob_grid.sum(dim=-1)
    log_prob_grid = log_prob_grid.view(width, height)
    return log_prob_grid, X, Y

# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
# parser = MDN.add_model_specific_args(parser)
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
    model = MDN(hparams)
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
    test_loader = DataLoader(
        dataset=data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    if hparams["mdn_type"] == "dependent":
        criterion = mdn_dependent.mdn_loss
    elif hparams.mdn_type == "independent":
        criterion = mdn_independent.mdn_loss
    json_output = {}
    for bidx, data in enumerate(tqdm(test_loader)):

        x, y_gt, command = data["x"], data["y"], data["command_embedding"]
        x = x.float().to(device)
        y_gt = y_gt.float().to(device)
        [B, N, _] = y_gt.shape

        command = command.to(device)

        pi, tril_or_sigma, mu = model.forward(x, command)

        loss = criterion(
            pi,
            tril_or_sigma,
            mu,
            y_gt
        )
        all_nlls.extend(loss.cpu().tolist())
        nll = loss.mean().item()
        nll_sum += nll

        if hparams["mdn_type"] == "dependent":
            comp = D.MultivariateNormal(loc=mu, scale_tril=tril_or_sigma)
        else:
            comp = D.Independent(D.Normal(loc=mu, scale=tril_or_sigma), 1)
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)
        y_pred = mix.sample(sample_shape=(1000,))
        y_pred = y_pred.permute(1, 0, 2)

        distances = (
            (y_gt.unsqueeze(1) - y_pred.unsqueeze(2)) * to_meters
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

        demd = 0.0
        for i in range(B):
            d = mmfp_utils.wemd_from_pred_samples((y_pred[i]* to_meters).cpu().numpy())
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

        y_pred = y_pred.squeeze().cpu().numpy()
        y_gt = y_gt.squeeze().cpu().numpy()

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
            _, _, height, width = x.shape
            log_prob_grid, X, Y = get_grid_logprob(height, width, mix, device)
            log_prob_grid = log_prob_grid.cpu().numpy()
            X = X.cpu().numpy()
            Y = Y.cpu().numpy()

            draw_heatmap_t2c(
                ref_obj,
                ego_car,
                img_path,
                y_gt,
                log_prob_grid,
                X,
                Y,
                save_path=os.path.join(
                    save_path, "test" + "-" + str(bidx) + "-heatmap.png"
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

        counter = counter + 1

        command_token = data["token"][0]
        json_output[command_token] = {
            "ade": ade,
            "demd": demd
        }
        for thres_ix, thres in enumerate(args.thresholds):

            json_output[command_token]["pa_"+str(thres)] = pas[thres_ix]

            #[[] for _ in range(len(args.thresholds))]


    json.dump(json_output, open(args.checkpoint_path.split("/")[-1].strip(".ckpt")+"_metrics.json", "w"))
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
