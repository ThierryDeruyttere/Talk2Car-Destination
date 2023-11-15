import argparse
import json
import os
import pathlib

import mmfp_utils
import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import PDPC
from talk2car import Talk2Car_Detector
from utils import draw_heatmap_t2c, draw_heatmap_frontal_t2c, draw_frontal_boxes, points_cam2img
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Talk2car_Detector", required=False)
parser.add_argument(
    "--data_dir",
    required=False,
    default="../../data"
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
parser.add_argument(
    "--component_topk",
    type=int,
    default=-1,
    help="How many components you choose for evaluation."
)
parser.add_argument(
    "--save_path",
    type=str,
    default="",
    help="Where to save the results."
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
        log_prob_grid = log_prob_grid.mean(dim=-1)
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
    if args.save_path == "":
        save_path = os.path.join(checkpoint_path[:-5], "results")
    else:
        save_path = args.save_path
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    hparams = checkpoint["hyper_parameters"]
    model = PDPC(hparams)
    model = model.to(device)
    model.load_state_dict(model_serialize)
    model.eval()

    if args.dataset == "Talk2Car_Detector":
        data_test = Talk2Car_Detector(
            split="test",
            dataset_root=args.data_dir,
            height=hparams["height"],
            width=hparams["width"],
            unrolled=hparams["unrolled"],
            use_ref_obj=hparams["use_ref_obj"]
        )
    # else:
    #     data_test = Talk2Car(
    #         split="val",
    #         root=args.data_dir,
    #         height=hparams["height"],
    #         width=hparams["width"],
    #         unrolled=hparams["unrolled"],
    #         use_ref_obj=hparams["use_ref_obj"]
    #     )

    nll_sum = 0.0
    demd_sum = 0.0
    pa_sums = [0.0 for _ in range(len(args.thresholds))]
    ade_sum = 0.0

    all_nlls = []
    all_demds = []
    all_pases = [[] for _ in range(len(args.thresholds))]
    all_ades = []

    # num_scale_0 = 0.0
    # num_scale_1 = 0.0
    # num_scale_2 = 0.0
    # num_scale_3 = 0.0

    counter = 0
    to_meters = torch.tensor([120.0, 80.0]).to(device)

    results = []
    test_loader = DataLoader(
        dataset=data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    criterion = model.criterion

    per_command_metrics = {}
    for bidx, data in enumerate(tqdm(test_loader)):

        x, y_gt, command = data["x"], data["y"], data["command_embedding"]
        x = x.float().to(device)
        y_gt = y_gt.float().to(device)
        [B, N, _] = y_gt.shape

        command = command.to(device)

        mu, sigma, pi, location = model.forward(x, command)
        mu, sigma, pi = model.predictor.prepare_outputs(mu, sigma, pi, location)

        # topk
        if args.component_topk > 0:
            pi, active_inds = pi.topk(args.component_topk, dim=1)
            mu = mu[:, active_inds.squeeze()]
            sigma = sigma[:, active_inds.squeeze()]
            # scale_ind_agg = scale_ind_agg[:, active_inds.squeeze()]

        pi = F.softmax(pi, dim=1)

        loss = criterion(
            mu, sigma, pi, y_gt
        )
        all_nlls.extend(loss.cpu().tolist())
        nll = loss.mean().item()
        nll_sum += nll

        comp = D.Independent(
            D.Normal(
                loc=mu,
                scale=sigma
            ), 1
        )
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)
        y_pred = mix.sample(sample_shape=(1000,))
        y_pred = y_pred.permute(1, 0, 2)

        distances = (
                (y_gt.unsqueeze(1) - y_pred.unsqueeze(2)) * to_meters.to(y_pred)
        ).norm(2, dim=-1).min(-1)[0]

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
            d = mmfp_utils.wemd_from_pred_samples((y_pred[i]).cpu().numpy())
            all_demds.extend([d])
            demd += d
        demd /= B
        demd_sum += demd

        # num_scale_0 += torch.count_nonzero((scale_ind_agg == 0.0).float())
        # num_scale_1 += torch.count_nonzero((scale_ind_agg == 1.0).float())
        # num_scale_2 += torch.count_nonzero((scale_ind_agg == 2.0).float())
        # num_scale_3 += torch.count_nonzero((scale_ind_agg == 3.0).float())

        result_row = {
            "bidx": bidx,
            "nll": float(nll),
            "demd": float(demd),
            "pa": pas,
            "ade": float(ade),
        }
        results.append(result_row)

        command_token = data["token"][0]
        per_command_metrics[command_token] = {
            "ade": ade,
            "demd": demd,
            "nll": nll,
        }
        for i, threshold in enumerate(args.thresholds):
            per_command_metrics[command_token]["pa_" + str(threshold)] = pas[i]

        y_pred = y_pred.squeeze().cpu().numpy()
        y_gt = y_gt.squeeze().cpu().numpy()

        if args.draw and counter < args.num_heatmaps_drawn:
            # Draw some predictions
            img_path, frontal_img_path, ego_car, ref_obj_pred, det_objs, _, command_text, frame_data = data_test.get_obj_info(
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

            # GT Ref Object
            item = data_test.data[bidx]

            gt_ref_index = item["gt_ref_obj_ix_frame_data"]
            gt_ref_box_coords = frame_data['map_objects_bbox'][gt_ref_index]
            gt_ref_box_coords_frontal = frame_data['image_objects_bbox'][gt_ref_index]

            draw_heatmap_t2c(
                ref_obj_pred,
                ego_car,
                img_path,
                y_gt,
                log_prob_grid,
                X,
                Y,
                save_path=os.path.join(
                    save_path, "test" + "-" + str(bidx) + f"-heatmap_top_down_topk_{args.component_topk}.png"
                ),
                det_objs=det_objs[:32],
                gt_ref_obj=gt_ref_box_coords
            )

            draw_heatmap_frontal_t2c(
                frame_data,
                frontal_img_path,
                log_prob_grid,
                X,
                Y,
                save_path=os.path.join(
                    save_path, "test" + "-" + str(bidx) + f"-heatmap_frontal_topk_{args.component_topk}.jpg"
                ),
                levels=20
            )


            # Detected Ref Object - 3D
            # detection_sample_index = data_test.command_index_mapping[item[0]["command_token"]]
            # detection_boxes = data_test.box_data[detection_sample_index]
            cam_intrinsic = frame_data["cam_intrinsic"]
            corners_3d_front = item["frontal_pred_box_corners"]
            if not isinstance(corners_3d_front, torch.Tensor):
                corners_3d_front = torch.from_numpy(np.array(corners_3d_front)).float()
            num_bbox = corners_3d_front.shape[0]
            points_3d_front = corners_3d_front.reshape(-1, 3)
            if not isinstance(cam_intrinsic, torch.Tensor):
                cam_intrinsic = torch.from_numpy(np.array(cam_intrinsic))
            cam_intrinsic = cam_intrinsic.reshape(3, 3).float().cpu()
            # project to 2d to get image coords (uv)
            uv_origin = points_cam2img(points_3d_front, cam_intrinsic)
            uv_origin = (uv_origin - 1).round()
            boxes_coords = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()
            boxes_coords = boxes_coords.tolist()
            det_ref_index = data["detection_pred_box_index"].item()

            draw_frontal_boxes(
                img_path=os.path.join(
                    save_path, "test" + "-" + str(bidx) + f"-heatmap_frontal_topk_{args.component_topk}.jpg"
                ),
                save_path=os.path.join(
                    save_path, "test" + "-" + str(bidx) + f"-heatmap_frontal_topk_{args.component_topk}.jpg"
                ),
                boxes_coords=boxes_coords,
                ref_ind=det_ref_index,
                remove_nonref=True,
                gt_box_coords=gt_ref_box_coords_frontal
            )

        counter = counter + 1

    # perc_scale_0 = num_scale_0 / counter / ((hparams["height"] / 4) * (hparams["width"] / 4))
    # perc_scale_1 = num_scale_1 / counter / ((hparams["height"] / 8) * (hparams["width"] / 8))
    # perc_scale_2 = num_scale_2 / counter / ((hparams["height"] / 16) * (hparams["width"] / 16))
    # perc_scale_3 = num_scale_3 / counter / ((hparams["height"] / 32) * (hparams["width"] / 32))
    #
    # num_scale_0 = num_scale_0 / counter
    # num_scale_1 = num_scale_1 / counter
    # num_scale_2 = num_scale_2 / counter
    # num_scale_3 = num_scale_3 / counter
    # num_scale_total = num_scale_0 + num_scale_1 + num_scale_2 + num_scale_3

    # all_nlls = torch.tensor(all_nlls)
    # all_nlls_eb = torch.std(all_nlls, dim=0) .item() / (counter ** 0.5)
    # all_ades = torch.tensor(all_ades)
    # all_ades_eb = torch.std(all_ades, dim=0).item() / (counter ** 0.5)
    # all_pases = torch.tensor(all_pases)
    # all_pases_eb = (torch.std(all_pases, dim=1) / (counter ** 0.5)).tolist()
    # all_demds = torch.tensor(all_demds)
    # all_demds_eb = torch.std(all_demds, dim=0).item() / (counter ** 0.5)

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(results, f)

    with open(os.path.join(save_path, f"per_command_metrics_fullconv_topk_{args.component_topk}.json"), "w") as f:
        json.dump(per_command_metrics, f)



if __name__ == "__main__":
    main(args)
