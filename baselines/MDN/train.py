import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from model import MDN
import argparse

parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)
parser.add_argument("--dataset", default="Talk2Car", required=False)
parser.add_argument("--data_dir", required=False, default="/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root")
parser.add_argument("--test", action="store_true", required=False)
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint to potentially continue training from.")
parser.add_argument("--seed", default=42, required=False)
parser.add_argument("--batch_size", default=16, required=False, type=int)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--save_dir", type=str, default="/home2/NoCsBack/hci/dusan/Results/MDN_Independent")
parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait for metric to improve before early stopping.")

# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
parser = MDN.add_model_specific_args(parser)

args = parser.parse_args()


def create_logger_name_for_args(args):
    name = ["MDN", args.dataset]

    name.append("backbone_{}".format(args.encoder))
    name.append("use_ref_obj_{}".format(args.use_ref_obj))
    name.append("lr_{}".format(args.lr))
    name.append("bs_{}".format(args.batch_size))
    name.append("epochs_{}".format(args.max_epochs))
    name.append("height_{}".format(args.height))
    name.append("width_{}".format(args.width))
    name.append("mdn_type_{}".format(args.mdn_type))
    name.append("conv_decay_{}".format(args.cov_decay))
    name.append("mu_diversity_{}".format(args.mu_diversity))
    name.append("pi_entropy_{}".format(args.pi_entropy))
    name.append("num_components_{}".format(args.num_components))

    return "_".join(name)


torch.backends.cudnn.benchmark = True


def main(args):
    seed_everything(args.seed)

    name = create_logger_name_for_args(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, 'lightning_logs'))

    wandb_logger = WandbLogger(
        name=name,
        project='MDN',
        save_dir=os.path.join(args.save_dir, 'lightning_logs')
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor="val_ade",
        mode="min",
        filename=name,
        verbose=True,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_ade',
        min_delta=0.0,
        patience=args.patience,
        verbose=True,
        mode='min'
    )

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        benchmark=True,
        gradient_clip_val=5,
        max_epochs=args.max_epochs,
        # overfit_batches=1
    )

    if args.test:
        model = MDN.load_from_checkpoint(args.checkpoint)
        trainer.test(model)
    else:
        model = MDN(args)
        # trainer.tune(model)
        trainer.fit(model)
        model = MDN.load_from_checkpoint(os.path.join(args.save_dir, name+".ckpt"))
        #trainer.save_checkpoint("final_" + name + ".ckpt")
        trainer.test(model)


if __name__ == "__main__":
    main(args)
