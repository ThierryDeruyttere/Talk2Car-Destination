import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from model import NonParametric
import argparse

parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)
parser.add_argument("--dataset", default="Talk2car", required=False)
parser.add_argument("--data_dir", required=False, default="/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root/")
parser.add_argument("--test", default=False, action="store_true", required=False)
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint to potentially continue training from.")
parser.add_argument("--seed", default=42, required=False)
parser.add_argument("--batch_size", default=16, required=False, type=int)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--save_dir", type=str, default="/home2/NoCsBack/hci/dusan/Results/NonParametric")
parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait for metric to improve before early stopping.")


# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
parser = NonParametric.add_model_specific_args(parser)

args = parser.parse_args()

def create_logger_name_for_args(args):
    name = ["NonParametric", args.dataset]

    name.append("backbone_{}".format(args.encoder))
    name.append("use_ref_obj_{}".format(args.use_ref_obj))
    name.append("lr_{}".format(args.lr))
    name.append("bs_{}".format(args.batch_size))
    name.append("epochs_{}".format(args.max_epochs))
    name.append("height_{}".format(args.height))
    name.append("width_{}".format(args.width))
    name.append("gaussian_size_{}".format(args.gaussian_size))
    name.append("gaussian_sigma_{}".format(args.gaussian_sigma))

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
        project='T2C_Path',
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
        # callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=5,
        max_epochs=args.max_epochs,
        overfit_batches=1
    )

    if args.test:
        model = NonParametric.load_from_checkpoint(args.checkpoint)
        trainer.test(model)
    else:
        model = NonParametric(args)
        trainer.fit(model)
        model = NonParametric.load_from_checkpoint(os.path.join(args.save_dir, name+".ckpt"))
        trainer.test(model)


if __name__ == "__main__":
    main(args)
