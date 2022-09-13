python train.py\
 --width=300\
 --height=200\
 --gpus $1,\
 --lr 3e-5\
 --num_workers=4\
 --batch_size=16\
 --max_epochs 50\
 --num_components 3\
 --data_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"\
 --encoder "ResNet-18"\
 --dataset "Talk2Car_Detector"\
 --save_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/baselines/end_position/MDN_v2/checkpoints_v2"\
 --use_ref_obj\
 --pi_entropy 0.0\
 --mdn_type "independent"\
 --patience 10

python train.py\
 --width=300\
 --height=200\
 --gpus $1,\
 --lr 3e-5\
 --num_workers=4\
 --batch_size=16\
 --max_epochs 50\
 --num_components 3\
 --data_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"\
 --encoder "ResNet-18"\
 --dataset "Talk2Car_Detector"\
 --save_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/baselines/end_position/MDN_v2/checkpoints_v2"\
 --use_ref_obj\
 --pi_entropy 0.1\
 --mdn_type "independent"\
 --patience 10

python train.py\
 --width=300\
 --height=200\
 --gpus $1,\
 --lr 3e-5\
 --num_workers=4\
 --batch_size=16\
 --max_epochs 50\
 --num_components 3\
 --data_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"\
 --encoder "ResNet-18"\
 --dataset "Talk2Car_Detector"\
 --save_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/baselines/end_position/MDN_v2/checkpoints_v2"\
 --use_ref_obj\
 --cov_decay 1.0\
 --mdn_type "independent"\
 --patience 10

python train.py\
 --width=300\
 --height=200\
 --gpus $1,\
 --lr 3e-5\
 --num_workers=4\
 --batch_size=16\
 --max_epochs 50\
 --num_components 3\
 --data_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"\
 --encoder "ResNet-18"\
 --dataset "Talk2Car_Detector"\
 --save_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/baselines/end_position/MDN_v2/checkpoints_v2"\
 --use_ref_obj\
 --cov_decay 5.0\
 --mdn_type "independent"\
 --patience 10