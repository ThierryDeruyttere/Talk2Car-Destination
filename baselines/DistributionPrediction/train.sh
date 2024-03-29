python train.py\
 --width 300\
 --height 200\
 --gpus $1,\
 --lr 3e-5\
 --num_workers 4\
 --batch_size 16\
 --max_epochs 50\
 --data_dir "../../data"\
 --encoder "ResNet-18"\
 --dataset "Talk2Car_Detector"\
 --save_dir "checkpoint"\
 --use_ref_obj\
 --cov_decay 1.0\
 --mvg_type "independent"\
 --patience 10