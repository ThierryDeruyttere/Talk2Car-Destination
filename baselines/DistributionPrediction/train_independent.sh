python train.py\
 --width 300\
 --height 200\
 --gpus $1,\
 --lr 3e-5\
 --num_workers 4\
 --batch_size 16\
 --max_epochs 50\
 --data_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"\
 --encoder "ResNet-18"\
 --dataset "Talk2Car_Detector"\
 --save_dir "/home2/NoCsBack/hci/dusan/Talk2Car_Path_Results/DistributionPrediction"\
 --use_ref_obj\
 --cov_decay 0.0\
 --mvg_type "independent"\
 --patience 10

python train.py\
 --width 300\
 --height 200\
 --gpus $1,\
 --lr 3e-5\
 --num_workers 4\
 --batch_size 16\
 --max_epochs 50\
 --data_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"\
 --encoder "ResNet-18"\
 --dataset "Talk2Car_Detector"\
 --save_dir "/home2/NoCsBack/hci/dusan/Talk2Car_Path_Results/DistributionPrediction"\
 --use_ref_obj\
 --cov_decay 1.0\
 --mvg_type "independent"\
 --patience 10

python train.py\
 --width 300\
 --height 200\
 --gpus $1,\
 --lr 3e-5\
 --num_workers 4\
 --batch_size 16\
 --max_epochs 50\
 --data_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"\
 --encoder "ResNet-18"\
 --dataset "Talk2Car_Detector"\
 --save_dir "/home2/NoCsBack/hci/dusan/Talk2Car_Path_Results/DistributionPrediction"\
 --use_ref_obj\
 --cov_decay 5.0\
 --mvg_type "independent"\
 --patience 10