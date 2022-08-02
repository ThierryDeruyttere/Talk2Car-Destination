python test.py\
 --gpu $1\
 --data_dir "../../data"\
 --dataset "Talk2Car_Detector"\
 --draw\
 --num_heatmaps_drawn 30\
 --checkpoint_path "../FinalModels/DistributionPrediction_Talk2Car_Detector_backbone_ResNet-18_use_ref_obj_True_lr_0.0001_bs_16_epochs_50_height_200_width_300_mvg_type_independent_conv_decay_1.0.ckpt"
# --checkpoint_path "/export/home2/NoCsBack/hci/dusan/FinalModels/DistributionPrediction_Talk2Car_Detector_backbone_ResNet-18_use_ref_obj_True_lr_0.0001_bs_16_epochs_50_height_200_width_300_mvg_type_independent_conv_decay_1.0.ckpt"
# --checkpoint_path "/home2/NoCsBack/hci/dusan/BaselinesOutputs/DistributionPredictionSweep/DistributionPrediction_Talk2Car_Detector_backbone_ResNet-18_use_ref_obj_True_lr_0.0001_bs_16_epochs_50_height_200_width_300_mvg_type_independent_conv_decay_1.0.ckpt"