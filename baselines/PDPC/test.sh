#python test.py\
# --gpu_index $1\
# --data_dir "../../data"\
# --dataset "Talk2Car_Detector"\
# --draw\
# --num_heatmaps_drawn 30\
# --component_topk -1\
# --checkpoint_path "../FinalModels/FullConv_Talk2Car_Detector_use_ref_obj_True_lr_3e-05_bs_32_epochs_50_height_192_width_288_sigma_decay_0.0.ckpt"
#
 python test.py\
 --gpu_index $1\
 --data_dir "../../data"\
 --dataset "Talk2Car_Detector"\
 --draw\
 --num_heatmaps_drawn 30\
 --component_topk -1\
 --checkpoint_path "checkpoint/PDPC_Talk2Car_Detector_use_ref_obj_True_lr_3e-05_bs_32_epochs_50_height_192_width_288_sigma_decay_0.0_pi_entropy_0.0_n_conv_5_combine_at_2_active_scale_inds_[1,1,1,1]_inner_channel256.ckpt"