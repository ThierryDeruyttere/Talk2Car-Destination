python test.py\
 --gpu_index $1\
 --data_dir "../../data"\
 --dataset "Talk2Car_Detector"\
 --draw\
 --num_heatmaps_drawn 30\
 --component_topk -1\
 --checkpoint_path "../FinalModels/FullConv_Talk2Car_Detector_use_ref_obj_True_lr_3e-05_bs_32_epochs_50_height_192_width_288_sigma_decay_0.0.ckpt"