_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus.py',
    '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

resume_from = '/mimer/NOBACKUP/groups/snic2021-7-127/enliden/databasesamplerv2/mmdetection3d/work_dirs_pp_no_fade/epoch_18.pth'
