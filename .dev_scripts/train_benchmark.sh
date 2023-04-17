PARTITION=$1
WORK_DIR=$2
CPUS_PER_TASK=${3:-4}

echo 'configs/detection/detector_with_enhance_head/faster-rcnn_r50_fpn_basic-enhance_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION example-1 configs/detection/detector_with_enhance_head/faster-rcnn_r50_fpn_basic-enhance_1x_coco.py $WORK_DIR/detector_with_enhance_head/faster_r50_1x_basic_enhance_head --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &


MASTER_PORT=1235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_base configs/detection/self_enhance_detector/self_enhance_light/faster_r50_base_loss.py      work_dirs/work_dirs/self_enhance_light/faster_base_loss --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_all  configs/detection/self_enhance_detector/self_enhance_light/faster_r50_with_allloss.py   work_dirs/work_dirs/self_enhance_light/faster_all_loss --cfg-options dist_params.port=16666

SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_perc configs/detection/self_enhance_detector/self_enhance_light/faster_r50_with_perc.py      work_dirs/work_dirs/self_enhance_light/faster_with_perc --cfg-options dist_params.port=34788

SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_strc configs/detection/self_enhance_detector/self_enhance_light/faster_r50_with_structure.py work_dirs/work_dirs/self_enhance_light/faster_with_structure --cfg-options dist_params.port=47852


SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_base configs/detection/self_enhance_detector/self_enhance_light/retinanet_r50_base_loss.py       work_dirs/work_dirs/self_enhance_light/retina_base_loss --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_all  configs/detection/self_enhance_detector/self_enhance_light/retinanet_r50_with_allloss.py    work_dirs/work_dirs/self_enhance_light/retina_all_loss --cfg-options dist_params.port=16666

SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_perc configs/detection/self_enhance_detector/self_enhance_light/retinanet_r50_with_perc.py       work_dirs/work_dirs/self_enhance_light/retina_with_perc --cfg-options dist_params.port=34788

SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_strc configs/detection/self_enhance_detector/self_enhance_light/retinanet_r50_with_structure.py  work_dirs/work_dirs/self_enhance_light/retina_with_structure --cfg-options dist_params.port=47852



# base rtts
MASTER_PORT=1235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atts_rtts configs/detection/self_enhance_detector/base_detector/atss_fpn_1x_rtts.py                   work_dirs/work_dirs/self_enhance_light/rtts/atts_base   --cfg-options dist_params.port=1666

MASTER_PORT=1238 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_rtts configs/detection/self_enhance_detector/base_detector/faster_rcnn_r50_fpn_1x_rtts.py      work_dirs/work_dirs/self_enhance_light/rtts/faster_base --cfg-options dist_params.port=1666

MASTER_PORT=2331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det reti_rtts configs/detection/self_enhance_detector/base_detector/retinanet_r50_fpn_1x_rtts.py      work_dirs/work_dirs/self_enhance_light/rtts/retina_base --cfg-options dist_params.port=1666

MASTER_PORT=2313 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_rtts configs/detection/self_enhance_detector/base_detector/tood_fpn_1x_rtts.py      work_dirs/work_dirs/self_enhance_light/rtts/tood_base --cfg-options dist_params.port=1666


# base urpc
MASTER_PORT=3235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atts_urpc configs/detection/self_enhance_detector/base_detector/atss_fpn_1x_urpc2020.py                   work_dirs/work_dirs/self_enhance_light/urpc/atts_base   --cfg-options dist_params.port=1666

MASTER_PORT=3238 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_urpc configs/detection/self_enhance_detector/base_detector/faster_rcnn_r50_fpn_1x_urpc2020.py      work_dirs/work_dirs/self_enhance_light/urpc/faster_base --cfg-options dist_params.port=1666

MASTER_PORT=4331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det reti_urpc configs/detection/self_enhance_detector/base_detector/retinanet_r50_fpn_1x_urpc2020.py      work_dirs/work_dirs/self_enhance_light/urpc/retina_base --cfg-options dist_params.port=1666

MASTER_PORT=5313 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_urpc configs/detection/self_enhance_detector/base_detector/tood_fpn_1x_urpc2020.py      work_dirs/work_dirs/self_enhance_light/urpc/tood_base --cfg-options dist_params.port=1666



# experiment 1 of rtts
MASTER_PORT=1235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_base_loss configs/detection/self_enhance_detector/rtts/faster_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/rtts/faster/base_loss   --cfg-options dist_params.port=1666

MASTER_PORT=1238 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_no_loss configs/detection/self_enhance_detector/rtts/faster_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/rtts/faster/base_loss_detach --cfg-options dist_params.port=1666

MASTER_PORT=2331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_loss_detach configs/detection/self_enhance_detector/rtts/faster_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/faster/no_loss --cfg-options dist_params.port=1666


# experiment 1 of urpc
MASTER_PORT=1235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_base_loss configs/detection/self_enhance_detector/urpc/faster_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/urpc/faster/base_loss   --cfg-options dist_params.port=1666

MASTER_PORT=1238 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_no_loss configs/detection/self_enhance_detector/urpc/faster_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/urpc/faster/base_loss_detach --cfg-options dist_params.port=1666

MASTER_PORT=2331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_loss_detach configs/detection/self_enhance_detector/urpc/faster_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/urpc/faster/no_loss --cfg-options dist_params.port=1666


# experiment 1 of rtts
MASTER_PORT=1235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_base_loss configs/detection/self_enhance_detector/rtts/retinanet_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/rtts/retinanet/base_loss   --cfg-options dist_params.port=1666

MASTER_PORT=1238 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_no_loss configs/detection/self_enhance_detector/rtts/retinanet_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/rtts/retinanet/base_loss_detach --cfg-options dist_params.port=1666

MASTER_PORT=2331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_loss_detach configs/detection/self_enhance_detector/rtts/retinanet_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/retinanet/no_loss --cfg-options dist_params.port=1666


# experiment 1 of urpc
MASTER_PORT=1235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_base_loss configs/detection/self_enhance_detector/urpc/retinanet_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/urpc/retinanet/base_loss   --cfg-options dist_params.port=1666

MASTER_PORT=1238 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_no_loss configs/detection/self_enhance_detector/urpc/retinanet_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/urpc/retinanet/base_loss_detach --cfg-options dist_params.port=1666

MASTER_PORT=2331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_loss_detach configs/detection/self_enhance_detector/urpc/retinanet_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/urpc/retinanet/no_loss --cfg-options dist_params.port=1666


## exp 2
MASTER_PORT=4318  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc1 configs/detection/self_enhance_detector/urpc/atss_r50_base_loss.py    work_dirs/work_dirs/self_enhance_light/urpc/atts/base_loss   --cfg-options dist_params.port=1666

MASTER_PORT=4617  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc2 configs/detection/self_enhance_detector/urpc/atss_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/urpc/atts/no_loss --cfg-options dist_params.port=1666

MASTER_PORT=8183  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc3 configs/detection/self_enhance_detector/urpc/tood_r50_base_loss.py      work_dirs/work_dirs/self_enhance_light/urpc/tood/base_loss --cfg-options dist_params.port=1666

MASTER_PORT=8483  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc4 configs/detection/self_enhance_detector/urpc/tood_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/urpc/tood/no_loss --cfg-options dist_params.port=1666



MASTER_PORT=1318  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts1 configs/detection/self_enhance_detector/rtts/atss_r50_base_loss.py    work_dirs/work_dirs/self_enhance_light/rtts/atts/base_loss   --cfg-options dist_params.port=1666

MASTER_PORT=2617  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts2 configs/detection/self_enhance_detector/rtts/atss_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/atts/no_loss --cfg-options dist_params.port=1666

MASTER_PORT=3183  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts3 configs/detection/self_enhance_detector/rtts/tood_r50_base_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/tood/base_loss --cfg-options dist_params.port=1666

MASTER_PORT=7483  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts4 configs/detection/self_enhance_detector/rtts/tood_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/tood/no_loss --cfg-options dist_params.port=1666


# exp2 faster loss ab
MASTER_PORT=9183  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc1 configs/detection/self_enhance_detector/exp_2_ab_faster_loss/faster_r50_fft_loss.py           work_dirs/work_dirs/self_enhance_light/exp2_ab_faster_loss/fft_los --cfg-options dist_params.port=1666

MASTER_PORT=8483  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc2 configs/detection/self_enhance_detector/exp_2_ab_faster_loss/faster_r50_spactial_loss.py      work_dirs/work_dirs/self_enhance_light/exp2_ab_faster_loss/spactial_loss --cfg-options dist_params.port=1666

MASTER_PORT=1318  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc3 configs/detection/self_enhance_detector/exp_2_ab_faster_loss/faster_r50_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp2_ab_faster_loss/struc_loss   --cfg-options dist_params.port=1666

MASTER_PORT=2617  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc4 configs/detection/self_enhance_detector/exp_2_ab_faster_loss/faster_r50_tv_loss.py            work_dirs/work_dirs/self_enhance_light/exp2_ab_faster_loss/tv_loss --cfg-options dist_params.port=1666

MASTER_PORT=3183  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc5 configs/detection/self_enhance_detector/exp_2_ab_faster_loss/faster_r50_with_fft_loss.py      work_dirs/work_dirs/self_enhance_light/exp2_ab_faster_loss/with_fft_loss --cfg-options dist_params.port=1666

MASTER_PORT=7483  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc6 configs/detection/self_enhance_detector/exp_2_ab_faster_loss/faster_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp2_ab_faster_loss/with_struc_loss --cfg-options dist_params.port=1666

MASTER_PORT=8483  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc2 configs/detection/self_enhance_detector/exp_2_ab_faster_loss/faster_r50_spactial_tv_loss.py      work_dirs/work_dirs/self_enhance_light/exp2_ab_faster_loss/spactial_tv_loss --cfg-options dist_params.port=1666


# exp2 retina loss ab
MASTER_PORT=9321  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r1 configs/detection/self_enhance_detector/exp_2_ab_retina_loss/retina_r50_fft_loss.py           work_dirs/work_dirs/self_enhance_light/exp2_ab_retina_loss/fft_los --cfg-options dist_params.port=1666

MASTER_PORT=8186  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r2 configs/detection/self_enhance_detector/exp_2_ab_retina_loss/retina_r50_spactial_loss.py      work_dirs/work_dirs/self_enhance_light/exp2_ab_retina_loss/spactial_loss --cfg-options dist_params.port=1666

MASTER_PORT=3138  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r3 configs/detection/self_enhance_detector/exp_2_ab_retina_loss/retina_r50_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp2_ab_retina_loss/struc_loss   --cfg-options dist_params.port=1666

MASTER_PORT=2917  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r4 configs/detection/self_enhance_detector/exp_2_ab_retina_loss/retina_r50_tv_loss.py            work_dirs/work_dirs/self_enhance_light/exp2_ab_retina_loss/tv_loss --cfg-options dist_params.port=1666

MASTER_PORT=3347  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r5 configs/detection/self_enhance_detector/exp_2_ab_retina_loss/retina_r50_with_fft_loss.py      work_dirs/work_dirs/self_enhance_light/exp2_ab_retina_loss/with_fft_loss --cfg-options dist_params.port=1666

MASTER_PORT=1962  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r6 configs/detection/self_enhance_detector/exp_2_ab_retina_loss/retina_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp2_ab_retina_loss/with_struc_loss --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto'

SRUN_ARGS='--quotatype=auto' MASTER_PORT=3347  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r5 configs/detection/self_enhance_detector/exp2_atss_ab_loss/atss_r50_base_loss.py      work_dirs/work_dirs/self_enhance_light/exp2_atss_ab_loss/base_loss --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1962  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r6 configs/detection/self_enhance_detector/exp2_atss_ab_loss/atss_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp2_atss_ab_loss/with_struc_loss --cfg-options dist_params.port=1666

# voc
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det voc_f   configs/detection/self_enhance_detector/exp3_voc/base_faster.py         work_dirs/work_dirs/self_enhance_light/exp3_voc/base_faster --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det voc_r   configs/detection/self_enhance_detector/exp3_voc/base_retina.py         work_dirs/work_dirs/self_enhance_light/exp3_voc/base_retina --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4931  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det voc_f_1 configs/detection/self_enhance_detector/exp3_voc/faster_base_loss.py    work_dirs/work_dirs/self_enhance_light/exp3_voc/faster_base_loss --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6128  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det voc_r_1 configs/detection/self_enhance_detector/exp3_voc/retina_base_loss.py    work_dirs/work_dirs/self_enhance_light/exp3_voc/retina_base_loss --cfg-options dist_params.port=1666


# try low pass
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_low_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp3_voc/faster_r64_low --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_low_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp3_voc/faster_r64_low_cm --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=10932  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_low_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp3_voc/faster_r96_low --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=96

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2831  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_low_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp3_voc/faster_r32_low --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=32

SRUN_ARGS='--quotatype=auto' MASTER_PORT=3161  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_low_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp3_voc/faster_r96_low_cm --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=96 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2347  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_low_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp3_voc/faster_r32_low_cm --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=32 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4931  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina configs/detection/self_enhance_detector/exp_3_low_strc_loss/retina_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp3_voc/retina_r64_low --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6128  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina configs/detection/self_enhance_detector/exp_3_low_strc_loss/retina_r50_with_struc_loss.py   work_dirs/work_dirs/self_enhance_light/exp3_voc/retina_r64_low_cm --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4931  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina configs/detection/self_enhance_detector/exp_3_low_strc_loss/retina_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp3_voc/retina_r96_low --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=96

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6128  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina configs/detection/self_enhance_detector/exp_3_low_strc_loss/retina_r50_with_struc_loss.py   work_dirs/work_dirs/self_enhance_light/exp3_voc/retina_r32_low --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=32

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1842  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina configs/detection/self_enhance_detector/exp_3_low_strc_loss/retina_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp3_voc/retina_r96_low_cm --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=96 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1373  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina configs/detection/self_enhance_detector/exp_3_low_strc_loss/retina_r50_with_struc_loss.py   work_dirs/work_dirs/self_enhance_light/exp3_voc/retina_r32_low_cm --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=32 model.enhance_model.generator.structure_loss.channel_mean=True



# try high pass
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r8_high     --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r8_high_cm  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1093  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r4_high     --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=4

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2831  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r16_high    --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=16

SRUN_ARGS='--quotatype=auto' MASTER_PORT=3161  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r4_high_cm  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=4 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2347  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r16_high_cm --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=16 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4931  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r8_high      --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6128  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r8_high_cm   --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4931  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r4_high      --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=4

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6128  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r16_high     --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=16

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1842  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r4_high_cm   --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=4 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1373  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_low_strc_loss/retina_r50_with_struc_loss.py          work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r16_high_cm  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=16 model.enhance_model.generator.structure_loss.channel_mean=True


# try high pass with gf
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r4_high_gh     --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r4_high_cm_gh  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1093  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r8_high_gh     --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=8

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2831  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r8_high_cm_gh    --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=8 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4931  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r4_high_gh      --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6128  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r4_high_cm_gh   --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4931  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r8_high_gh      --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=8

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6128  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r8_high_cm_gh     --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=8 model.enhance_model.generator.structure_loss.channel_mean=True


# try high pass with gf, loss weight 0.1
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r4_high_gh_lw01     --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r4_high_cm_gh_lw01  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1093  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r8_high_gh_lw01     --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=8 model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2831  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/faster_r8_high_cm_gh_lw01    --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=8 model.enhance_model.generator.structure_loss.channel_mean=True model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4931  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r4_high_gh_lw01      --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6128  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r4_high_cm_gh_lw01   --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4931  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r8_high_gh_lw01      --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=8 model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6128  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina   configs/detection/self_enhance_detector/exp_3_high_strc_loss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/retina_r8_high_cm_gh_lw01     --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.radius=8 model.enhance_model.generator.structure_loss.channel_mean=True model.enhance_model.generator.structure_loss.loss_weight=0.1


SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/atss_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/atss_r4_high_gh     --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/atss_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/ats_r4_high_cm_gh  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/tood_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/tood_r4_high_gh    --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/tood_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/tood_r4_high_cm_gh  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True


SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/atss_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/atss_r4_high_gh_lw01     --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/atss_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/ats_r4_high_cm_gh_lw01  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/tood_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/tood_r4_high_gh_lw01    --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster   configs/detection/self_enhance_detector/exp_3_high_strc_loss/tood_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_3_high_strc_loss/tood_r4_high_cm_gh_lw01  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True model.enhance_model.generator.structure_loss.loss_weight=0.1



SRUN_ARGS='--quotatype=auto' MASTER_PORT=1325  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atss_rtss   configs/detection/self_enhance_detector/urpc_with_strucloss/atss_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/urpc_with_strucloss/atss_r4_high_gh     --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=12670  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atss_rtss   configs/detection/self_enhance_detector/urpc_with_strucloss/atss_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/urpc_with_strucloss/ats_r4_high_cm_gh  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4792  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_rtss   configs/detection/self_enhance_detector/urpc_with_strucloss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/urpc_with_strucloss/faster_r4_high_gh    --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7093  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_rtss   configs/detection/self_enhance_detector/urpc_with_strucloss/faster_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/urpc_with_strucloss/faster_r4_high_cm_gh  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=3489  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_rtss   configs/detection/self_enhance_detector/urpc_with_strucloss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/urpc_with_strucloss/retina_r4_high_gh     --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2392  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_rtss   configs/detection/self_enhance_detector/urpc_with_strucloss/retina_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/urpc_with_strucloss/retina_r4_high_cm_gh  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7362  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_rtss   configs/detection/self_enhance_detector/urpc_with_strucloss/tood_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/urpc_with_strucloss/tood_r4_high_gh    --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=5932  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_rtss   configs/detection/self_enhance_detector/urpc_with_strucloss/tood_r50_with_struc_loss.py         work_dirs/work_dirs/self_enhance_light/urpc_with_strucloss/tood_r4_high_cm_gh  --cfg-options dist_params.port=1666 model.enhance_model.generator.structure_loss.channel_mean=True


# exp 4
SRUN_ARGS='--quotatype=auto' MASTER_PORT=1325  ./tools/slurm_train.sh mm_det voc_f_base   configs/detection/self_enhance_detector/exp_4_extra_experiment/voc_base_faster.py                   work_dirs/work_dirs/self_enhance_light/voc/voc_base_faster              --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1267  ./tools/slurm_train.sh mm_det voc_r_base   configs/detection/self_enhance_detector/exp_4_extra_experiment/voc_base_retina.py                   work_dirs/work_dirs/self_enhance_light/voc/voc_base_retina              --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4792  ./tools/slurm_train.sh mm_det voc_f_se     configs/detection/self_enhance_detector/exp_4_extra_experiment/voc_faster_with_strucloss.py         work_dirs/work_dirs/self_enhance_light/voc/voc_faster_with_strucloss    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7093  ./tools/slurm_train.sh mm_det voc_r_se     configs/detection/self_enhance_detector/exp_4_extra_experiment/voc_retina_with_strucloss.py         work_dirs/work_dirs/self_enhance_light/voc/voc_retina_with_strucloss    --cfg-options randomness.seed=None


SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_base     configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_faster_r50_base_loss.py          work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_faster_r50_base_loss          --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_struc    configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_faster_r50_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_faster_r50_struc_loss         --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_wstruc   configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_faster_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_faster_r50_with_struc_loss    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_struc    configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_retina_r50_struc_loss.py         work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_retina_r50_struc_loss         --cfg-options dist_params.port=1666 randomness.seed=None


SRUN_ARGS='--quotatype=auto' MASTER_PORT=1127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_wstruc   configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_faster_r50_no_loss.py    work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_faster_no_loss    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=6370  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_struc    configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_retina_r50_no_loss.py         work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_retina_no_loss         --cfg-options dist_params.port=1666 randomness.seed=None


SRUN_ARGS='--quotatype=auto' MASTER_PORT=4792  ./tools/slurm_train.sh mm_det voc_f_se     configs/detection/self_enhance_detector/exp_4_extra_experiment/voc_faster_with_strucloss.py         work_dirs/work_dirs/self_enhance_light/voc/voc_faster_with_strucloss_new    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7093  ./tools/slurm_train.sh mm_det voc_r_se     configs/detection/self_enhance_detector/exp_4_extra_experiment/voc_retina_with_strucloss.py         work_dirs/work_dirs/self_enhance_light/voc/voc_retina_with_strucloss_new    --cfg-options randomness.seed=None


SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_wstruc   configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_faster_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_faster_r50_with_struc_loss_detach    --cfg-options randomness.seed=None  model.detach_enhance_img=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7127  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_wstruc   configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_retina_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_retina_r50_with_struc_loss_detach    --cfg-options randomness.seed=None  model.detach_enhance_img=True

SRUN_ARGS='--quotatype=auto' MASTER_PORT=3138  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r3 configs/detection/self_enhance_detector/exp_2_ab_retina_loss/retina_r50_spactial_loss.py         work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/retina_spactialloss   --cfg-options  randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2917  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_r4 configs/detection/self_enhance_detector/exp_2_ab_retina_loss/retina_r50_tv_loss.py            work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/retina_tv_loss --cfg-options  randomness.seed=None


SRUN_ARGS='--quotatype=auto' MASTER_PORT=8134  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_wstruc   configs/detection/self_enhance_detector/exp_4_extra_experiment/rtss_faster_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/rtss_faster_r50_with_struc_loss    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=8134  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_wstruc   configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_retina_r50_with_base_loss.py    work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/retina_spactial_tv_loss    --cfg-options randomness.seed=None


## structure

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1325  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_111   configs/detection/self_enhance_detector/exp_5_structure/urpc_faster_r50_with_struc_loss_111.py         work_dirs/work_dirs/self_enhance_light/exp_5_structure/faster_with_struc_loss_111  --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1670  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_333   configs/detection/self_enhance_detector/exp_5_structure/urpc_faster_r50_with_struc_loss_333.py         work_dirs/work_dirs/self_enhance_light/exp_5_structure/faster_with_struc_loss_333  --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=4792  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_555   configs/detection/self_enhance_detector/exp_5_structure/urpc_faster_r50_with_struc_loss_555.py         work_dirs/work_dirs/self_enhance_light/exp_5_structure/faster_with_struc_loss_555  --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7093  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_111   configs/detection/self_enhance_detector/exp_5_structure/urpc_retina_r50_with_struc_loss_111.py         work_dirs/work_dirs/self_enhance_light/exp_5_structure/retina_with_struc_loss_111  --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=3489  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_333   configs/detection/self_enhance_detector/exp_5_structure/urpc_retina_r50_with_struc_loss_333.py         work_dirs/work_dirs/self_enhance_light/exp_5_structure/retina_with_struc_loss_333  --cfg-options dist_params.port=1666

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2392  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_555   configs/detection/self_enhance_detector/exp_5_structure/urpc_retina_r50_with_struc_loss_555.py         work_dirs/work_dirs/self_enhance_light/exp_5_structure/retina_with_struc_loss_555  --cfg-options dist_params.port=1666


SRUN_ARGS='--quotatype=auto' MASTER_PORT=4792  ./tools/slurm_train.sh mm_det voc_f_se     configs/detection/self_enhance_detector/exp_4_extra_experiment/voc_faster_with_strucloss.py         work_dirs/work_dirs/self_enhance_light/voc/voc_faster_with_strucloss    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7093  ./tools/slurm_train.sh mm_det voc_r_se     configs/detection/self_enhance_detector/exp_4_extra_experiment/voc_retina_with_strucloss.py         work_dirs/work_dirs/self_enhance_light/voc/voc_retina_with_strucloss    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_wstruc   configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_faster_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_faster_r50_with_struc_loss_tmp    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_wstruc   configs/detection/self_enhance_detector/exp_4_extra_experiment/urpc_faster_r50_with_struc_loss.py    work_dirs/work_dirs/self_enhance_light/exp_4_extra_experiment/urpc_faster_r50_with_struc_loss_tmp2    --cfg-options randomness.seed=None


# --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_rtts   configs/detection/cycle_det/cycle_det_faster/cycle_faster_ufpn_1x_rtts.py           work_dirs/work_dirs/cycle_det/faster/cycle_faster_ufpn_1x_rtts

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_urpc   configs/detection/cycle_det/cycle_det_faster/cycle_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/cycle_det/faster/cycle_faster_ufpn_1x_urpc2020

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atss_rtts     configs/detection/cycle_det/cycle_det_atss/cycle_atss_ufpn_1x_rtts.py               work_dirs/work_dirs/cycle_det/atss/cycle_atss_ufpn_1x_rtts

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atss_urpc     configs/detection/cycle_det/cycle_det_atss/cycle_atss_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/atss/cycle_atss_ufpn_1x_urpc2020

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_rtts   configs/detection/cycle_det/cycle_det_retina/cycle_retinanet_ufpn_1x_rtts.py        work_dirs/work_dirs/cycle_det/retina/cycle_retinanet_ufpn_1x_rtts

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_urpc   configs/detection/cycle_det/cycle_det_retina/cycle_retinanet_ufpn_1x_urpc2020.py    work_dirs/work_dirs/cycle_det/retina/cycle_retinanet_ufpn_1x_urpc2020

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_rtts     configs/detection/cycle_det/cycle_det_tood/cycle_tood_ufpn_1x_rtts.py               work_dirs/work_dirs/cycle_det/tood/cycle_tood_ufpn_1x_rtts

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_urpc     configs/detection/cycle_det/cycle_det_tood/cycle_tood_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/tood/cycle_tood_ufpn_1x_urpc2020



SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_rtts   configs/detection/cycle_det/cycle_det_faster/cycle_faster_ufpn_1x_rtts.py           work_dirs/work_dirs/cycle_det/faster/cycle_faster_ufpn_1x_rtts_random  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_urpc   configs/detection/cycle_det/cycle_det_faster/cycle_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/cycle_det/faster/cycle_faster_ufpn_1x_urpc2020_random  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atss_rtts     configs/detection/cycle_det/cycle_det_atss/cycle_atss_ufpn_1x_rtts.py               work_dirs/work_dirs/cycle_det/atss/cycle_atss_ufpn_1x_rtts_random  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atss_urpc     configs/detection/cycle_det/cycle_det_atss/cycle_atss_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/atss/cycle_atss_ufpn_1x_urpc2020_random  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_rtts   configs/detection/cycle_det/cycle_det_retina/cycle_retinanet_ufpn_1x_rtts.py        work_dirs/work_dirs/cycle_det/retina/cycle_retinanet_ufpn_1x_rtts_random  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_urpc   configs/detection/cycle_det/cycle_det_retina/cycle_retinanet_ufpn_1x_urpc2020.py    work_dirs/work_dirs/cycle_det/retina/cycle_retinanet_ufpn_1x_urpc2020_random  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_rtts     configs/detection/cycle_det/cycle_det_tood/cycle_tood_ufpn_1x_rtts.py               work_dirs/work_dirs/cycle_det/tood/cycle_tood_ufpn_1x_rtts_random  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_urpc     configs/detection/cycle_det/cycle_det_tood/cycle_tood_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/tood/cycle_tood_ufpn_1x_urpc2020_random  --cfg-options randomness.seed=None



SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_rtts   configs/detection/cycle_det/cycle_det_faster/cycle_faster_ufpn_1x_rtts.py           work_dirs/work_dirs/cycle_det/faster/cycle_faster_ufpn_1x_rtts_random_s1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_urpc   configs/detection/cycle_det/cycle_det_faster/cycle_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/cycle_det/faster/cycle_faster_ufpn_1x_urpc2020_random_s1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=2561  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atss_rtts     configs/detection/cycle_det/cycle_det_atss/cycle_atss_ufpn_1x_rtts.py               work_dirs/work_dirs/cycle_det/atss/cycle_atss_ufpn_1x_rtts_random_s1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=1361  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det atss_urpc     configs/detection/cycle_det/cycle_det_atss/cycle_atss_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/atss/cycle_atss_ufpn_1x_urpc2020_random_s1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_rtts   configs/detection/cycle_det/cycle_det_retina/cycle_retinanet_ufpn_1x_rtts.py        work_dirs/work_dirs/cycle_det/retina/cycle_retinanet_ufpn_1x_rtts_random_s1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_urpc   configs/detection/cycle_det/cycle_det_retina/cycle_retinanet_ufpn_1x_urpc2020.py    work_dirs/work_dirs/cycle_det/retina/cycle_retinanet_ufpn_1x_urpc2020_random_s1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_rtts     configs/detection/cycle_det/cycle_det_tood/cycle_tood_ufpn_1x_rtts.py               work_dirs/work_dirs/cycle_det/tood/cycle_tood_ufpn_1x_rtts_random_s1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det tood_urpc     configs/detection/cycle_det/cycle_det_tood/cycle_tood_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/tood/cycle_tood_ufpn_1x_urpc2020_random_s1  --cfg-options randomness.seed=None


# 2 gpu
SRUN_ARGS='--quotatype=auto' MASTER_PORT=1157  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det faster_rtts   configs/detection/cycle_det/cycle_det_faster/cycle_faster_ufpn_1x_rtts.py           work_dirs/work_dirs/cycle_det/faster/cycle_faster_ufpn_1x_rtts_random_s1  --auto-scale-lr  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=2561  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det atss_rtts     configs/detection/cycle_det/cycle_det_atss/cycle_atss_ufpn_1x_rtts.py               work_dirs/work_dirs/cycle_det/atss/cycle_atss_ufpn_1x_rtts_random_s1  --auto-scale-lr  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det retina_rtts   configs/detection/cycle_det/cycle_det_retina/cycle_retinanet_ufpn_1x_rtts.py        work_dirs/work_dirs/cycle_det/retina/cycle_retinanet_ufpn_1x_rtts_random_s1  --auto-scale-lr  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det tood_rtts     configs/detection/cycle_det/cycle_det_tood/cycle_tood_ufpn_1x_rtts.py               work_dirs/work_dirs/cycle_det/tood/cycle_tood_ufpn_1x_rtts_random_s1  --auto-scale-lr  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=2157  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det faster_urpc   configs/detection/cycle_det/cycle_det_faster/cycle_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/cycle_det/faster/cycle_faster_ufpn_1x_urpc2020_random_s1  --auto-scale-lr  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=1361  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det atss_urpc     configs/detection/cycle_det/cycle_det_atss/cycle_atss_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/atss/cycle_atss_ufpn_1x_urpc2020_random_s1  --auto-scale-lr  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det retina_urpc   configs/detection/cycle_det/cycle_det_retina/cycle_retinanet_ufpn_1x_urpc2020.py    work_dirs/work_dirs/cycle_det/retina/cycle_retinanet_ufpn_1x_urpc2020_random_s1  --auto-scale-lr  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det tood_urpc     configs/detection/cycle_det/cycle_det_tood/cycle_tood_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/tood/cycle_tood_ufpn_1x_urpc2020_random_s1  --auto-scale-lr  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7539  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det faster_urpc   configs/detection/cycle_det/cycle_det_faster/cycle_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/cycle_det/faster/cycle_faster_ufpn_1x_urpc2020_random_bs8  --auto-scale-lr  --cfg-options randomness.seed=None train_dataloader.batch_size=8  train_dataloader.num_workers=8
SRUN_ARGS='--quotatype=auto' MASTER_PORT=1329  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det atss_urpc     configs/detection/cycle_det/cycle_det_atss/cycle_atss_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/atss/cycle_atss_ufpn_1x_urpc2020_random_bs8  --auto-scale-lr  --cfg-options randomness.seed=None train_dataloader.batch_size=8  train_dataloader.num_workers=8
SRUN_ARGS='--quotatype=auto' MASTER_PORT=4802  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det retina_urpc   configs/detection/cycle_det/cycle_det_retina/cycle_retinanet_ufpn_1x_urpc2020.py    work_dirs/work_dirs/cycle_det/retina/cycle_retinanet_ufpn_1x_urpc2020_random_bs8  --auto-scale-lr  --cfg-options randomness.seed=None train_dataloader.batch_size=8  train_dataloader.num_workers=8
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3480  GPUS=2  GPUS_PER_NODE=2  ./tools/slurm_train.sh mm_det tood_urpc     configs/detection/cycle_det/cycle_det_tood/cycle_tood_ufpn_1x_urpc2020.py           work_dirs/work_dirs/cycle_det/tood/cycle_tood_ufpn_1x_urpc2020_random_bs8  --auto-scale-lr  --cfg-options randomness.seed=None train_dataloader.batch_size=8  train_dataloader.num_workers=8



source plantform/lqit
cd code/lqit/lqit

SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_atss    configs/detection/aenet/urpc/aenet_atss_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/atss/test_1    --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_faster  configs/detection/aenet/urpc/aenet_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/faster/test_1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_retina  configs/detection/aenet/urpc/aenet_retina_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/retina/test_1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_tood    configs/detection/aenet/urpc/aenet_tood_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/tood/test_1    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts_atss    configs/detection/aenet/rtts/aenet_atss_ufpn_1x_rtts.py         work_dirs/work_dirs/aenet_det/rtts/atss/test_1    --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts_faster  configs/detection/aenet/rtts/aenet_faster_ufpn_1x_rtts.py       work_dirs/work_dirs/aenet_det/rtts/faster/test_1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts_retina  configs/detection/aenet/rtts/aenet_retina_ufpn_1x_rtts.py       work_dirs/work_dirs/aenet_det/rtts/retina/test_1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts_tood    configs/detection/aenet/rtts/aenet_tood_ufpn_1x_rtts.py         work_dirs/work_dirs/aenet_det/rtts/tood/test_1    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1325  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det voc_faster  configs/detection/aenet/voc/aenet_faster_ufpn_1x_voc.py       work_dirs/work_dirs/aenet_det/voc/faster/test_1  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=8254  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det voc_retina  configs/detection/aenet/voc/aenet_retina_ufpn_1x_voc.py       work_dirs/work_dirs/aenet_det/voc/retina/test_1  --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_atss    configs/detection/aenet/urpc_exp2/aenet_atss_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/atss/test_2    --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_faster  configs/detection/aenet/urpc_exp2/aenet_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/faster/test_2  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_retina  configs/detection/aenet/urpc_exp2/aenet_retina_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/retina/test_2  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_tood    configs/detection/aenet/urpc_exp2/aenet_tood_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/tood/test_2    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_atss    configs/detection/aenet/urpc_exp2/aenet_atss_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/atss/test_3    --cfg-options randomness.seed=None model.enhance_head.structure_loss.loss_weight=0.1
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_faster  configs/detection/aenet/urpc_exp2/aenet_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/faster/test_3  --cfg-options randomness.seed=None model.enhance_head.structure_loss.loss_weight=0.1
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_retina  configs/detection/aenet/urpc_exp2/aenet_retina_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/retina/test_3  --cfg-options randomness.seed=None model.enhance_head.structure_loss.loss_weight=0.1
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_tood    configs/detection/aenet/urpc_exp2/aenet_tood_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/tood/test_3    --cfg-options randomness.seed=None model.enhance_head.structure_loss.loss_weight=0.1

SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_atss    configs/detection/aenet/urpc_exp3/aenet_atss_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/atss/test_4    --cfg-options randomness.seed=0
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_faster  configs/detection/aenet/urpc_exp3/aenet_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/faster/test_4  --cfg-options randomness.seed=0
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_retina  configs/detection/aenet/urpc_exp3/aenet_retina_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/retina/test_4  --cfg-options randomness.seed=0
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_tood    configs/detection/aenet/urpc_exp3/aenet_tood_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/tood/test_4    --cfg-options randomness.seed=0

SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_atss    configs/detection/aenet/urpc_exp3/aenet_atss_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/atss/test_5    --cfg-options randomness.seed=0 model.enhance_head.structure_loss.loss_weight=0.1
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_faster  configs/detection/aenet/urpc_exp3/aenet_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/faster/test_5  --cfg-options randomness.seed=0 model.enhance_head.structure_loss.loss_weight=0.1
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_retina  configs/detection/aenet/urpc_exp3/aenet_retina_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/retina/test_5  --cfg-options randomness.seed=0 model.enhance_head.structure_loss.loss_weight=0.1
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_tood    configs/detection/aenet/urpc_exp3/aenet_tood_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/tood/test_5    --cfg-options randomness.seed=0 model.enhance_head.structure_loss.loss_weight=0.1



SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_atss    configs/detection/aenet/urpc/aenet_atss_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/atss/randomseed    --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_faster  configs/detection/aenet/urpc/aenet_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/faster/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_retina  configs/detection/aenet/urpc/aenet_retina_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/retina/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_tood    configs/detection/aenet/urpc/aenet_tood_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/tood/randomseed    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts_atss    configs/detection/aenet/rtts/aenet_atss_ufpn_1x_rtts.py         work_dirs/work_dirs/aenet_det/rtts/atss/randomseed    --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts_faster  configs/detection/aenet/rtts/aenet_faster_ufpn_1x_rtts.py       work_dirs/work_dirs/aenet_det/rtts/faster/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts_retina  configs/detection/aenet/rtts/aenet_retina_ufpn_1x_rtts.py       work_dirs/work_dirs/aenet_det/rtts/retina/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts_tood    configs/detection/aenet/rtts/aenet_tood_ufpn_1x_rtts.py         work_dirs/work_dirs/aenet_det/rtts/tood/randomseed    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=1325  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det voc_faster  configs/detection/aenet/voc/aenet_faster_ufpn_1x_voc.py       work_dirs/work_dirs/aenet_det/voc/faster/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=8254  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det voc_retina  configs/detection/aenet/voc/aenet_retina_ufpn_1x_voc.py       work_dirs/work_dirs/aenet_det/voc/retina/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=8254  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det voc_retina  configs/detection/aenet/voc/aenet_retina_ufpn_1x_voc.py       work_dirs/work_dirs/aenet_det/voc/retina/randomseed_2  --cfg-options randomness.seed=None


SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_fpn    configs/detection/aenet/exp_fpn/aenet_faster_fpn_1x_urpc2020.py        work_dirs/work_dirs/aenet_det/study_fpn/faster_fpn/randomseed    --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_pafpn  configs/detection/aenet/exp_fpn/aenet_faster_pafpn_1x_urpc2020.py      work_dirs/work_dirs/aenet_det/study_fpn/faster_pafpn/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_fpn    configs/detection/aenet/exp_fpn/aenet_retina_fpn_1x_urpc2020.py        work_dirs/work_dirs/aenet_det/study_fpn/retina_fpn/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_ufpn   configs/detection/aenet/exp_fpn/aenet_retina_pafpn_1x_urpc2020.py      work_dirs/work_dirs/aenet_det/study_fpn/retina_pafpn/randomseed    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_aug    configs/detection/aenet/exp_cycle/aenet_faster_ufpn_1x_urpc2020.py      work_dirs/work_dirs/aenet_det/study_cycle/faster/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_aug    configs/detection/aenet/exp_cycle/aenet_retina_ufpn_1x_urpc2020.py      work_dirs/work_dirs/aenet_det/study_cycle/retina/randomseed    --cfg-options randomness.seed=None

SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4   ./tools/slurm_train.sh mm_det faster_aug    configs/detection/aenet/exp_dataaug/aenet_faster_ufpn_1x_urpc2020.py               work_dirs/work_dirs/aenet_det/study_aug/faster/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4   ./tools/slurm_train.sh mm_det faster_aug    configs/detection/aenet/exp_dataaug/aenet_retina_ufpn_1x_urpc2020.py               work_dirs/work_dirs/aenet_det/study_aug/retina/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=7531  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_aug    configs/detection/aenet/exp_dataaug/dataaug_faster_rcnn_r50_ufpn_1x_urpc2020.py        work_dirs/work_dirs/aenet_det/study_aug/faster_base/randomseed  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_aug    configs/detection/aenet/exp_dataaug/dataaug_retinanet_r50_ufpn_1x_urpc2020.py      work_dirs/work_dirs/aenet_det/study_aug/retina_base/randomseed    --cfg-options randomness.seed=None


SRUN_ARGS='--quotatype=auto' MASTER_PORT=17884 GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_atss    configs/detection/aenet/urpc/aenet_atss_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/atss/randomseed_4    --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=3970  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_faster  configs/detection/aenet/urpc/aenet_faster_ufpn_1x_urpc2020.py       work_dirs/work_dirs/aenet_det/urpc/faster/randomseed_4  --cfg-options randomness.seed=None
SRUN_ARGS='--quotatype=auto' MASTER_PORT=6031  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc_tood    configs/detection/aenet/urpc/aenet_tood_ufpn_1x_urpc2020.py         work_dirs/work_dirs/aenet_det/urpc/tood/randomseed_4    --cfg-options randomness.seed=None
