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
