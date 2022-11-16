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

MASTER_PORT=2331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_loss_detach configs/detection/rtts/base_detector/faster_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/faster/no_loss --cfg-options dist_params.port=1666


# experiment 1 of urpc
MASTER_PORT=1235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_base_loss configs/detection/self_enhance_detector/urpc/faster_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/urpc/faster/base_loss   --cfg-options dist_params.port=1666

MASTER_PORT=1238 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_no_loss configs/detection/self_enhance_detector/urpc/faster_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/urpc/faster/base_loss_detach --cfg-options dist_params.port=1666

MASTER_PORT=2331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_loss_detach configs/detection/self_enhance_detector/urpc/faster_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/urpc/no_loss --cfg-options dist_params.port=1666


# experiment 1 of rtts
MASTER_PORT=1235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_base_loss configs/detection/self_enhance_detector/rtts/retinanet_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/rtts/retinanet/base_loss   --cfg-options dist_params.port=1666

MASTER_PORT=1238 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_no_loss configs/detection/self_enhance_detector/rtts/retinanet_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/rtts/retinanet/base_loss_detach --cfg-options dist_params.port=1666

MASTER_PORT=2331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_loss_detach configs/detection/rtts/base_detector/retinanet_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/retinanet/no_loss --cfg-options dist_params.port=1666


# experiment 1 of urpc
MASTER_PORT=1235 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_base_loss configs/detection/self_enhance_detector/urpc/retinanet_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/urpc/retinanet/base_loss   --cfg-options dist_params.port=1666

MASTER_PORT=1238 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_no_loss configs/detection/self_enhance_detector/urpc/retinanet_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/urpc/retinanet/base_loss_detach --cfg-options dist_params.port=1666

MASTER_PORT=2331 SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det retina_loss_detach configs/detection/self_enhance_detector/urpc/retinanet_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/retinanet/no_loss --cfg-options dist_params.port=1666
