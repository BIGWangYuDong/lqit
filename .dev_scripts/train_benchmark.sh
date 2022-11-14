PARTITION=$1
WORK_DIR=$2
CPUS_PER_TASK=${3:-4}

echo 'configs/detection/detector_with_enhance_head/faster-rcnn_r50_fpn_basic-enhance_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION example-1 configs/detection/detector_with_enhance_head/faster-rcnn_r50_fpn_basic-enhance_1x_coco.py $WORK_DIR/detector_with_enhance_head/faster_r50_1x_basic_enhance_head --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &


SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_base configs/detection/self_enhance_detector/self_enhance_light/faster_r50_base_loss.py work_dirs/work_dirs/self_enhance_light/faster_base_loss --cfg-options dist_parrams.port=1666

SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_base configs/detection/self_enhance_detector/self_enhance_light/faster_r50_with_allloss.py work_dirs/work_dirs/self_enhance_light/faster_all_loss --cfg-options dist_parrams.port=16666

SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_base configs/detection/self_enhance_detector/self_enhance_light/faster_r50_with_perc.py work_dirs/work_dirs/self_enhance_light/faster_with_perc --cfg-options dist_parrams.port=34788

SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_base configs/detection/self_enhance_detector/self_enhance_light/faster_r50_with_structure.py work_dirs/work_dirs/self_enhance_light/faster_with_structure --cfg-options dist_parrams.port=47852


SRUN_ARGS='--quotatype=auto' GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det faster_base configs/detection/self_enhance_detector/self_enhance_light/faster_r50_base_loss.py work_dirs/work_dirs/self_enhance_light/faster_base_loss --cfg-options dist_parrams.port=1666
