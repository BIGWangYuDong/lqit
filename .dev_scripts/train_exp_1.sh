
# experiment 1 of rtts
MASTER_PORT=1235  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts1 configs/detection/self_enhance_detector/rtts/faster_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/rtts/faster/base_loss   --cfg-options dist_params.port=1666 &

MASTER_PORT=1238  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts2 configs/detection/self_enhance_detector/rtts/faster_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/rtts/faster/base_loss_detach --cfg-options dist_params.port=1666 &

MASTER_PORT=2331  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts3 configs/detection/self_enhance_detector/rtts/faster_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/faster/no_loss --cfg-options dist_params.port=1666 &


# experiment 1 of urpc
MASTER_PORT=2355  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc1 configs/detection/self_enhance_detector/urpc/faster_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/urpc/faster/base_loss   --cfg-options dist_params.port=1666 &

MASTER_PORT=2578  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc2 configs/detection/self_enhance_detector/urpc/faster_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/urpc/faster/base_loss_detach --cfg-options dist_params.port=1666 &

MASTER_PORT=2781  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc3 configs/detection/self_enhance_detector/urpc/faster_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/urpc/faster/no_loss --cfg-options dist_params.port=1666 &


# experiment 1 of rtts
MASTER_PORT=3163  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts4 configs/detection/self_enhance_detector/rtts/retinanet_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/rtts/retinanet/base_loss   --cfg-options dist_params.port=1666 &

MASTER_PORT=3658  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts5 configs/detection/self_enhance_detector/rtts/retinanet_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/rtts/retinanet/base_loss_detach --cfg-options dist_params.port=1666 &

MASTER_PORT=4287  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det rtts6 configs/detection/self_enhance_detector/rtts/retinanet_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/rtts/retinanet/no_loss --cfg-options dist_params.port=1666 &


# experiment 1 of urpc
MASTER_PORT=4318  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc4 configs/detection/self_enhance_detector/urpc/retinanet_r50_base_loss.py                   work_dirs/work_dirs/self_enhance_light/urpc/retinanet/base_loss   --cfg-options dist_params.port=1666 &

MASTER_PORT=4617  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc5 configs/detection/self_enhance_detector/urpc/retinanet_r50_base_loss_detach.py      work_dirs/work_dirs/self_enhance_light/urpc/retinanet/base_loss_detach --cfg-options dist_params.port=1666 &

MASTER_PORT=8183  GPUS=4  GPUS_PER_NODE=4  ./tools/slurm_train.sh mm_det urpc6 configs/detection/self_enhance_detector/urpc/retinanet_r50_no_loss.py      work_dirs/work_dirs/self_enhance_light/urpc/retinanet/no_loss --cfg-options dist_params.port=1666 &
