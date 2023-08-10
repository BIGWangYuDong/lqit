PARTITION=$1
WORK_DIR=$2
CPUS_PER_TASK=${3:-4}

echo 'configs/detection/detector_with_enhance_head/faster-rcnn_r50_fpn_basic-enhance_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION example-1 configs/detection/detector_with_enhance_head/faster-rcnn_r50_fpn_basic-enhance_1x_coco.py $WORK_DIR/detector_with_enhance_head/faster_r50_1x_basic_enhance_head --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &





./tools/slurm_train.sh llm subjective_qa   configs/detection/ruod_dataset/atss_r50_fpn_1x_ruod.py                      work_dirs/ruod_640/atss              --cfg-options default_hooks.checkpoint.max_keep_ckpts=1
./tools/slurm_train.sh llm subjective_qa   configs/detection/ruod_dataset/faster-rcnn_r50_fpn_1x_ruod.py               work_dirs/ruod_640/faster            --cfg-options default_hooks.checkpoint.max_keep_ckpts=1
./tools/slurm_train.sh llm subjective_qa   configs/detection/ruod_dataset/cascade-rcnn_r50_fpn_1x_ruod.py              work_dirs/ruod_640/cascade           --cfg-options default_hooks.checkpoint.max_keep_ckpts=1
./tools/slurm_train.sh llm subjective_qa   configs/detection/ruod_dataset/retinanet_r50_fpn_1x_ruod.py                 work_dirs/ruod_640/retinanet         --cfg-options default_hooks.checkpoint.max_keep_ckpts=1
./tools/slurm_train.sh llm subjective_qa   configs/detection/ruod_dataset/fcos_r50-caffe_fpn_gn-head_1x_ruod.py        work_dirs/ruod_640/fcos              --cfg-options default_hooks.checkpoint.max_keep_ckpts=1
./tools/slurm_train.sh llm subjective_qa   configs/detection/ruod_dataset/tood_r50_fpn_1x_ruod.py                      work_dirs/ruod_640/tood              --cfg-options default_hooks.checkpoint.max_keep_ckpts=1
./tools/slurm_train.sh llm subjective_qa   configs/detection/ruod_dataset/paa.py                                       work_dirs/ruod_640/paa              

