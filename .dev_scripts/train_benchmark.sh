PARTITION=$1
WORK_DIR=$2
CPUS_PER_TASK=${3:-4}

echo 'configs/detection/fft_filter_experiment1/baseline_faster.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION example-1 configs/detection/fft_filter_experiment1/baseline_faster.py $WORK_DIR/fft_filter_experiment1/baseline_faster --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
