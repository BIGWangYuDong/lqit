# GET STARTED

## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

LQIT works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.6+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name lqit python=3.8 -y
conda activate lqit
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## Installation

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMEval](https://github.com/open-mmlab/mmeval/tree/main/mmeval) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc1"
mim install mmeval
```

**Step 1.** Install [MMDetection](https://github.com/open-mmlab/mmdetection) from source.

```shell
# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git -b 3.x
# "-b 3.x" means checkout to the `3.x` branch.
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**NOTE:** Due to the MMDetection 3.0 is not fully ready, it is recommended to install the relevant dependencies from source. If MMDetection 3.0 is stable and has an official tag, users can using [MIM](https://github.com/open-mmlab/mim) to install [MMDetection](https://github.com/open-mmlab/mmdetection).

**Step. 2** Install LQIT from source

```shell
pip install -v -e .
```

Moreover, it is suggested to install pre-commit to develop the code:

```shell
pip install -U pre-commit

# from the repository folder
pre-commit install

# check all files lint
pre-commit run --all-files
```

If you want to install from source, you can refer to the following codes:

```shell
# install mmengine
git clone git@github.com:open-mmlab/mmengine.git
cd mmengine && pip install -e . && cd ../

# install mmcv
git clone git@github.com:open-mmlab/mmcv.git -b 2.x
cd mmcv
pip install -r requirements/optional.txt
pip install -e .

# install mmeval
git clone git@github.com:open-mmlab/mmeval.git
cd mmeval && pip install -e . && cd ../

# install mmdet
git clone git@github.com:open-mmlab/mmdetection.git -b 3.x
cd mmdetection && pip install -e . && cd ../

git clone git@github.com:BIGWangYuDong/lqit.git
cd lqit
pip install -e .
```
