# GET STARTED

## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

LQIT works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.6+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](##installation). Otherwise, you can follow these steps for the preparation.
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

We recommend that users follow our best practices to install LQIT. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMEval](https://github.com/open-mmlab/mmeval/tree/main/mmeval) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc1"
mim install mmeval
```

**Step. 1** Install LQIT from source.

```shell
git clone https://github.com/BIGWangYuDong/lqit.git
cd lqit
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**Note:**

a. When specifying `-e` or `develop`, MMDetection is installed on dev mode, any local modifications made to the code will take effect without reinstallation.

b. If you would like to use `opencv-python-headless` instead of `opencv-python`, you can install it before installing MMCV.

c. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements.
To use optional dependencies like `albumentations` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`).
Valid keys for the extras field are: `all`, `tests`, `build`, `optional`, `det`, and `det_opt`.

d. If you would like to use `albumentations`, we suggest using `pip install -r requirements/albu.txt` or
`pip install -U albumentations --no-binary qudida,albumentations`. If you simply use `pip install albumentations>=0.3.2`,
it will install `opencv-python-headless` simultaneously (even though you have already
installed `opencv-python`). We recommended checking the environment after installing `albumentation` to
ensure that `opencv-python` and `opencv-python-headless` are not installed at the same time, because it might cause unexpected issues if they both are installed. Please refer
to [official documentation](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) for more details.

**Step 2.** Optional install [OpenMMLab codebases](https://github.com/open-mmlab) as a dependency. Install it with [MIM](https://github.com/open-mmlab/mim). e.g.

Installing [MMDetection](https://github.com/open-mmlab/mmdetection).

```shell
mim install "mmdet>=3.0.0rc0"
```

### Verify the installation

To verify whether LQIT is installed correctly, we provide some sample codes to run an inference demo.

`TODO later`

**Step 1.**

### Customize Installation

#### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However, if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in the `conda install` command.
```

#### Install MMEngine without MIM

To install MMEngine with pip instead of MIM, please follow [MMEngine installation guides](https://mmengine.readthedocs.io/en/latest/get_started/installation.html).

For example, you can install MMEngine by the following command.

```shell
pip install mmengine
```

#### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html). This requires manually specifying a find-url based on the PyTorch version and its CUDA version.

For example, the following command installs MMCV built for PyTorch 1.12.x and CUDA 11.6.

```shell
pip install "mmcv>=2.0.0rc1" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

#### Install MMEval without MIM

To install MMEval with pip instead of MIM, please follow [MMEval installation guides](https://mmeval.readthedocs.io/en/latest/get_started/installation.html).

For example, you can install MMEval by the following command.

```shell
pip install mmeval
```

#### Install OpenMMLab codebases without MIM

OpenMMLab Codebases provide a detailed installation tutorial, please follow relative installation guides.

For example, you can find [MMDetection installation guides](https://mmdetection.readthedocs.io/en/3.x/get_started.html), and install MMDetection by the following command.

```shell
pip install "mmdet>=3.0.0rc0"
```

## Contributing to LQIT

We appreciate all contributions to improve LQIT. Please refer to [CONTRIBUTING.md](../../CONTRIBUTING.md) for more details about the contributing guideline.
