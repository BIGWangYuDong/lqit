# 开始你的第一步

## 依赖

本节中，我们将演示如何用 PyTorch 准备一个环境。

LQIT 支持在 Linux，Windows 和 macOS 上运行。它需要 Python 3.6 以上，CUDA 9.2 以上和 PyTorch 1.6 以上.

```{note}
如果你对 PyTorch 有经验并且已经安装了它，你可以直接跳转到[下一小节](##安装流程)。否则，你可以按照下述步骤进行准备
```

**步骤 0.** 从 [官方网站](https://docs.conda.io/en/latest/miniconda.html) 下载并安装 Miniconda。

**步骤 1.** 创建并激活一个 conda 环境。

```shell
conda create --name lqit python=3.8 -y
conda activate lqit
```

**步骤 2.** 基于 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装 PyTorch。

在 GPU 平台上：

```shell
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## 安装流程

我们建议用户按照我们的最佳实践来安装 LQIT。当然，整个过程是高度可定制的。详细信息请参阅[自定义安装](#customize-installation)部分。

### 最佳实践

**步骤 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine), [MMEval](https://github.com/open-mmlab/mmeval/tree/main/mmeval) 和 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmeval
```

**步骤 1.** 通过源码安装 LQIT

```shell
git clone https://github.com/BIGWangYuDong/lqit.git
cd lqit
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

**注意：**

(1) 按照上述说明， LQIT 安装在 `dev` 模式下，因此在本地对代码做的任何修改都会生效，无需重新安装；

(2) 如果希望使用 `opencv-python-headless` 而不是 `opencv-python`， 可以在安装 MMEngine 之前安装；

(3) 一些安装依赖是可以选择的。例如只需要安装最低运行要求的版本，则可以使用 `pip install -v -e .` 命令。如果希望使用可选择的像 `albumentations` 和 `imagecorruptions` 这种依赖项，可以使用 `pip install -r requirements/optional.txt` 进行手动安装，或者在使用 `pip` 时指定所需的附加功能（例如 `pip install -v -e .[optional]`），支持附加功能的有效键值包括 `all`、`tests`、`build` 以及 `optional`。

(4) 如果希望使用 `albumentations`，我们建议使用 `pip install -r requirements/albu.txt` 或者 `pip install -U albumentations --no-binary qudida,albumentations` 进行安装。 如果简单地使用 `pip install albumentations>=0.3.2` 进行安装，则会同时安装 `opencv-python-headless`（即便已经安装了 `opencv-python` 也会再次安装）。我们建议在安装 `albumentations` 后检查环境，以确保没有同时安装 `opencv-python` 和 `opencv-python-headless`，因为同时安装可能会导致一些问题。更多细节请参考[官方文档](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) 。

**步骤 1.** 选择安装必要的 [OpenMMLab](https://github.com/open-mmlab) 代码库作为依赖，并通过 [MIM](https://github.com/open-mmlab/mim) 进行安装。例如安装 [MMDetection](https://github.com/open-mmlab/mmdetection):

```shell
mim install "mmdet>=3.0.0"
```

### 验证安装

为了验证 LQIT 是否安装正确，我们提供了一些示例代码来执行模型推理。

`TODO later`

**Step 1.**

### 自定义安装

在安装 PyTorch 时，你需要指定 CUDA 的版本。如果你不清楚应该选择哪一个，请遵循我们的建议。

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 系列 以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向后兼容 (backward compatible) 的，但 CUDA 10.2 能够提供更好的兼容性，也更加轻量。

请确保你的 GPU 驱动版本满足最低的版本需求，参阅 NVIDIA 官方的 [CUDA工具箱和相应的驱动版本关系表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) 。

```{note}
如果按照我们的最佳实践进行安装，CUDA 运行时库就足够了，因为我们提供相关 CUDA 代码的预编译，不需要进行本地编译。
但如果你希望从源码进行 MMCV 的编译，或是进行其他 CUDA 算子的开发，那么就必须安装完整的 CUDA 工具链，参见
[NVIDIA 官网](https://developer.nvidia.com/cuda-downloads) ，另外还需要确保该 CUDA 工具链的版本与 PyTorch 安装时
的配置相匹配（如用 `conda install` 安装 PyTorch 时指定的 cudatoolkit 版本）。
```

#### 不使用 MIM 安装 MMEngine

要使用 pip 而不是 MIM 来安装 MMEngine，请遵照 [MMEngine 安装指南](https://mmengine.readthedocs.io/en/latest/get_started/installation.html) 。

例如，你可以通过以下命令安装 MMEngine

```shell
pip install mmengine
```

#### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此其对 PyTorch 的依赖比较复杂。MIM 会自动解析这些
依赖，选择合适的 MMCV 预编译包，使安装更简单，但它并不是必需的。

要使用 pip 而不是 MIM 来安装 MMCV，请遵照 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html) 。
它需要您用指定 url 的形式手动指定对应的 PyTorch 和 CUDA 版本。

例如，下述命令将会安装基于 PyTorch 1.12.x 和 CUDA 11.6 编译的 mmcv。

```shell
pip install "mmcv>=2.0.0" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

#### 不使用 MIM 安装 MMEval

要使用 pip 而不是 MIM 来安装 MMEval，请遵照 [MMEval 安装指南](https://mmeval.readthedocs.io/en/latest/get_started/installation.html) 。

例如，你可以通过以下命令安装 MMEval：

```shell
pip install mmeval
```

#### 不使用 MIM 安装其他 OpenMMLab 代码库

OpenMMLab 代码库提供了详细的安装教程，你可以通过相关的安装指南进行安装。

例如，你可以通过 [MMDetection 安装指南](https://mmdetection.readthedocs.io/en/3.x/get_started.html)并通过涂璇命令来安装 MMDetection：

```shell
pip install "mmdet>=3.0.0"
```

## Contributing to LQIT

我们期待大家能够为 LQIT 提供更多的贡献。有关贡献指南的更多信息请参阅 [CONTRIBUTING.md](../../CONTRIBUTING.md)。
