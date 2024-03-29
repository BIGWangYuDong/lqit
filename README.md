# Low-Quality Image ToolBox

<div align="center">
  <img src="resources/lqit-logo.jpg" width="600"/>
</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

LQIT is an open source Low-Quality Image Toolbox, including low-quality (underwater, foggy, low-light, etc.) image enhancement tasks,
and related high-level computer vision tasks (such as object detection). LQIT depends on [PyTorch](https://pytorch.org/) and [OpenMMLab 2.0 series](https://github.com/open-mmlab).

The main branch works with **PyTorch 1.6+**.
The compatibility to earlier versions of PyTorch is not fully tested.

## What's New

**v0.0.1rc2** was released in 28/10/2023:

- Support [FeiShu (Lark) robot](configs/lark/README.md)
- Support [TIENet](https://link.springer.com/article/10.1007/s11760-023-02695-9), [UOD-AIR](https://ieeexplore.ieee.org/abstract/document/9949063), and [RDFFNet](https://link.springer.com/article/10.1007/s11760-022-02410-0)
- Release `RTTS` foggy object detection models

Please refer to [changelog](docs/en/notes/changelog.md) for details and release history.

## Installation & Dataset Preparation

LQIT depends on [PyTorch](https://pytorch.org/), [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEval](https://github.com/open-mmlab/mmeval).
It also can use [OpenMMLab codebases](https://github.com/open-mmlab) as a dependency, such as [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master).

Please refer to [Installation](docs/en/get_started.md) for installation of LQIT and [data preparation](data/README.md) for dataset preparation.

## Contributing

We appreciate all contributions to improve LQIT. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## License

LQIT is released under the [Apache 2.0 license](LICENSE), while some specific features in this library are with other licenses. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.

## Contact

If you have any questions, please contact Yudong Wang at yudongwang1226@gmail.com or yudongwang@tju.edu.cn.
