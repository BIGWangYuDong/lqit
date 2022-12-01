# 低质图像开源工具箱

<div align="center">

[English](/README.md) | 简体中文

</div>

## 简介

LQIT 是一个低质图像开源工具箱，包括低质（水下、雾天、低照度等）图像增强和相关高层应用任务。
LQIT 基于 [PyTorch](https://pytorch.org/) 和 [OpenMMLab 2.0 系列](https://github.com/open-mmlab) 。

主分支代码目前支持 PyTorch 1.6 以上的版本。早期 PyTorch 版本的兼容性尚未经过全面的测试。

## 安装与准备数据集

LQIT 依赖于 [PyTorch](https://pytorch.org/), [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMEval](https://github.com/open-mmlab/mmeval) 。
它也可以把 [OpenMMLab](https://github.com/open-mmlab) 相关代码库作为三方依赖，例如 [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master) 。

请参考[安装文档](docs/en/get_started.md)进行安装和参考[数据准备](data/README.md)准备数据集。

## 贡献指南

我们感谢所有的贡献者为改进和提升 LQIT 所作出的努力。请参考[贡献指南](CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 联系

有任何问题可以通过 yudongwang1226@gmail.com 或者 yudongwang@tju.edu.cn 进行联系和讨论。
