# 低质图像开源工具箱

<div align="center">
  <img src="resources/lqit-logo.jpg" width="600"/>
</div>

<div align="center">

[English](README.md) | 简体中文

</div>

## 简介

LQIT 是一个低质图像开源工具箱，包括低质（水下、雾天、低照度等）图像增强和相关高层应用任务。
LQIT 基于 [PyTorch](https://pytorch.org/) 和 [OpenMMLab 2.0 系列](https://github.com/open-mmlab) 。

主分支代码目前支持 PyTorch 1.6 以上的版本。早期 PyTorch 版本的兼容性尚未经过全面的测试。

## 更新

**v0.0.1rc2** 版本已经在 2023.10.28 发布：

- 支持了[飞书机器人](configs/lark/README.md)
- 支持了 [TIENet](https://link.springer.com/article/10.1007/s11760-023-02695-9)、[UOD-AIR](https://ieeexplore.ieee.org/abstract/document/9949063) 和 [RDFFNet](https://link.springer.com/article/10.1007/s11760-022-02410-0)
- 发布了雾天目标检测 `RTTS` 数据集的模型权重

可通过查阅[更新日志](docs/en/notes/changelog.md)了解更多细节以及发布历史。

## 安装与准备数据集

LQIT 依赖于 [PyTorch](https://pytorch.org/), [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMEval](https://github.com/open-mmlab/mmeval) 。
它也可以把 [OpenMMLab](https://github.com/open-mmlab) 相关代码库作为三方依赖，例如 [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master) 。

请参考[安装文档](docs/zh_cn/get_started.md)进行安装和参考[数据准备](data/README_zh-CN.md)准备数据集。

## 贡献指南

我们感谢所有的贡献者为改进和提升 LQIT 所作出的努力。请参考[贡献指南](CONTRIBUTING_zh-CN.md)来了解参与项目贡献的相关指引。

## 开源许可证

`LQIT` 采用 [Apache 2.0 开源许可证](LICENSE)，但是其中有一部分功能并不是使用的 Apache2.0 许可证，我们在 [许可证](LICENSES.md) 中详细地列出了这些功能以及他们对应的许可证，如果您正在从事盈利性活动，请谨慎参考此文档。

## 联系

有任何问题可以通过 yudongwang1226@gmail.com 或者 yudongwang@tju.edu.cn 进行联系和讨论。
