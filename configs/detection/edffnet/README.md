# Edge-Guided Dynamic Feature Fusion Network for Object Detection under Foggy Conditions

> [Edge-Guided Dynamic Feature Fusion Network for Object Detection under Foggy Conditions](https://link.springer.com/article/10.1007/s11760-022-02410-0)

<!-- [ALGORITHM] -->

## Abstract

Hazy images are often subject to blurring, low contrast and other visible quality degradation, making it challenging to solve object detection tasks. Most methods solve the domain shift problem by deep domain adaptive technology, ignoring the inaccurate object classification and localization caused by quality degradation. Different from common methods, we present an edge-guided dynamic feature fusion network (EDFFNet), which formulates the edge head as a guide to the localization task. Despite the edge head being straightforward, we demonstrate that it makes the model pay attention to the edge of object instances and improves the generalization and localization ability of the network. Considering the fuzzy details and the multi-scale problem of hazy images, we propose a dynamic fusion feature pyramid network (DF-FPN) to enhance the feature representation ability of the whole model. A unique advantage of DF-FPN is that the contribution to the fused feature map will dynamically adjust with the learning of the network. Extensive experiments verify that EDFFNet achieves 2.4% AP and 3.6% AP gains over the ATSS baseline on RTTS and Foggy Cityscapes, respectively.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/BIGWangYuDong/lqit/assets/48282753/82087e24-4ef6-40b4-ae95-a5893e293c1e"/>
</div>

## Results on RTTS

| Architecture | Neck  | Lr schd | Edge Head |  lr  | box AP |                          Config                          |                                                                                                                                             Download                                                                                                                                             |
| :----------: | :---: | :-----: | :-------: | :--: | :----: | :------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     ATSS     |  FPN  |   1x    |     -     | 0.01 |  48.2  |  [config](../rtts_dataset/atss_r50_fpn_1x_rtts-coco.py)  |                 [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/atss_r50_fpn_1x_rtts-coco_20231023_210916-98b5356b.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/atss_r50_fpn_1x_rtts-coco_20231023_210916.log.json)                 |
|     ATSS     |  FPN  |   1x    |     -     | 0.02 |  49.6  |      [config](./atss_r50_fpn_1x_rtts-coco_lr002.py)      |           [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/atss_r50_fpn_1x_rtts-coco_lr002_20231028_104029-114517ae.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/atss_r50_fpn_1x_rtts-coco_lr002_20231028_104029.log.json)           |
|     ATSS     | DFFPN |   1x    |     -     | 0.02 |  50.3  |     [config](./atss_r50_dffpn_1x_rtts-coco_lr002.py)     |         [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/atss_r50_dffpn_1x_rtts-coco_lr002_20231028_104638-8f22abd9.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/atss_r50_dffpn_1x_rtts-coco_lr002_20231028_104638.log.json)         |
|     ATSS     | DFFPN |   1x    |     Y     | 0.02 |  50.8  | [config](./edffnet_atss_r50_dffpn_1x_rtts-coco_lr002.py) | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/edffnet_atss_r50_dffpn_1x_rtts-coco_lr002_20231028_111154-89311078.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/edffnet_atss_r50_dffpn_1x_rtts-coco_lr002_20231028_111154.log.json) |

## Citation

```latex
@article{he2023edge,
  title={Edge-guided dynamic feature fusion network for object detection under foggy conditions},
  author={He, Wanru and Guo, Jichang and Wang, Yudong and Zheng, Sida},
  journal={Signal, Image and Video Processing},
  volume={17},
  number={5},
  pages={1975--1983},
  year={2023},
  publisher={Springer}
}
```
