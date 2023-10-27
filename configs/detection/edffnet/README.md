# Edge-Guided Dynamic Feature Fusion Network for Object Detection under Foggy Conditions

<!-- [ALGORITHM] -->

## Abstract

Hazy images are often subject to blurring, low contrast and other visible quality degradation, making it challenging to solve object detection tasks. Most methods solve the domain shift problem by deep domain adaptive technology, ignoring the inaccurate object classification and localization caused by quality degradation. Different from common methods, we present an edge-guided dynamic feature fusion network (EDFFNet), which formulates the edge head as a guide to the localization task. Despite the edge head being straightforward, we demonstrate that it makes the model pay attention to the edge of object instances and improves the generalization and localization ability of the network. Considering the fuzzy details and the multi-scale problem of hazy images, we propose a dynamic fusion feature pyramid network (DF-FPN) to enhance the feature representation ability of the whole model. A unique advantage of DF-FPN is that the contribution to the fused feature map will dynamically adjust with the learning of the network. Extensive experiments verify that EDFFNet achieves 2.4% AP and 3.6% AP gains over the ATSS baseline on RTTS and Foggy Cityscapes, respectively.

<div align=center>
<img src="https://github.com/BIGWangYuDong/lqit/assets/48282753/82087e24-4ef6-40b4-ae95-a5893e293c1e"/>
</div>

## Results and Analysis

Coming soon

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
