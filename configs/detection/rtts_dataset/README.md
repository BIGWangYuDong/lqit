# Benchmarking single-image dehazing and beyond

> [Benchmarking single-image dehazing and beyond](https://ieeexplore.ieee.org/abstract/document/8451944)

<!-- [DATASET] -->

We present a comprehensive study and evaluation of existing single-image dehazing algorithms, using a new large-scale benchmark consisting of both synthetic and real-world hazy images, called REalistic Single-Image DEhazing (RESIDE). RESIDE highlights diverse data sources and image contents, and is divided into five subsets, each serving different training or evaluation purposes. We further provide a rich variety of criteria for dehazing algorithm evaluation, ranging from full-reference metrics to no-reference metrics and to subjective evaluation, and the novel task-driven evaluation. Experiments on RESIDE shed light on the comparisons and limitations of the state-of-the-art dehazing algorithms, and suggest promising future directions.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/BIGWangYuDong/lqit/assets/48282753/263606fa-5c92-4c6c-baad-b42c0998c7d8" height="400"/>
</div>

## Results

| Architecture  | Backbone |  Style  | Lr schd | box AP |                         Config                         |                                                                                                                                          Download                                                                                                                                          |
| :-----------: | :------: | :-----: | :-----: | :----: | :----------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN  |   R-50   | pytorch |   1x    |  48.1  |    [config](./faster-rcnn_r50_fpn_1x_rtts-coco.py)     |        [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2faster-rcnn_r50_fpn_1x_rtts-coco_20231023_211050-81f577b7.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2faster-rcnn_r50_fpn_1x_rtts-coco_20231023_211050.log.json)        |
| Cascade R-CNN |   R-50   | pytorch |   1x    |  50.8  |    [config](./cascade-rcnn_r50_fpn_1x_rtts-coco.py)    |       [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2cascade-rcnn_r50_fpn_1x_rtts-coco_20231023_211029-ebfd7705.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2cascade-rcnn_r50_fpn_1x_rtts-coco_20231023_211029.log.json)       |
|   RetinaNet   |   R-50   | pytorch |   1x    |  33.7  |     [config](./retinanet_r50_fpn_1x_rtts-coco.py)      |          [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2retinanet_r50_fpn_1x_rtts-coco_20231023_211252-594f407a.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2retinanet_r50_fpn_1x_rtts-coco_20231023_211252.log.json)          |
|     FCOS      |   R-50   |  caffe  |   1x    |  41.0  | [config](./fcos_r50-caffe_fpn_gn-head_1x_rtts-coco.py) | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2fcos_r50-caffe_fpn_gn-head_1x_rtts-coco_20231023_211216-b7e2e105.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2fcos_r50-caffe_fpn_gn-head_1x_rtts-coco_20231023_211216.log.json) |
|     ATSS      |   R-50   | pytorch |   1x    |  48.2  |        [config](./atss_r50_fpn_1x_rtts-coco.py)        |               [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2atss_r50_fpn_1x_rtts-coco_20231023_210916-98b5356b.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2atss_r50_fpn_1x_rtts-coco_20231023_210916.log.json)               |
|     TOOD      |   R-50   | pytorch |   1x    |  50.8  |        [config](./tood_r50_fpn_1x_rtts-coco.py)        |               [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2tood_r50_fpn_1x_rtts-coco_20231023_211348-6339a1f6.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2tood_r50_fpn_1x_rtts-coco_20231023_211348.log.json)               |
|      PAA      |   R-50   | pytorch |   1x    |  49.3  |        [config](./paa_r50_fpn_1x_rtts-coco.py)         |                [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2paa_r50_fpn_1x_rtts-coco_20231024_001806-04ca4793.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2paa_r50_fpn_1x_rtts-coco_20231024_001806.log.json)                |

## Citation

```latex
@article{li2018benchmarking,
 title={Benchmarking single-image dehazing and beyond},
 author={Li, Boyi and Ren, Wenqi and Fu, Dengpan and Tao, Dacheng and Feng, Dan and Zeng, Wenjun and Wang, Zhangyang},
 journal={IEEE Transactions on Image Processing},
 volume={28},
 number={1},
 pages={492--505},
 year={2018},
 publisher={IEEE}
}
```
