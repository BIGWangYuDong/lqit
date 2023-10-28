# TIENet: Task-oriented Image Enhancement Network for degraded object detection

> [TIENet: Task-oriented Image Enhancement Network for degraded object detection](https://link.springer.com/article/10.1007/s11760-023-02695-9)

<!-- [ALGORITHM] -->

## Abstract

Degraded images often suffer from low contrast, color deviations, and blurring details, which significantly affect the performance of detectors. Many previous works have attempted to obtain high-quality images based on human perception using image enhancement algorithms. However, these enhancement algorithms usually suppress the performance of degraded object detection. In this paper, we propose a taskoriented image enhancement network (TIENet) to directly improve degraded object detection’s performance by enhancing the degraded images. Unlike common human perception-based image-to-image methods, TIENet is a zero-reference enhancement network, which obtains a detectionfavorable structure image that is added to the original degraded image. In addition, this paper presents a fast Fourier transform-based structure loss for the enhancement task. With the new loss, our TIENet enables the structure image obtained to enhance more useful detection-favorable structural information and suppress irrelevant information. Extensive experiments and comprehensive evaluations on underwater (URPC2020) and foggy (RTTS) datasets show that our proposed framework can achieve 0.5–1.6% AP absolute improvements on classic detectors, including Faster R-CNN, RetinaNet, FCOS, ATSS, PAA, and TOOD. Besides, our method also generalizes well to the PASCAL VOC dataset, which can achieve 0.2–0.7% gains. We expect this study can draw more attention to high-level task-oriented degraded image enhancement.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/BIGWangYuDong/lqit/assets/48282753/c007e4d8-5aeb-439b-9adc-9530af8d421d"/>
</div>

## Results

### URPC2020

|     Architecture      | Lr schd | box AP |                            Config                             |                                                                                                                                           Download                                                                                                                                           |
| :-------------------: | :-----: | :----: | :-----------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     Faster R-CNN      |   1x    |  43.5  | [config](./base_detector/faster-rcnn_r50_fpn_1x_urpc-coco.py) |        [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r50_fpn_1x_urpc-coco_20220226_105840-09ef8403.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r50_fpn_1x_urpc-coco_20220226_105840.log.json)        |
| Faster R-CNN + TIENet |   1x    |  44.3  |    [config](./tienet_faster-rcnn_r50_fpn_1x_urpc-coco.py)     | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_faster-rcnn_r50_fpn_1x_urpc-coco_20221121_003439-0eb8ea32.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_faster-rcnn_r50_fpn_1x_urpc-coco_20221121_003439.log.json) |
|       RetinaNet       |   1x    |  40.7  |  [config](./base_detector/retinanet_r50_fpn_1x_urpc-coco.py)  |          [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/retinanet_r50_fpn_1x_urpc-coco_20220405_214951-a39f054e.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/retinanet_r50_fpn_1x_urpc-coco_20220405_214951.log.json)          |
|  RetinaNet + TIENet   |   1x    |  42.2  |     [config](./tienet_retinanet_r50_fpn_1x_urpc-coco.py)      |   [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_retinanet_r50_fpn_1x_urpc-coco_20221119_190211-2d1f311c.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_retinanet_r50_fpn_1x_urpc-coco_20221119_190211.log.json)   |
|         ATSS          |   1x    |  44.8  |    [config](./base_detector/atss_r50_fpn_1x_urpc-coco.py)     |               [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/atss_r50_fpn_1x_urpc-coco_20220405_160345-cf776917.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/atss_r50_fpn_1x_urpc-coco_20220405_160345.log.json)               |
|     ATSS + TIENet     |   1x    |  45.9  |        [config](./tienet_atss_r50_fpn_1x_urpc-coco.py)        |        [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_atss_r50_fpn_1x_urpc-coco_20230209_181359-473de7c1.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_atss_r50_fpn_1x_urpc-coco_20230209_181359.log.json)        |
|         TOOD          |   1x    |  45.4  |    [config](./base_detector/tood_r50_fpn_1x_urpc-coco.py)     |               [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/tood_r50_fpn_1x_urpc-coco_20220405_164450-1fbf815b.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/tood_r50_fpn_1x_urpc-coco_20220405_164450.log.json)               |
|     TOOD + TIENet     |   1x    |  46.7  |        [config](./tienet_tood_r50_fpn_1x_urpc-coco.py)        |        [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_tood_r50_fpn_1x_urpc-coco_20221119_212831-5dc036d5.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_tood_r50_fpn_1x_urpc-coco_20221119_212831.log.json)        |

### RTTS

|     Architecture      | Lr schd | box AP |                            Config                             |                                                                                                                                                        Download                                                                                                                                                        |
| :-------------------: | :-----: | :----: | :-----------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     Faster R-CNN      |   1x    |  48.1  | [config](./base_detector/faster-rcnn_r50_fpn_1x_rtts-coco.py) | [model](https://github.com/BIGWangYuDong/lqit/releases/download/untagged-cd79fcb6ab215a0cf240/faster-rcnn_r50_fpn_1x_rtts-coco_20231023_211050-81f577b7.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/untagged-cd79fcb6ab215a0cf240/faster-rcnn_r50_fpn_1x_rtts-coco_20231023_211050.log.json) |
| Faster R-CNN + TIENet |   1x    |  49.2  |    [config](./tienet_faster-rcnn_r50_fpn_1x_rtts-coco.py)     |              [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_faster-rcnn_r50_fpn_1x_rtts-coco_20221120_215748-50af5920.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_faster-rcnn_r50_fpn_1x_rtts-coco_20221120_215748.log.json)              |
|       RetinaNet       |   1x    |  33.7  |  [config](./base_detector/retinanet_r50_fpn_1x_rtts-coco.py)  |   [model](https://github.com/BIGWangYuDong/lqit/releases/download/untagged-cd79fcb6ab215a0cf240/retinanet_r50_fpn_1x_rtts-coco_20231023_211252-594f407a.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/untagged-cd79fcb6ab215a0cf240/retinanet_r50_fpn_1x_rtts-coco_20231023_211252.log.json)   |
|  RetinaNet + TIENet   |   1x    |  34.1  |     [config](./tienet_retinanet_r50_fpn_1x_rtts-coco.py)      |                [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_retinanet_r50_fpn_1x_rtts-coco_20221204_213217-b43e333d.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_retinanet_r50_fpn_1x_rtts-coco_20221204_213217.log.json)                |
|         ATSS          |   1x    |  48.2  |    [config](./base_detector/atss_r50_fpn_1x_rtts-coco.py)     |        [model](https://github.com/BIGWangYuDong/lqit/releases/download/untagged-cd79fcb6ab215a0cf240/atss_r50_fpn_1x_rtts-coco_20231023_210916-98b5356b.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/untagged-cd79fcb6ab215a0cf240/atss_r50_fpn_1x_rtts-coco_20231023_210916.log.json)        |
|     ATSS + TIENet     |   1x    |  49.5  |        [config](./tienet_atss_r50_fpn_1x_rtts-coco.py)        |                     [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_atss_r50_fpn_1x_rtrs-coco_20221120_105748-ec573a04.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_atss_r50_fpn_1x_rtrs-coco_20221120_105748.log.json)                     |
|         TOOD          |   1x    |  50.8  |    [config](./base_detector/tood_r50_fpn_1x_rtts-coco.py)     |        [model](https://github.com/BIGWangYuDong/lqit/releases/download/untagged-cd79fcb6ab215a0cf240/tood_r50_fpn_1x_rtts-coco_20231023_211348-6339a1f6.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/untagged-cd79fcb6ab215a0cf240/tood_r50_fpn_1x_rtts-coco_20231023_211348.log.json)        |
|     TOOD + TIENet     |   1x    |  52.1  |        [config](./tienet_tood_r50_fpn_1x_rtts-coco.py)        |                     [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_tood_r50_fpn_1x_rtts-coco_20221119_230205-e028a3bb.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/tienet_tood_r50_fpn_1x_rtts-coco_20221119_230205.log.json)                     |

## Citation

```latex
@article{wang2023tienet,
  title={{TIENet}: task-oriented image enhancement network for degraded object detection},
  author={Wang, Yudong and Guo, Jichang and Wang, Ruining and He, Wanru and Li, Chongyi},
  journal={Signal, Image and Video Processing},
  pages={1--8},
  year={2023},
  publisher={Springer}
}
```
