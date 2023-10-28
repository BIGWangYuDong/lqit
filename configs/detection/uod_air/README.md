# Underwater Object Detection Aided by Image Reconstruction

> [Underwater Object Detection Aided by Image Reconstruction](https://ieeexplore.ieee.org/abstract/document/9949063)

<!-- [ALGORITHM] -->

## Abstract

Underwater object detection plays an important role in a variety of marine applications. However, the complexity of the underwater environment (e.g. complex background) and the quality degradation problems (e.g. color deviation) significantly affect the performance of the deep learning-based detector. Many previous works tried to improve the underwater image quality by overcoming the degradation of underwater or designing stronger network structures to enhance the detector feature extraction ability to achieve a higher performance in underwater object detection. However, the former usually inhibits the performance of underwater object detection while the latter does not consider the gap between open-air and underwater domains. This paper presents a novel framework to combine underwater object detection with image reconstruction through a shared backbone and Feature Pyramid Network (FPN). The loss between the reconstructed image and the original image in the reconstruction task is used to make the shared structure have better generalization capability and adaptability to the underwater domain, which can improve the performance of underwater object detection. Moreover, to combine different level features more effectively, UNet-based FPN (UFPN) is proposed to integrate better semantic and texture information obtained from deep and shallow layers, respectively. Extensive experiments and comprehensive evaluation on the URPC2020 dataset show that our approach can lead to 1.4% mAP and 1.0% mAP absolute improvements on RetinaNet and Faster R-CNN baseline with negligible extra overhead.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/BIGWangYuDong/lqit/assets/48282753/9a959a8c-e11e-4586-920f-dc86130accc4"/>
</div>

## Results

|              Architecture              | Neck | Lr schd |  lr  | box AP |                             Config                             |                                                                                                                                                 Download                                                                                                                                                 |
| :------------------------------------: | :--: | :-----: | :--: | :----: | :------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|              Faster R-CNN              | FPN  |   1x    | 0.02 |  43.5  | [config](./base_detector/faster-rcnn_r50_fpn_1x_urpc-coco.py)  |              [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r50_fpn_1x_urpc-coco_20220226_105840-09ef8403.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r50_fpn_1x_urpc-coco_20220226_105840.log.json)              |
|              Faster R-CNN              | UFPN |   1x    | 0.02 |  44.0  | [config](./base_detector/faster-rcnn_r50_ufpn_1x_urpc-coco.py) |             [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/faster-rcnn_r50_ufpn_1x_urpc-coco_20231027_211425-61d901bb.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/faster-rcnn_r50_ufpn_1x_urpc-coco_20231027_211425.log.json)             |
| Faster R-CNN with Image Reconstruction | UFPN |   1x    | 0.02 |  44.3  |    [config](./uod-air_faster-rcnn_r50_ufpn_1x_urpc-coco.py)    |     [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/uod-air_faster-rcnn_r50_ufpn_1x_urpc-coco_20231027_145407-6ae6d373.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/uod-air_faster-rcnn_r50_ufpn_1x_urpc-coco_20231027_145407.log.json)     |
|               RetinaNet                | FPN  |   1x    | 0.01 |  40.7  |  [config](./base_detector/retinanet_r50_fpn_1x_urpc-coco.py)   |                [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/retinanet_r50_fpn_1x_urpc-coco_20220405_214951-a39f054e.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/retinanet_r50_fpn_1x_urpc-coco_20220405_214951.log.json)                |
|               RetinaNet                | UFPN |   1x    | 0.01 |  41.8  |  [config](./base_detector/retinanet_r50_ufpn_1x_urpc-coco.py)  |               [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/retinanet_r50_ufpn_1x_urpc-coco_20231027_215756-7803a5f9.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/retinanet_r50_ufpn_1x_urpc-coco_20231027_215756.log.json)               |
|  RetinaNet with Image Reconstruction   | UFPN |   1x    | 0.01 |  42.3  |     [config](./uod-air_retinanet_r50_ufpn_1x_urpc-coco.py)     |       [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/uod-air_retinanet_r50_ufpn_1x_urpc-coco_20231027_224724-fe3acfba.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/uod-air_retinanet_r50_ufpn_1x_urpc-coco_20231027_224724.log.json)       |
|  RetinaNet with Image Reconstruction   | UFPN |   1x    | 0.02 |  43.3  |     [config](./uod-air_retinanet_r50_ufpn_1x_urpc-coco.py)     | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/uod-air_retinanet_r50_ufpn_1x_urpc-coco_lr002_20231027_215752-b727baaf.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc2/uod-air_retinanet_r50_ufpn_1x_urpc-coco_lr002_20231027_215752.log.json) |

**Note:** The original paper was developed based on MMDetection 2.0. LQIT optimized the network structure. LQIT has aligned the AP results on Faster R-CNN, but got 0.1 AP fluctuation on RetinaNet.

## Citation

```latex
@inproceedings{wang2022underwater,
  title={Underwater Object Detection Aided by Image Reconstruction},
  author={Wang, Yudong and Guo, Jichang and He, Wanru},
  booktitle={2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2022},
  organization={IEEE}
}
```
