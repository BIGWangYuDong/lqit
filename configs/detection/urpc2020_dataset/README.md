# Underwater Robot Professional Contest 2020

> [Underwater Robot Professional Contest 2020](https://www.heywhale.com/home/competition/5e535a612537a0002ca864ac/content/0)
>
> [Datasets at OpenI](https://openi.pcl.ac.cn/OpenOrcinus_orca/URPC_opticalimage_dataset/datasets)

<!-- [DATASET] -->

The Object Detection Algorithm Competition is the first phase of the National Underwater Robotics (Zhanjiang) Competition jointly organized by the National Natural Science Foundation of China, Pengcheng Laboratory, and the People's Government of Zhanjiang. This competition focuses on the field of underwater object detection algorithms and innovatively combines artificial intelligence with underwater robots. It opens up optical and acoustic images of the real underwater environment to a wider community of artificial intelligence and algorithm researchers, establishing a new domain for object detection and recognition. The competition is divided into two categories: "Optical Image Object Detection" and "Acoustic Image Object Detection." This project has collected relevant datasets for the "Optical Image Object Detection" category, with the hope that these data will be of assistance to researchers in related fields.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/BIGWangYuDong/lqit/assets/48282753/7faecf69-4172-4614-97d1-5362e7c368ce" height="400"/>
</div>

## Results

### Validation-set Results

| Architecture  |  Backbone   |  Style  | Lr schd | box AP |                                 Config                                  |                                                                                                                                           Download                                                                                                                                           |
| :-----------: | :---------: | :-----: | :-----: | :----: | :---------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN  |    R-50     | pytorch |   1x    |  43.5  |    [config](./train_validation/faster-rcnn_r50_fpn_1x_urpc-coco.py)     |        [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r50_fpn_1x_urpc-coco_20220226_105840-09ef8403.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r50_fpn_1x_urpc-coco_20220226_105840.log.json)        |
| Faster R-CNN  |    R-101    | pytorch |   1x    |  44.8  |    [config](./train_validation/faster-rcnn_r101_fpn_1x_urpc-coco.py)    |       [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r101_fpn_1x_urpc-coco_20220227_182523-de4a666c.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r101_fpn_1x_urpc-coco_20220227_182523.log.json)       |
| Faster R-CNN  | X-101-32x4d | pytorch |   1x    |  44.6  | [config](./train_validation/faster-rcnn_x101-32x4d_fpn_1x_urpc-coco.py) | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_x101-32x4d_fpn_1x_urpc-coco_20230511_190905-7074a9f7.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_x101-32x4d_fpn_1x_urpc-coco_20230511_190905.log.json) |
| Faster R-CNN  | X-101-64x4d | pytorch |   1x    |  45.3  | [config](./train_validation/faster-rcnn_x101-64x4d_fpn_1x_urpc-coco.py) | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_x101-64x4d_fpn_1x_urpc-coco_20220405_193758-5d2a37e4.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_x101-64x4d_fpn_1x_urpc-coco_20220405_193758.log.json) |
| Cascade R-CNN |    R-50     | pytorch |   1x    |  44.3  |    [config](./train_validation/cascade-rcnn_r50_fpn_1x_urpc-coco.py)    |       [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/cascade-rcnn_r50_fpn_1x_urpc-coco_20220405_160342-044e6858.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/cascade-rcnn_r50_fpn_1x_urpc-coco_20220405_160342.log.json)       |
|   RetinaNet   |    R-50     | pytorch |   1x    |  40.7  |     [config](./train_validation/retinanet_r50_fpn_1x_urpc-coco.py)      |          [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/retinanet_r50_fpn_1x_urpc-coco_20220405_214951-a39f054e.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/retinanet_r50_fpn_1x_urpc-coco_20220405_214951.log.json)          |
|     FCOS      |    R-50     |  caffe  |   1x    |  41.4  | [config](./train_validation/fcos_r50-caffe_fpn_gn-head_1x_urpc-coco.py) | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/fcos_r50-caffe_fpn_gn-head_1x_urpc-coco_20220227_204555-305ab6aa.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/fcos_r50-caffe_fpn_gn-head_1x_urpc-coco_20220227_204555.log.json) |
|     ATSS      |    R-50     | pytorch |   1x    |  44.8  |        [config](./train_validation/atss_r50_fpn_1x_urpc-coco.py)        |               [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/atss_r50_fpn_1x_urpc-coco_20220405_160345-cf776917.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/atss_r50_fpn_1x_urpc-coco_20220405_160345.log.json)               |
|     TOOD      |    R-50     | pytorch |   1x    |  45.4  |        [config](./train_validation/tood_r50_fpn_1x_urpc-coco.py)        |               [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/tood_r50_fpn_1x_urpc-coco_20220405_164450-1fbf815b.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/tood_r50_fpn_1x_urpc-coco_20220405_164450.log.json)               |
|    SSD300     |    VGG16    |    -    |  120e   |  35.1  |          [config](./train_validation/ssd300_120e_urpc-coco.py)          |                   [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/ssd300_120e_urpc-coco_20230426_122625-b6f0b01e.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/ssd512_120e_urpc-coco_20220405_185511.log.json)                   |
|    SSD512     |    VGG16    |    -    |  120e   |  38.6  |          [config](./train_validation/ssd300_120e_urpc-coco.py)          |                   [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/ssd512_120e_urpc-coco_20220405_185511-88c18764.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/ssd512_120e_urpc-coco_20220405_185511.log.json)                   |

## Test-set Results

Coming soon

## Citation

**Note:** The URL may not be valid, but this link is cited by many papers.

```latex
@online{urpc,
title = {Underwater Robot Professional Contest},
url = {http://uodac.pcl.ac.cn/},
}
```
