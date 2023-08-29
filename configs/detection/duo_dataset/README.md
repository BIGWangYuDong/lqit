# Detecting Underwater Objects

> [Detecting Underwater Objects](https://arxiv.org/abs/2106.05681)

<!-- [DATASET] -->

Underwater object detection for robot picking has attracted a lot of interest. However, it is still an unsolved problem due to several challenges. We take steps towards making it more realistic by addressing the following challenges. Firstly, the currently available datasets basically lack the test set annotations, causing researchers must compare their method with other SOTAs on a self-divided test set (from the training set). Training other methods lead to an increase in workload and different researchers divide different datasets, resulting there is no unified benchmark to compare the performance of different algorithms. Secondly, these datasets also have other shortcomings, e.g., too many similar images or incomplete labels. Towards these challenges we introduce a dataset, Detecting Underwater Objects (DUO), and a corresponding benchmark, based on the collection and re-annotation of all relevant datasets. DUO contains a collection of diverse underwater images with more rational annotations. The corresponding benchmark provides indicators of both efficiency and accuracy of SOTAs (under the MMDtection framework) for academic research and industrial applications, where JETSON AGX XAVIER is used to assess detector speed to simulate the robot-embedded environment.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/48282753/233964524-73b49b46-03c2-48ba-9786-697c9d2c081a.png" height="400"/>
</div>

**Note:** DUO contains URPC2020, the categories of both datasets are same. DUO introduced URPC2020 and other underwater object detection datasets in the paper.

**TODO:**

- [ ] Support DUO Dataset and release models.
- [ ] Unify Dataset name in `LQIT`

## Results and Models

### URPC2020

| Architecture  |  Backbone   |  Style  | Lr schd | box AP |                         Config                         |                                                                                                                                           Download                                                                                                                                           |
| :-----------: | :---------: | :-----: | :-----: | :----: | :----------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN  |    R-50     | pytorch |   1x    |  43.5  |    [config](./faster-rcnn_r50_fpn_1x_urpc-coco.py)     |        [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r50_fpn_1x_urpc-coco_20220226_105840-09ef8403.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r50_fpn_1x_urpc-coco_20220226_105840.log.json)        |
| Faster R-CNN  |    R-101    | pytorch |   1x    |  44.8  |    [config](./faster-rcnn_r101_fpn_1x_urpc-coco.py)    |       [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r101_fpn_1x_urpc-coco_20220227_182523-de4a666c.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_r101_fpn_1x_urpc-coco_20220227_182523.log.json)       |
| Faster R-CNN  | X-101-32x4d | pytorch |   1x    |  44.6  | [config](./faster-rcnn_x101-32x4d_fpn_1x_urpc-coco.py) | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_x101-32x4d_fpn_1x_urpc-coco_20230511_190905-7074a9f7.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_x101-32x4d_fpn_1x_urpc-coco_20230511_190905.log.json) |
| Faster R-CNN  | X-101-64x4d | pytorch |   1x    |  45.3  | [config](./faster-rcnn_x101-64x4d_fpn_1x_urpc-coco.py) | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_x101-64x4d_fpn_1x_urpc-coco_20220405_193758-5d2a37e4.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/faster-rcnn_x101-64x4d_fpn_1x_urpc-coco_20220405_193758.log.json) |
| Cascade R-CNN |    R-50     | pytorch |   1x    |  44.3  |    [config](./cascade-rcnn_r50_fpn_1x_urpc-coco.py)    |       [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/cascade-rcnn_r50_fpn_1x_urpc-coco_20220405_160342-044e6858.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/cascade-rcnn_r50_fpn_1x_urpc-coco_20220405_160342.log.json)       |
|   RetinaNet   |    R-50     | pytorch |   1x    |  40.7  |     [config](./retinanet_r50_fpn_1x_urpc-coco.py)      |          [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/retinanet_r50_fpn_1x_urpc-coco_20220405_214951-a39f054e.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/retinanet_r50_fpn_1x_urpc-coco_20220405_214951.log.json)          |
|     FCOS      |    R-50     |  caffe  |   1x    |  41.4  | [config](./fcos_r50-caffe_fpn_gn-head_1x_urpc-coco.py) | [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/fcos_r50-caffe_fpn_gn-head_1x_urpc-coco_20220227_204555-305ab6aa.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/fcos_r50-caffe_fpn_gn-head_1x_urpc-coco_20220227_204555.log.json) |
|     ATSS      |    R-50     | pytorch |   1x    |  44.8  |        [config](./atss_r50_fpn_1x_urpc-coco.py)        |               [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/atss_r50_fpn_1x_urpc-coco_20220405_160345-cf776917.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/atss_r50_fpn_1x_urpc-coco_20220405_160345.log.json)               |
|     TOOD      |    R-50     | pytorch |   1x    |  45.4  |        [config](./tood_r50_fpn_1x_urpc-coco.py)        |               [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/tood_r50_fpn_1x_urpc-coco_20220405_164450-1fbf815b.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/tood_r50_fpn_1x_urpc-coco_20220405_164450.log.json)               |
|    SSD300     |    VGG16    |    -    |  120e   |  35.1  |          [config](./ssd300_120e_urpc-coco.py)          |                   [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/ssd300_120e_urpc-coco_20230426_122625-b6f0b01e.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/ssd512_120e_urpc-coco_20220405_185511.log.json)                   |
|    SSD512     |    VGG16    |    -    |  120e   |  38.6  |          [config](./ssd300_120e_urpc-coco.py)          |                   [model](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/ssd512_120e_urpc-coco_20220405_185511-88c18764.pth) \| [log](https://github.com/BIGWangYuDong/lqit/releases/download/v0.0.1rc1/ssd512_120e_urpc-coco_20220405_185511.log.json)                   |

### DUO

Coming soon

## Citation

- If you use `URPC2020` or other `URPC` series dataset in your research, please cite it as below:

  **Note:** The URL may not be valid, but this link is cited by many papers.

  ```latex
  @online{urpc,
  title = {Underwater Robot Professional Contest},
  url = {http://uodac.pcl.ac.cn/},
  }
  ```

- If you use `DUO` dataset in your research, please cite it as below:

  ```latex
  @inproceedings{liu2021dataset,
    title={A dataset and benchmark of underwater object detection for robot picking},
    author={Liu, Chongwei and Li, Haojie and Wang, Shuchang and Zhu, Ming and Wang, Dong and Fan, Xin and Wang, Zhihui},
    booktitle={2021 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
    pages={1--6},
    year={2021},
    organization={IEEE}
  }
  ```
