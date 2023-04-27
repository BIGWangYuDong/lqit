# Detecting Underwater Objects

> [Detecting Underwater Objects](https://arxiv.org/abs/2106.05681)
> [Underwater Robot Professional Contest 2020](https://www.heywhale.com/home/competition/5e535a612537a0002ca864ac/content/0)

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

| Architecture  |  Backbone   |  Style  | Lr schd | box AP |                         Config                         |         Download         |
| :-----------: | :---------: | :-----: | :-----: | :----: | :----------------------------------------------------: | :----------------------: |
| Faster R-CNN  |    R-50     | pytorch |   1x    |  43.5  |    [config](./faster-rcnn_r50_fpn_1x_urpc-coco.py)     | [model](<>) \| [log](<>) |
| Faster R-CNN  |    R-101    | pytorch |   1x    |  44.8  |    [config](./faster-rcnn_r101_fpn_1x_urpc-coco.py)    | [model](<>) \| [log](<>) |
| Faster R-CNN  | X-101-32x4d | pytorch |   1x    |  44.6  | [config](./faster-rcnn_x101-32x4d_fpn_1x_urpc-coco.py) | [model](<>) \| [log](<>) |
| Faster R-CNN  | X-101-64x4d | pytorch |   1x    |  45.3  | [config](./faster-rcnn_x101-64x4d_fpn_1x_urpc-coco.py) | [model](<>) \| [log](<>) |
| Cascade R-CNN |    R-50     | pytorch |   1x    |  44.3  |    [config](./cascade-rcnn_r50_fpn_1x_urpc-coco.py)    | [model](<>) \| [log](<>) |
|   RetinaNet   |    R-50     | pytorch |   1x    |  40.7  |     [config](./retinanet_r50_fpn_1x_urpc-coco.py)      | [model](<>) \| [log](<>) |
|     FCOS      |    R-50     | cafffe  |   1x    |  41.4  | [config](./fcos_r50-caffe_fpn_gn-head_1x_urpc-coco.py) | [model](<>) \| [log](<>) |
|     ATSS      |    R-50     | pytorch |   1x    |  44.8  |        [config](./atss_r50_fpn_1x_urpc-coco.py)        | [model](<>) \| [log](<>) |
|     TOOD      |    R-50     | pytorch |   1x    |  45.4  |        [config](./tood_r50_fpn_1x_urpc-coco.py)        | [model](<>) \| [log](<>) |
|    SSD300     |    VGG16    |    -    |  120e   |  35.1  |          [config](./ssd300_120e_urpc-coco.py)          | [model](<>) \| [log](<>) |

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
