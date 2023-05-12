# RUOD

```latex
@article{fu2023rethinking,
  title={Rethinking general underwater object detection: Datasets, challenges, and solutions},
  author={Fu, Chenping and Liu, Risheng and Fan, Xin and Chen, Puyang and Fu, Hao and Yuan, Wanqi and Zhu, Ming and Luo, Zhongxuan},
  journal={Neurocomputing},
  volume={517},
  pages={243--256},
  year={2023},
  publisher={Elsevier}
}
```

The dataset contains 14,000 images (9,800 for training and 4,200 for testing) with more than 74,000 bounding boxes, covering ten categories: fish, echinus, corals, starfish, holothurian, scallop, diver, cuttlefish, turtle, and jellyfish.

## Download RUOD Dataset

The Real-world Underwater Object Detection (RUOD) dataset can be download from [here](https://github.com/dlut-dimt/RUOD) .

The data structure is as follows:

```text
lqit
├── lqit
├── tools
├── configs
├── data
│   ├── RUOD
│   │   ├── annotations
│   │   │   ├── instances_train.json
│   │   │   ├── instances_test.json
│   │   ├── train
│   │   │   ├── 000002.jpg
│   │   │   ├── 000003.jpg
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── 000001.jpg
│   │   │   ├── 000004.jpg
│   │   │   ├── ...
```
