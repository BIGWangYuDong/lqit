# URPC2020

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

该数据集包含 14,000 张水下图像（其中包含 9,800 张训练图像和 4,200 张测试图像），超过 74,000 个标注框，涵盖十类：鱼（fish）、海胆（echinus）、珊瑚（corals）、海星（starfish）、海参（holothurian）、扇贝（scallop）、潜水员（diver）、墨鱼（cuttlefish）、乌龟（turtle）和水母（jellyfish）

## 下载 RUOD 数据

真实水下目标检测（Real-world Underwater Object Detection, RUOD）数据集可从[此处](https://github.com/dlut-dimt/RUOD)下载。

数据存放结构默认如下：

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
