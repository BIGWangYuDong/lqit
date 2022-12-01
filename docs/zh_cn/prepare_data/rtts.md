# RTTS

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

该数据集包含 4,322 张雾天图像，包含五个类：自行车 (bicycle)、公共汽车 (bus)、汽车 (car)、摩托车 (motorbike)和人 (person)。

## 下载 RTTS 数据

数据集

真实任务驱动数据集 (Real-word Task-driven Testing Set, RTTS) 是 RESIDE 数据集的一部分，可以从 [这里](<>) 下载。

我们将 RTTS 数据集随机分为训练组和测试组，分别有 3,457 和 865 张图像。
如果用户想自己划分，应该先使用`tools/misc/write_txt.py`来划分train和val集合。
然后 `tools/dataset_converters/xml_to_json.py` 可以用来将 xml 样式的注释转换为 coco 格式。

数据存放结构默认如下：

```text
lqit
├── lqit
├── tools
├── configs
├── data
│   ├── RESIDE
│   │   ├── RTTS
│   │   │   ├── ImageSets
│   │   │   │   ├── Main
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── annotations_xml     # pascal voc style annotations
│   │   │   │   ├── AM_Bing_211.xml
│   │   │   │   ├── AM_Bing_217.xml
│   │   │   │   ├── ...
│   │   │   ├── annotations_json    # coco style annotations
│   │   │   │   ├── rtts_train.json
│   │   │   │   ├── rtts_val.json
│   │   │   ├── JPEGImages
│   │   │   │   ├── AM_Bing_211.png
│   │   │   │   ├── AM_Bing_217.png
│   │   │   │   ├── ...
```
