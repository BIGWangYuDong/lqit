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

The dataset contains 4,322 foggy images, covering five categories: bicycle, bus, car, motorbike, and person.

## Download RTTS Dataset

The Real-word Task-driven Testing Set (RTTS) dataset is a part of RESIDE dataset, which can be downloaded from [here](<>).

We randomly divides the RTTS dataset into training and testing groups with 3,457 and 865 images, respectively.
If users want to divide by their own, `tools/misc/write_txt.py` should be used to split the train and val set first.
Then `tools/dataset_converters/xml_to_json.py` can use to convert xml style annotations to coco format.

The data structure is as follows:

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
