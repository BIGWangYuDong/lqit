# Is Underwater Image Enhancement All Object Detectors Need?

> [Is Underwater Image Enhancement All Object Detectors Need?](https://arxiv.org/abs/2311.18814)

<!-- [ALGORITHM] -->

## Abstract

Underwater object detection is a crucial and challenging problem in marine engineering and aquatic robotics. The difficulty is partly because of the degradation of underwater images caused by light selective absorption and scattering. Intuitively, enhancing underwater images can benefit high-level applications like underwater object detection. However, it is still unclear whether all object detectors need underwater image enhancement as preprocessing. We therefore pose the questions “Does underwater image enhancement really improve underwater object detection?” and “How does underwater image enhancement contribute to underwater object detection?” . With these two questions, we conduct extensive studies. Specifically, we use 18 state-of-the-art underwater image enhancement algorithms, covering traditional, CNN-based, and GAN-based algorithms, to preprocess underwater object detection data. Then, we retrain seven popular deep learning-based object detectors using the corresponding results enhanced by different algorithms, obtaining 126 underwater object detection models. Coupled with seven object detection models retrained using raw underwater images, we employ these 133 models to comprehensively analyze the effect of underwater image enhancement on underwater object detection. We expect this study can provide sufficient exploration to answer the aforementioned questions and draw more attention of the community to the joint problem of underwater image enhancement and underwater object detection. The pretrained models and results are publicly available and will be regularly updated.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/user-attachments/assets/433bb5ff-b5fa-4833-bb58-b9c020bd8f4f"/>
</div>

## Results and Analysis

### URPC2020

Base configs can be found at [configs/detection/duo_dataset](../duo_dataset/).

### RUOD

Base configs can be found at [configs/detection/ruod_dataset](../ruod_dataset/).

## Usage

Different enhanced results are placed in different folders under the same root directory.
The data structure is as follows:

```text
lqit
├── lqit
├── tools
├── configs
├── data
│   ├── URPC
│   │   ├── ImageSets
│   │   ├── ImageMetas          # get image meta information from scripts
│   │   ├── annotations_json    # coco style annotations
│   │   ├── JPEGImages          # Raw images
│   │   ├── UIEC2Net            # UIEC^2Net enhanced result folder
│   │   ├── UColor              # UColor enhanced result folder
│   │   ├── ...                 # different enhanced result folder
│   ├── RUOD
│   │   ├── annotations         # annotations
│   │   ├── train               # Raw training images
│   │   ├── test                # Raw testing images
│   │   ├── UIEC2Net            # UIEC^2Net enhanced folder
│   │   │   ├── train
│   │   │   ├── test
│   │   ├── ...
```

Coming soon:

- [ ] Different enhancement scripts
- [ ] Release models

## Citation

```latex
@article{10376393,
  author={Wang, Yudong and Guo, Jichang and He, Wanru and Gao, Huan and Yue, Huihui and Zhang, Zenan and Li, Chongyi},
  journal={IEEE Journal of Oceanic Engineering},
  title={Is Underwater Image Enhancement All Object Detectors Need?},
  year={2024},
  volume={49},
  number={2},
  pages={606-621},
  keywords={Image enhancement;Object detection;Detectors;Image color analysis;Task analysis;Visualization;Underwater tracking;Autonomous underwater vehicles;Aquatic robots;Image degradation;joint task;underwater image enhancement;underwater object detection},
  doi={10.1109/JOE.2023.3302888}
}
```
