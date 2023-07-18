# Underwater Object Detection Aided by Image Reconstruction

> [Underwater Object Detection Aided by Image Reconstruction](https://ieeexplore.ieee.org/abstract/document/9949063)

<!-- [ALGORITHM] -->

## Abstract

Underwater object detection plays an important role in a variety of marine applications. However, the complexity of the underwater environment (e.g. complex background) and the quality degradation problems (e.g. color deviation) significantly affect the performance of the deep learning-based detector. Many previous works tried to improve the underwater image quality by overcoming the degradation of underwater or designing stronger network structures to enhance the detector feature extraction ability to achieve a higher performance in underwater object detection. However, the former usually inhibits the performance of underwater object detection while the latter does not consider the gap between open-air and underwater domains. This paper presents a novel framework to combine underwater object detection with image reconstruction through a shared backbone and Feature Pyramid Network (FPN). The loss between the reconstructed image and the original image in the reconstruction task is used to make the shared structure have better generalization capability and adaptability to the underwater domain, which can improve the performance of underwater object detection. Moreover, to combine different level features more effectively, UNet-based FPN (UFPN) is proposed to integrate better semantic and texture information obtained from deep and shallow layers, respectively. Extensive experiments and comprehensive evaluation on the URPC2020 dataset show that our approach can lead to 1.4% mAP and 1.0% mAP absolute improvements on RetinaNet and Faster R-CNN baseline with negligible extra overhead.

<div align=center>
<img src="https://github.com/BIGWangYuDong/lqit/assets/48282753/9a959a8c-e11e-4586-920f-dc86130accc4"/>
</div>

## Results and Analysis

Coming soon

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
