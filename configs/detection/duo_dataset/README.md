# Detecting Underwater Objects

> [Detecting Underwater Objects](https://arxiv.org/abs/2106.05681)

<!-- [DATASET] -->

Underwater object detection for robot picking has attracted a lot of interest. However, it is still an unsolved problem due to several challenges. We take steps towards making it more realistic by addressing the following challenges. Firstly, the currently available datasets basically lack the test set annotations, causing researchers must compare their method with other SOTAs on a self-divided test set (from the training set). Training other methods lead to an increase in workload and different researchers divide different datasets, resulting there is no unified benchmark to compare the performance of different algorithms. Secondly, these datasets also have other shortcomings, e.g., too many similar images or incomplete labels. Towards these challenges we introduce a dataset, Detecting Underwater Objects (DUO), and a corresponding benchmark, based on the collection and re-annotation of all relevant datasets. DUO contains a collection of diverse underwater images with more rational annotations. The corresponding benchmark provides indicators of both efficiency and accuracy of SOTAs (under the MMDtection framework) for academic research and industrial applications, where JETSON AGX XAVIER is used to assess detector speed to simulate the robot-embedded environment.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/48282753/233964524-73b49b46-03c2-48ba-9786-697c9d2c081a.png" height="400"/>
</div>

## Results

Coming soon

## Citation

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
