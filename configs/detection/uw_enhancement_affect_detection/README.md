# Is Underwater Image Enhancement All Object Detectors Need?

<!-- [ALGORITHM] -->

## Abstract

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

Coming soon
