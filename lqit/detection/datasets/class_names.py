from mmengine.utils import is_str


def coco_classes() -> list:
    """COCO Detection Dataset Classes."""
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]


def cityscapes_classes() -> list:
    """Cityscapes Dataset Classes."""
    return [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def voc_classes() -> list:
    """Pascal VOC Dataset Classes."""
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def urpc_classes() -> list:
    """Underwater Robot Professional Contest Dataset Classes."""
    return ['holothurian', 'echinus', 'scallop', 'starfish']


def rtts_classes() -> list:
    """Foggy Object Detection Dataset Classes."""
    return ['bicycle', 'bus', 'car', 'motorbike', 'person']


dataset_aliases = {
    'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'],
    'coco': ['coco', 'mscoco', 'ms_coco'],
    'cityscapes': ['cityscapes'],
    'urpc': ['urpc', 'urpcdet', 'urpc2020'],
    'rtts': ['rtts', 'foggydet']
}


def get_classes(dataset_name: str) -> list:
    """Get class names of a dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        list: A list of dataset classes.
    """
    assert is_str(dataset_name), \
        f'dataset_name must a str, but got {type(dataset_name)}'
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if dataset_name in alias2name:
        class_names = eval(alias2name[dataset_name] + '_classes()')
    else:
        raise ValueError(f'Unrecognized dataset: {dataset_name}')

    return class_names
