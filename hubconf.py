"""File for accessing YOLOv5 models via PyTorch Hub.

Usage:
    import torch
    model = torch.hub.load('anibali/yolov5', 'yolov5s', pretrained=True)
"""

dependencies = ['torch', 'torchvision', 'yaml', 'scipy']

import os.path


model_urls = {
    'yolov5s': f'https://github.com/anibali/yolov5/releases/download/v5.0/yolov5s-6641d604.pth',
    'yolov5m': f'https://github.com/anibali/yolov5/releases/download/v5.0/yolov5m-6b4327bb.pth',
    'yolov5l': f'https://github.com/anibali/yolov5/releases/download/v5.0/yolov5l-3165254b.pth',
    'yolov5x': f'https://github.com/anibali/yolov5/releases/download/v5.0/yolov5x-82f8f4cf.pth',
    'yolov5s6': f'https://github.com/anibali/yolov5/releases/download/v5.0/yolov5s6-fd16cab0.pth',
    'yolov5m6': f'https://github.com/anibali/yolov5/releases/download/v5.0/yolov5m6-11226298.pth',
    'yolov5l6': f'https://github.com/anibali/yolov5/releases/download/v5.0/yolov5l6-34d51d89.pth',
    'yolov5x6': f'https://github.com/anibali/yolov5/releases/download/v5.0/yolov5x6-8a12fc4e.pth',
}


COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


def _create_yolov5(name, pretrained, progress, channels, classes):
    """Creates a YOLOv5 model.

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        progress (bool): show download progress
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    import torch
    from yolov5.models.yolo import Model

    config = os.path.join(os.path.dirname(__file__), 'yolov5', 'models', f'{name}.yaml')
    model = Model(config, channels, classes)
    if pretrained:
        if not (channels == 3 and classes == 80):
            raise ValueError('pretrained weights are for channels=3 and classes=80')
        state_dict = torch.hub.load_state_dict_from_url(model_urls[name], progress=progress)
        model.load_state_dict(state_dict)
        model.names = COCO_CLASS_NAMES
    return model


def yolov5s(pretrained=False, progress=True, channels=3, classes=80):
    """YOLOv5-small model from https://github.com/ultralytics/yolov5."""
    return _create_yolov5('yolov5s', pretrained, progress, channels, classes)


def yolov5m(pretrained=False, progress=True, channels=3, classes=80):
    """YOLOv5-medium model from https://github.com/ultralytics/yolov5."""
    return _create_yolov5('yolov5m', pretrained, progress, channels, classes)


def yolov5l(pretrained=False, progress=True, channels=3, classes=80):
    """YOLOv5-large model from https://github.com/ultralytics/yolov5."""
    return _create_yolov5('yolov5l', pretrained, progress, channels, classes)


def yolov5x(pretrained=False, progress=True, channels=3, classes=80):
    """YOLOv5-xlarge model from https://github.com/ultralytics/yolov5."""
    return _create_yolov5('yolov5x', pretrained, progress, channels, classes)


def yolov5s6(pretrained=False, progress=True, channels=3, classes=80):
    """YOLOv5-small model from https://github.com/ultralytics/yolov5."""
    return _create_yolov5('yolov5s6', pretrained, progress, channels, classes)


def yolov5m6(pretrained=False, progress=True, channels=3, classes=80):
    """YOLOv5-medium model from https://github.com/ultralytics/yolov5."""
    return _create_yolov5('yolov5m6', pretrained, progress, channels, classes)


def yolov5l6(pretrained=False, progress=True, channels=3, classes=80):
    """YOLOv5-large model from https://github.com/ultralytics/yolov5."""
    return _create_yolov5('yolov5l6', pretrained, progress, channels, classes)


def yolov5x6(pretrained=False, progress=True, channels=3, classes=80):
    """YOLOv5-xlarge model from https://github.com/ultralytics/yolov5."""
    return _create_yolov5('yolov5x6', pretrained, progress, channels, classes)
