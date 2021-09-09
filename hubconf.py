"""File for accessing YOLOv5 models via PyTorch Hub https://pytorch.org/hub/ultralytics_yolov5/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
"""

dependencies = ['torch', 'torchvision', 'yaml', 'scipy']

import os.path

import torch


def _create_yolov5(name, pretrained, channels, classes):
    """Creates a YOLOv5 model.

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        torch.nn.Module: pytorch model
    """
    from models.yolo import Model

    config = os.path.join(os.path.dirname(__file__), 'models', f'{name}.yaml')
    model = Model(config, channels, classes)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            f'https://github.com/ultralytics/yolov5/releases/download/v5.0/{name}.pt',
            map_location=torch.device('cpu'),
        )
        msd = model.state_dict()  # model state_dict
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
        model.load_state_dict(csd, strict=False)  # load
        if len(ckpt['model'].names) == classes:
            model.names = ckpt['model'].names  # set class names attribute
    return model


def yolov5s(pretrained=True, channels=3, classes=80):
    # YOLOv5-small model https://github.com/ultralytics/yolov5
    return _create_yolov5('yolov5s', pretrained, channels, classes)


def yolov5m(pretrained=True, channels=3, classes=80):
    # YOLOv5-medium model https://github.com/ultralytics/yolov5
    return _create_yolov5('yolov5m', pretrained, channels, classes)


def yolov5l(pretrained=True, channels=3, classes=80):
    # YOLOv5-large model https://github.com/ultralytics/yolov5
    return _create_yolov5('yolov5l', pretrained, channels, classes)


def yolov5x(pretrained=True, channels=3, classes=80):
    # YOLOv5-xlarge model https://github.com/ultralytics/yolov5
    return _create_yolov5('yolov5x', pretrained, channels, classes)


def yolov5s6(pretrained=True, channels=3, classes=80):
    # YOLOv5-small model https://github.com/ultralytics/yolov5
    return _create_yolov5('yolov5s6', pretrained, channels, classes)


def yolov5m6(pretrained=True, channels=3, classes=80):
    # YOLOv5-medium model https://github.com/ultralytics/yolov5
    return _create_yolov5('yolov5m6', pretrained, channels, classes)


def yolov5l6(pretrained=True, channels=3, classes=80):
    # YOLOv5-large model https://github.com/ultralytics/yolov5
    return _create_yolov5('yolov5l6', pretrained, channels, classes)


def yolov5x6(pretrained=True, channels=3, classes=80):
    # YOLOv5-xlarge model https://github.com/ultralytics/yolov5
    return _create_yolov5('yolov5x6', pretrained, channels, classes)
