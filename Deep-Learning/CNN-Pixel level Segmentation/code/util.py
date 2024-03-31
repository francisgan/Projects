import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def iou(pred, target, n_classes = 21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
        float: Mean IoU across all classes.
    """
    pred = pred.permute(0,2,3,1)
    pred = torch.max(pred, 3)[1]
    iou_list = []
    for i in range(n_classes):
        TP = torch.sum((pred == i) & (target == i)).item()
        FP = torch.sum((pred == i) & (target != i)).item()
        FN = torch.sum((pred != i) & (target == i)).item()
        denominator = TP + FP + FN
        if denominator > 0:
            iou_list.append(TP / denominator)

    return np.mean(iou_list) if iou_list else 0

def pixel_acc(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
        float: Pixel-wise accuracy.
    """
    pred = pred.permute(0,2,3,1)
    pred = torch.max(pred, 3)[1]
    return (pred==target).float().mean().cpu()