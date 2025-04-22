import torch
import numpy as np

def generate_boxes_and_scores(bbox_pred, class_pred, image_size=(512, 512), conf_threshold=0.1, iou_threshold=0.5):
    boxes = []
    scores = []
    B, C, H, W = class_pred.shape
    img_h, img_w = image_size

    for b in range(B):
        for i in range(H):
            for j in range(W):
                score = torch.softmax(class_pred[b, :, i, j], dim=0)
                label = torch.argmax(score)
                conf = score[label].item()

                if conf > conf_threshold and label != 0:
                    x, y, w, h = bbox_pred[b, :, i, j]
                    x_min = ((j + x.item()) / W) * img_w
                    y_min = ((i + y.item()) / H) * img_h
                    x_max = ((j + w.item()) / W) * img_w
                    y_max = ((i + h.item()) / H) * img_h
                    x_min = max(0, min(x_min, img_w))
                    y_min = max(0, min(y_min, img_h))
                    x_max = max(0, min(x_max, img_w))
                    y_max = max(0, min(y_max, img_h))
                    boxes.append([x_min, y_min, x_max, y_max])
                    scores.append((label.item(), conf))

    if len(boxes) > 0:
        boxes, scores = non_maximum_suppression(torch.tensor(boxes), torch.tensor(scores), iou_threshold)
    return boxes, scores

def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return [], []
    keep = []
    scores = scores[:, 1]
    indices = scores.argsort(descending=True)

    while len(indices) > 0:
        current = indices[0]
        keep.append(current.item())
        if len(indices) == 1:
            break
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[indices[1:]]
        iou = compute_iou(current_box, remaining_boxes)
        indices = indices[1:][iou < iou_threshold]
    return boxes[keep].tolist(), scores[keep].tolist()

def compute_iou(box1, box2):
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = box1_area + box2_area - intersection
    iou = intersection / union
    return iou