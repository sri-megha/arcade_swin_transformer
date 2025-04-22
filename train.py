import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import Swin
from dataset import ARCADE_Dataset, collate_fn
import argparse

class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.SmoothL1Loss()

    def forward(self, bbox_pred, class_pred, targets):
        device = bbox_pred.device
        batch_size, _, H, W = class_pred.shape
        total_loss = 0.0

        for i in range(batch_size):
            boxes = targets[i]["boxes"].to(device)
            labels = targets[i]["labels"].to(device)

            if boxes.shape[0] == 0:
                class_loss = self.classification_loss(class_pred[i], torch.zeros(H * W, dtype=torch.long, device=device))
                total_loss += class_loss
                continue

            pred_boxes = bbox_pred[i].permute(1, 2, 0).reshape(-1, 4)
            pred_classes = class_pred[i].permute(1, 2, 0).reshape(-1, class_pred.shape[1])

            ious = torch.zeros((pred_boxes.shape[0], boxes.shape[0]), device=device)
            for j in range(boxes.shape[0]):
                ious[:, j] = compute_iou(pred_boxes, boxes[j:j+1])

            max_ious, max_indices = ious.max(dim=1)
            positive_mask = max_ious > 0.5
            if positive_mask.sum() == 0:
                class_loss = self.classification_loss(pred_classes, torch.zeros(H * W, dtype=torch.long, device=device))
                total_loss += class_loss
                continue

            assigned_labels = labels[max_indices[positive_mask]]
            assigned_boxes = boxes[max_indices[positive_mask]]

            class_loss = self.classification_loss(pred_classes[positive_mask], assigned_labels)
            regression_loss = self.regression_loss(pred_boxes[positive_mask], assigned_boxes)
            total_loss += class_loss + regression_loss

        return total_loss / batch_size

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

def train_model(data_dir, num_epochs=10, batch_size=2):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ARCADE_Dataset(
        root_dir=data_dir,
        transform=transform,
        num_samples=1000,
        mode="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = Swin(num_classes=27)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DetectionLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            bbox_pred, class_pred = model(images)
            loss = criterion(bbox_pred, class_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred_classes = torch.argmax(class_pred, dim=1)
            for i in range(images.shape[0]):
                if targets[i]["labels"].shape[0] > 0:
                    gt_labels = targets[i]["labels"].to(device)
                    pred_labels = pred_classes[i].flatten()
                    correct += (pred_labels == gt_labels).sum().item()
                    total += gt_labels.shape[0]

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin model for stenosis detection")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ARCADE dataset")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    args = parser.parse_args()
    train_model(args.data_dir, args.num_epochs, args.batch_size)