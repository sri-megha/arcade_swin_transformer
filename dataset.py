import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ARCADE_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(512, 512), num_samples=1000, mode="train"):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        if mode == "train":
            self.annotation_path = os.path.join(root_dir, "annotations", "train.json")
        elif mode == "test":
            self.annotation_path = os.path.join(root_dir, "annotations", "test.json")
        else:
            raise ValueError("Mode must be 'train' or 'test'")
        self.transform = transform
        self.image_size = image_size
        self.num_samples = num_samples

        with open(self.annotation_path, "r") as f:
            self.coco = json.load(f)

        self.image_id_map = {img["id"]: img["file_name"] for img in self.coco["images"]}
        self.annotations = {img_id: [] for img_id in self.image_id_map}

        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            x_min, y_min, w, h = ann["bbox"]
            x_max, y_max = x_min + w, y_min + h
            bbox = [x_min, y_min, x_max, y_max]
            label = ann["category_id"]
            self.annotations[img_id].append((bbox, label))

        self.img_ids = list(self.image_id_map.keys())[:self.num_samples]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_name = self.image_id_map[img_id]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.image_size)

        if self.annotations[img_id]:
            bboxes, labels = zip(*self.annotations[img_id])
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        if self.transform:
            img = self.transform(img)

        target = {"boxes": bboxes, "labels": labels}
        return img, target

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets