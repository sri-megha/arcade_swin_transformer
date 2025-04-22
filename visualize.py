import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Swin
from dataset import ARCADE_Dataset, collate_fn
from utils import generate_boxes_and_scores
import argparse

def visualize_predictions(model, data_loader, device, class_names, num_images=5):
    model.eval()
    images_processed = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            bbox_pred, class_pred = model(images)

            for i in range(images.shape[0]):
                if images_processed >= num_images:
                    break
                boxes, scores = generate_boxes_and_scores(bbox_pred[i:i+1], class_pred[i:i+1])

                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                ax = plt.gca()

                for box, (label, conf) in zip(boxes, scores):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = plt.Rectangle((x_min, y_min), width, height, fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x_min, y_min - 10, f"{class_names[label]}: {conf:.2f}", color='red', fontsize=12, weight='bold')

                plt.axis('off')
                plt.show()
                images_processed += 1

            if images_processed >= num_images:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Swin model predictions")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ARCADE dataset")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to trained model")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to visualize")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ARCADE_Dataset(
        root_dir=args.data_dir,
        transform=transform,
        num_samples=100,
        mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = Swin(num_classes=27)
    model.load_state_dict(torch.load(args.model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_names = [f"Class_{i}" for i in range(27)]
    visualize_predictions(model, test_loader, device, class_names, args.num_images)