# Swin Transformer for Stenosis Detection

This repository contains an implementation of a Swin Transformer-based model for object detection, specifically designed for stenosis detection on the ARCADE dataset. The model is built using PyTorch and processes images to predict bounding boxes and class labels for stenosis-related objects. The model was developed on a CPU with a limited batch size (2) and a subset of 1000 images due to computational constraints. With GPU availability, the batch size and number of images can be increased for improved training performance.

## Table of Contents

- Project Overview
- Dataset
- Installation
- Usage
- File Structure
- Training
- Evaluation
- Contributing
- License

## Project Overview

The Swin Transformer model is adapted for object detection to identify and localize stenosis in medical images from the ARCADE dataset. The model uses a hierarchical architecture with shifted window-based multi-head self-attention (MSA) and patch merging to efficiently process high-resolution images. It outputs bounding box predictions and class probabilities for 27 classes.

## Dataset

The ARCADE dataset contains medical images with annotations for stenosis detection in COCO format, divided into two subsets: `stenosis` and `syntax`. Each subset includes training, validation, and test splits, with images resized to 512x512 pixels. Annotations include bounding boxes and category IDs.

- **Location**: Place the dataset in a directory (e.g., `./arcade/`).

- **Structure**:

  ```
  arcade/
  ├── stenosis/
  │   ├── test/
  │   │   ├── images/
  │   │   └── annotations/
  │   │       └── test.json
  │   ├── train/
  │   │   ├── images/
  │   │   └── annotations/
  │   │       └── train.json
  │   └── val/
  │       ├── images/
  │       └── annotations/
  │           └── val.json
  ├── syntax/
  │   ├── test/
  │   │   ├── images/
  │   │   └── annotations/
  │   │       └── test.json
  │   ├── train/
  │   │   ├── images/
  │   │   └── annotations/
  │   │       └── train.json
  │   └── val/
  │       ├── images/
  │       └── annotations/
  │           └── val.json
  ```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/swin-stenosis-detection.git
   cd swin-stenosis-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Dataset**:

   - Download the ARCADE dataset and place it in the `./arcade/` directory.
   - Ensure the `train.json`, `test.json`, and `val.json` annotations are in the respective `annotations/` subdirectories for each subset (`stenosis` or `syntax`).

2. **Train the Model**:

   - Run the training script, specifying the dataset subset (e.g., `./arcade/stenosis` or `./arcade/syntax`):

     ```bash
     python train.py --data_dir ./arcade/stenosis --num_epochs 10 --batch_size 2
     ```

   - Note: The batch size and number of images were limited during development due to CPU constraints. On a GPU, you can increase `--batch_size` (e.g., to 8 or 16) and `--num_samples` in the dataset for better performance.

3. **Visualize Predictions**:

   - Use the visualization script to see predictions on test images:

     ```bash
     python visualize.py --data_dir ./arcade/stenosis --model_path ./model.pth --num_images 5
     ```

## File Structure

- `model.py`: Defines the Swin Transformer model, including `SwinEmbedding`, `PatchMerging`, `ShiftedWindowMSA`, and other components.
- `dataset.py`: Implements the `ARCADE_Dataset` class for loading the ARCADE dataset and `collate_fn` for batching.
- `utils.py`: Contains utility functions for generating bounding boxes, non-maximum suppression (NMS), and IoU computation.
- `train.py`: Implements the training loop and detection loss function.
- `visualize.py`: Contains the function to visualize model predictions.
- `requirements.txt`: Lists Python dependencies.
- `README.md`: Project documentation.

## Training

The model is trained using the `train.py` script, which:

- Loads the ARCADE dataset using `ARCADE_Dataset`.
- Uses a custom `DetectionLoss` combining classification and regression losses.
- Optimizes the model with Adam and a learning rate of 1e-4.
- Saves the trained model weights.

Example command:

```bash
python train.py --data_dir ./arcade/stenosis --num_epochs 10 --batch_size 2
```

Note: Training was performed on a CPU with a batch size of 2 and 1000 images. With GPU support, you can increase the batch size and dataset size for better results.

## Evaluation

Evaluation is performed by visualizing predictions on the test set using `visualize.py`. The model outputs bounding boxes and class labels, which are compared against ground truth annotations.

To evaluate:

```bash
python visualize.py --data_dir ./arcade/stenosis --model_path ./model.pth --num_images 5
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.