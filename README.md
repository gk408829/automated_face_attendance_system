# Face Recognition Attendance System

## Project Overview

This project implements an automated attendance tracking system using computer vision and deep learning. The system detects faces in classroom images, classifies gender, and identifies individuals to maintain accurate attendance records.

## Features

- **Face Detection**: Uses YOLOv8 to locate and extract faces from classroom photos
- **Gender Classification**: Determines gender of detected faces using a fine-tuned ResNet model
- **Face Identification**: Recognizes specific individuals using facial embeddings
- **Attendance Tracking**: Generates attendance records based on face recognition results
- **Data Simulation**: Creates synthetic classroom datasets for testing and training

## Architecture

The system consists of three main components:

1. **Face Detection Model**: YOLOv8-based detector trained to locate faces in images
2. **Gender Classification Model**: ResNet18-based binary classifier to determine gender
3. **Face Identification Model**: Deep neural network that generates embeddings to identify specific individuals

## Installation

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/yourusername/face-recognition-attendance.git
cd face-recognition-attendance

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- facenet-pytorch
- ultralytics (YOLOv8)
- pytorch-lightning
- scikit-learn
- PIL
- numpy
- pandas
- matplotlib
- tqdm
- opencv-python

## Dataset Preparation

The system uses the Labeled Faces in the Wild (LFW) dataset, enhancing it with:

1. **Gender Labels**: Manual gender annotations stored in JSON format
2. **Simulated Classroom Images**: Generated composite images with varying face positions, orientations, and backgrounds
3. **YOLO Annotations**: Formatted data for YOLOv8 training

### Dataset Structure

```

DataLFW/
├── person_name1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person_name2/
│   └── ...
├── gender_labels.json
├── backgrounds/
│   └── classrooms/
├── SimulatedData/
│   ├── sim_001.jpg
│   ├── sim_002.jpg
│   └── ...
└── yolo_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

## Usage

### Training Models

1. **Face Detector Training**:

```python
from ultralytics import YOLO

# Initialize a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model on our custom dataset
results = model.train(
    data="path/to/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="face_detector_yolov8",
    patience=10,
    device=0,
)
```

2. **Gender Classifier Training**:

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Initialize the model
gender_model = GenderClassifier(class_weights=class_weights)

# Define trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[EarlyStopping(monitor="val_loss", patience=3)]
)

# Train the model
trainer.fit(gender_model, train_loader, val_loader)
```

3. **Face Identification Model Training**:

```python
# Initialize the model
face_id_model = FaceIDModel(num_classes=num_classes)

# Train the model
trainer = pl.Trainer(max_epochs=30)
trainer.fit(face_id_model, train_loader, val_loader)
```

### Processing Images

```python
# Load the trained models
face_detector = YOLO("path/to/face_detector.pt")
gender_model = GenderClassifier.load_from_checkpoint("gender_classifier.ckpt")
face_identifier = FaceIDModel.load_from_checkpoint("face_id_model.ckpt")

# Process a classroom image
results_df, attendance_df = process_single_photo("classroom_image.jpg")
```

## Evaluation Metrics

The system evaluates model performance using:

- **Precision, Recall, F1-Score**: Measures of classification accuracy
- **Confusion Matrix**: Visual representation of model predictions
- **IoU (Intersection over Union)**: For face detection quality assessment

## Sample Results

The attendance system outputs:

1. **Face Predictions**: CSV file with detected faces, identities, and confidence scores
2. **Attendance Log**: CSV file recording present/absent status for each student
3. **Visualization**: Annotated images showing detected faces with identity labels

## Acknowledgments

This project utilizes:

- Labeled Faces in the Wild (LFW) dataset
- YOLOv8 object detection framework
- PyTorch and PyTorch Lightning

## License

This project is licensed under the MIT License - see the LICENSE file for details.
