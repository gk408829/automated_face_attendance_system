# Import Libraries/Packages
import os
import json
import uuid
import random
import glob
import numpy as np
from tqdm.notebook import tqdm
from PIL import Image, ImageFilter, ImageDraw
from collections import Counter
from sklearn.datasets import fetch_lfw_people
from facenet_pytorch import MTCNN
import torch
from shutil import copy2
from ultralytics import YOLO
from pathlib import Path
import cv2
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split
import torchvision.models as models
from sklearn.model_selection import StratifiedShuffleSplit
import pytorch_lightning as pl
from pathlib import Path
import shutil
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device

# Define directories
data_dir = "/Users/gauravkhanal/Documents/attendance_system/DataLFW"
background_dir = os.path.join(data_dir, "backgrounds", "classrooms")
gender_labels_file = os.path.join(data_dir, "gender_labels.json")
sim_output_dir = "/Users/gauravkhanal/Documents/attendance_system/SimulatedData"
bounding_box_file = os.path.join(sim_output_dir, "bounding_boxes.json")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(background_dir, exist_ok=True)
os.makedirs(sim_output_dir, exist_ok=True)

# Load LFW Dataset and subsample 20 individuals
lfw = fetch_lfw_people(min_faces_per_person=10, resize=1.0)

# Extract metadata
names = lfw.target_names
labels = lfw.target
counts = Counter(labels)

# Filter for more samples
qualified_indices = [i for i, c in counts.items() if c >= 15]
selected_names = [names[i] for i in qualified_indices][:20]

# Create directory substructure
for name in selected_names:
    os.makedirs(os.path.join(data_dir, name.replace(" ", "_")), exist_ok=True)

# Save Filtered Images
for img, label in tqdm(zip(lfw.images, lfw.target), total=len(lfw.images), desc="Saving images"):
    person = names[label]
    if person in selected_names:
        path = os.path.join(data_dir, person.replace(" ", "_"))
        img_pil = Image.fromarray((img * 255).astype("uint8"))
        img_pil.save(os.path.join(path, f"{uuid.uuid4().hex}.jpg"))

# Define gender labels
gender_labels = {
    "Abdullah_Gul": "male",
    "Alvaro_Uribe": "male",
    "Andy_Roddick": "male",
    "Ariel_Sharon": "male",
    "Atal_Bihari_Vajpayee": "male",
    "Bill_Simon": "male",
    "Colin_Powell": "male",
    "Dominique_de_Villepin":
    "male", "George_W_Bush": "male",
    "Gerhard_Schroeder": "male",
    "Gloria_Macapagal_Arroyo": "female",
    "Jacques_Chirac": "male",
    "John_Kerry": "male",
    "Jose_Maria_Aznar": "male",
    "Lindsay_Davenport": "female",
    "Mohammed_Al-Douri": "male",
    "Serena_Williams": "female",
    "Vladimir_Putin": "male",
    "Winona_Ryder": "female",
    "Silvio_Berlusconi": "male"
}

# Save labels as json file
with open(gender_labels_file, "w") as f:
    json.dump(gender_labels, f, indent=2)

# Initialize MTCNN and bounding box data
mtcnn = MTCNN(keep_all=False, device=device)
bbox_data = {}

# Get all image paths first for accurate progress tracking
all_image_paths = []
for person_folder in tqdm(os.listdir(data_dir), desc="Scanning folders"):
    person_path = os.path.join(data_dir, person_folder)
    if os.path.isdir(person_path):
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith('.jpg'):
                all_image_paths.append(os.path.join(person_path, img_file))

# Perform face detection and obtain bounding box data
for img_path in tqdm(all_image_paths, desc="Detecting faces"):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                # Store path as string for JSON compatibility
                bbox_data[img_path] = boxes[0].tolist()
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
        continue

# Save results
with open(bounding_box_file, "w") as f:
    json.dump(bbox_data, f, indent=2)

print(f"Face detection complete. Detected {len(bbox_data)} faces out of {len(all_image_paths)} images.")

# Define simulation configurations
sim_config = {
    "face_base": data_dir,
    "bg_base": background_dir,
    "output_dir": sim_output_dir,
    "output_size": (640, 480),
    "num_images": 300,
    "faces_per_image": (3, 7),
    "apply_blur": True,
    "background_ext": ".jpg",
    "face_scale_range": (0.1, 0.3),
    "rotation_range": (-15, 15),
    "blur_chance": 0.3,
    "blur_radius_range": (0.5, 1.5)
}

# Define function to compute how much two bounding boxes overlap relative to their combined area
# i.e., Intersection over Union (IoU)
def calculate_iou(box1, box2):
    # Find intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # Calculate individual areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Compute IoU
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Define function to check if a new bounding box significantly overlaps with any existing boxes
def has_overlap(bbox, existing_bboxes, threshold=0.01):
    return any(calculate_iou(bbox, b) > threshold for b in existing_bboxes)

# Collect face image paths
face_paths = []
for person in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person)
    if os.path.isdir(person_path):
        for img_file in glob.glob(os.path.join(person_path, "*.jpg")):
            face_paths.append((img_file, person))

# Collecting Background Image Paths
bg_paths = [os.path.join(background_dir, f)
            for f in os.listdir(background_dir)
            if f.lower().endswith(sim_config["background_ext"])]

if not bg_paths:
    raise ValueError("No background images found. Please upload at least one.")

# Split face_paths into male and female lists

# Initialization
female_faces = []
male_faces = []

# Gender classification
for path, label in face_paths:
    gender = gender_labels.get(label, "male")  # default to male
    if gender == "female":
        female_faces.append((path, label))
    else:
        male_faces.append((path, label))

# Sanity check
if not female_faces:
    raise ValueError("No female faces found. Cannot oversample females.")
if not male_faces:
    raise ValueError("No male faces found.")

# Set female oversampling ratio
female_ratio = 0.6

# Initialize male and female statistics
total_female = 0
total_male = 0

# Initialization
annotations = {}

# Face selection
for img_idx in tqdm(range(sim_config["num_images"])):
    num_faces = random.randint(*sim_config["faces_per_image"])
    num_female = int(num_faces * female_ratio)
    num_male = num_faces - num_female

    # Sample females (if needed)
    selected_females = random.choices(female_faces, k=num_female) if len(female_faces) < num_female \
                       else random.sample(female_faces, num_female)

    # Sample males (no replacement)
    if len(male_faces) < num_male:
        continue
    selected_males = random.sample(male_faces, num_male)

    # Combine and shuffle selections for natural arrangement
    selected_faces = selected_females + selected_males
    random.shuffle(selected_faces)

    # Create background canvas
    bg_path = random.choice(bg_paths)
    bg = Image.open(bg_path).convert("RGBA").resize(sim_config["output_size"])
    canvas = bg.copy()
    image_bboxes = []

    for face_path, label in selected_faces:
        try:
            face = Image.open(face_path).convert("RGBA")

            # Resize
            scale = random.uniform(*sim_config["face_scale_range"])
            face_size = int(sim_config["output_size"][0] * scale)
            face = face.resize((face_size, face_size))

            # Rotate
            face = face.rotate(random.uniform(*sim_config["rotation_range"]), expand=True)

            # Optional blur
            if sim_config["apply_blur"] and random.random() < sim_config["blur_chance"]:
                face = face.filter(ImageFilter.GaussianBlur(
                    radius=random.uniform(*sim_config["blur_radius_range"])))

            # Occlusion
            if random.random() < 0.3:
                draw = ImageDraw.Draw(face)
                occ_w = int(face.width * random.uniform(0.2, 0.4))
                occ_h = int(face.height * random.uniform(0.1, 0.3))
                occ_x = random.randint(0, face.width - occ_w)
                occ_y = random.randint(0, face.height - occ_h)
                draw.rectangle([occ_x, occ_y, occ_x + occ_w, occ_y + occ_h],
                               fill=(0, 0, 0, 180))

            # Positioning
            max_x = sim_config["output_size"][0] - face.size[0]
            max_y = sim_config["output_size"][1] - face.size[1]
            if max_x <= 0 or max_y <= 0:
                continue

            # Intelligent placement
            for attempt in range(10):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                bbox = [x, y, x + face.size[0], y + face.size[1]]

                if not has_overlap(bbox, [b["bbox"] for b in image_bboxes]):
                    canvas.paste(face, (x, y), face)
                    image_bboxes.append({"label": label, "bbox": bbox})
                    # Count gender
                    if gender_labels[label] == "female":
                        total_female += 1
                    else:
                        total_male += 1
                    break

        except Exception as e:
            print(f"Skipping {face_path}: {e}")
            continue

    if not image_bboxes:
        continue

    # Save composite image and record face positions
    image_name = f"sim_{img_idx:03d}.jpg"
    image_path = os.path.join(sim_output_dir, image_name)
    canvas.convert("RGB").save(image_path)
    annotations[image_name] = image_bboxes

# Save annotations in a json file
annotations_path = os.path.join(sim_output_dir, "annotations.json")
with open(annotations_path, "w") as f:
    json.dump(annotations, f, indent=2)

# Print summary
print(f"Done. {len(annotations)} images saved to '{sim_output_dir}'")
print(f"Total female faces: {total_female}")
print(f"Total male faces: {total_male}")
female_pct = (total_female / (total_female + total_male)) * 100
print(f"Female face percentage: {female_pct:.2f}%")

# Define paths
annotation_file = "/content/drive/MyDrive/case_study/DataLFW/SimulatedData/annotations.json"
output_base = "/content/drive/MyDrive/case_study/DataLFW/yolo_dataset"

# Define split ratio
train_split = 0.8

# Load annotation and gender label data
with open(annotation_file, "r") as f:
    annotations = json.load(f)

with open(gender_labels_file, "r") as f:
    gender_labels = json.load(f)

# Prepare image list and shuffle for train/val split
image_names = list(annotations.keys())
random.shuffle(image_names)
split_index = int(len(image_names) * train_split)

# Create YOLOv8 directory structure
for subset in ['train', 'val']:
    os.makedirs(os.path.join(output_base, "images", subset), exist_ok=True)
    os.makedirs(os.path.join(output_base, "labels", subset), exist_ok=True)

# Loop through and process
for i, image_name in enumerate(tqdm(image_names)):
    subset = "train" if i < split_index else "val"

    img_path = os.path.join(sim_output_dir, image_name)
    label_path = os.path.join(output_base, "labels", subset, image_name.replace(".jpg", ".txt"))
    img_output_path = os.path.join(output_base, "images", subset, image_name)

    try:
        img = Image.open(img_path)
        w, h = img.size

        with open(label_path, "w") as f_out:
            for ann in annotations[image_name]:
                x_min, y_min, x_max, y_max = ann["bbox"]
                label = ann["label"]

                # Determine class ID based on gender
                gender = gender_labels.get(label, "male")
                class_id = 0 if gender == "female" else 1  # female=0, male=1

                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                box_width = (x_max - x_min) / w
                box_height = (y_max - y_min) / h

                f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        copy2(img_path, img_output_path)

    except Exception as e:
        print(f"Skipping {image_name}: {e}")

print(f"YOLOv8 dataset with gender classes created at: {output_base}")

# Define paths
yaml_path = os.path.join(output_base, "dataset.yaml")
train_images = os.path.join(output_base, "images", "train")
val_images = os.path.join(output_base, "images", "val")
output_crop_dir = os.path.join(output_base, "cropped_faces_for_gender")
os.makedirs(output_crop_dir, exist_ok=True)

# Write YOLOv8 config in yaml
yaml_content = f"""path: {output_base}
train: images/train
val: images/val

nc: 2
names: ['female', 'male']
"""

with open(yaml_path, 'w') as f:
    f.write(yaml_content.strip())

# Verify yaml
print(f"YOLOv8 config saved to:\n{yaml_path}")
print("Config content preview:")
print(yaml_content)

# Define a path to save YOLO runs
drive_path = "/content/drive/MyDrive/case_study/yolo_models"
os.makedirs(drive_path, exist_ok=True)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Train YOLOv8 model
results = yolo_model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    name="face_detector_yolov8",
    patience=10,
    device=0,
    pretrained=True,
)

# Load Best Model
best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
face_model = YOLO(best_model_path)

# Copy ALL training artifacts (not just weights)
# !cp -r "/content/runs/detect/face_detector_yolov8" "{drive_path}"

# Evaluate model on validation set
print("Evaluating model on validation set")
metrics = face_model.val()

# Visualize evaluation metrics
plot_files = [
    'PR_curve.png',
    'R_curve.png',
    'P_curve.png',
    'F1_curve.png'
]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, plot_file in zip(axes.flat, plot_files):
    img_path = os.path.join(results.save_dir, plot_file)
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(plot_file.split('.')[0])
plt.tight_layout()
plt.show()

# Crop detected faces from the validation images
results = face_model.predict(source=val_images, conf=0.3, save=True)

for result in tqdm(results, desc="Cropping faces"):
    try:
        img_path = result.path
        img = cv2.imread(img_path)
        boxes = result.boxes.xyxy

        # Bounding box handling & cropping
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            if x1 < x2 and y1 < y2:
                cropped_face = img[y1:y2, x1:x2]
                # Save cropped faces
                out_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_face{i}.jpg"
                out_path = os.path.join(output_crop_dir, out_name)
                cv2.imwrite(out_path, cropped_face)

    except Exception as e:
        print(f"Error processing {result.path}: {e}")

print(f"Cropped faces saved to: {output_crop_dir}")

# Define paths again
lfw_path = os.path.abspath("/content/drive/MyDrive/case_study/DataLFW")
gender_labels_path = "/content/drive/MyDrive/case_study/DataLFW/SimulatedData/gender_labels.json"
gender_dataset_path = os.path.join(lfw_path, "gender_dataset")

# Create directories
os.makedirs(gender_dataset_path, exist_ok=True)
os.makedirs(os.path.join(gender_dataset_path, "male"), exist_ok=True)
os.makedirs(os.path.join(gender_dataset_path, "female"), exist_ok=True)

# Load gender labels
with open(gender_labels_path, "r") as f:
    gender_labels = json.load(f)

# Copy LFW images to male/female folders
# Directory Processing
for person in tqdm(os.listdir(lfw_path), desc="Processing persons"):
    person_path = os.path.join(lfw_path, person)
    if not os.path.isdir(person_path) or person == "gender_dataset":
        continue
    # Gender Validation
    gender = gender_labels.get(person)
    if gender not in ["male", "female"]:
        continue
    target_dir = os.path.join(gender_dataset_path, gender)
    # Image Copying
    for img_name in os.listdir(person_path):
        src = os.path.join(person_path, img_name)
        dst = os.path.join(target_dir, f"{person}_{img_name}")
        if os.path.isfile(src):
            shutil.copy(src, dst)

print("Gender dataset created!")

# Create gender dataset class
class GenderDataset(Dataset):
    # Initialization
    def __init__(self, data_dir, transform=None, augment_female=False):
        self.samples = []
        self.transform = transform
        self.augment_female = augment_female

        # Load images with gender labels
        for gender in ["male", "female"]:
            label = 0 if gender == "male" else 1
            gender_path = os.path.join(data_dir, gender)
            for img in tqdm(os.listdir(gender_path), desc=f"{gender} images"):
                img_path = os.path.join(gender_path, img)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples) # total number of samples

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        # Female-specific augmentation
        if self.augment_female and label == 1:
            image = self._augment_image(image)

        # Apply standard transforms
        if self.transform:
            image = self.transform(image)
        return image, label

    # Female Augmentation
    def _augment_image(self, image):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.9),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2))
        ])(image)

# Define PyTorch data transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define dataset loading and train test
full_dataset = GenderDataset(gender_dataset_path, transform=transform, augment_female=True)
labels = [label for _, label in full_dataset]

# Print class distribution
print("Class distribution:", dict(zip(*np.unique(labels, return_counts=True))))

# Perform stratified train-test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(split.split(np.zeros(len(labels)), labels))

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

# Compute class weights for the loss function
class_counts = np.bincount(labels)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
class_weights /= class_weights.sum()
print(f"Class weights for loss: {class_weights}")

# Utilize WeightedRandomSampler
train_labels = [full_dataset[i][1] for i in train_idx]
sample_weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define gender classification model
class GenderClassifier(pl.LightningModule):
    def __init__(self, class_weights=None):
        super().__init__()
        # Backbone: Pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace final layer for binary classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        # Loss function with class weighting
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Forward Pass
    def forward(self, x):
        return self.backbone(x)

    # Training Loop
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    # Validation Loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        pred_labels = torch.argmax(preds, dim=1)
        acc = accuracy_score(y.cpu(), pred_labels.cpu())
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss

    # Optimizer Configuration
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Load gender model
gender_model = GenderClassifier(class_weights=class_weights)

# Define early stopping
early_stop_callback = EarlyStopping(
    monitor="val_loss",               # Metric to monitor
    patience=3,                       # Wait 3 epochs without improvement
    mode="min",                       # Minimize val_loss
    verbose=True
)

# Trainer configuration
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",  # Use GPU
    logger=False,                                               # Disables TensorBoard
    callbacks=[early_stop_callback]
)

# Fit the model
trainer.fit(gender_model, train_loader, val_loader)

# Validate model
trainer.validate(gender_model, val_loader)

# Save model checkpoint
trainer.save_checkpoint("gender_classifier.ckpt")

# Detailed evaluation
gender_model.eval()
all_preds, all_labels = [], []

for x, y in tqdm(val_loader, desc="Predicting"):
    with torch.no_grad():
        y_hat = gender_model(x)
        preds = torch.argmax(y_hat, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(y.cpu().numpy())

# Generate classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["male", "female"], zero_division=0))

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=["male", "female"]).plot()
plt.title("Confusion Matrix")
plt.show()

# Face Identification Configuration
data_dir = "/content/drive/MyDrive/case_study/DataLFW"
cropped_faces_dir = "/content/drive/MyDrive/case_study/DataLFW/yolo_dataset/cropped_faces_for_gender"
batch_size = 32        # Number of images processed per batch during training/inference
img_size = 160         # Input image dimensions (height and width in pixels)
num_workers = 2        # Number of CPU threads for parallel data loading
max_epochs = 30        # Maximum training iterations over the entire dataset
val_split = 0.2        # Fraction of data reserved for validation (here, 20%)
use_resnet50 = False   # Flag to choose between backbones

# Define PyTorch transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load gender label names
with open(os.path.join(data_dir, "SimulatedData", "gender_labels.json")) as f:
    gender_labels = json.load(f)

valid_classes = set(gender_labels.keys())

# Custom dataset
class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # All subfolders
        all_classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        # Only keep classes present in gender_labels.json
        filtered = sorted([c for c in all_classes if c in valid_classes])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(filtered)}
        return filtered, class_to_idx

# Load data
dataset = FilteredImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

# Check class names
print(class_names)

# Check no. of classes
print(f"\n Using {num_classes} identities from gender_labels.json")

# Split data
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Class weights and sampler
train_labels = [dataset.samples[i][1] for i in train_dataset.indices]
class_counts = np.bincount(train_labels)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Define face identification / recognition model class
class FaceIDModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()
        if use_resnet50:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"logits": logits, "y": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
# Define model
face_id_model = FaceIDModel(num_classes=num_classes)

# Define callbacks
checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best-faceid")
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)

# Define trainer
trainer2 = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=False
)

# Fit the model
trainer2.fit(face_id_model, train_loader, val_loader)

# Validate the model
trainer2.validate(face_id_model, val_loader)

# Evaluate
face_id_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        logits = face_id_model(x)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Print confusion matrix
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Reds", xticks_rotation=90)
plt.title("Confusion Matrix")
plt.show()

# Optimized Face Prediction Pipeline
def predict_faces(model, cropped_faces_dir, transform, class_names, batch_size=32, device='cuda'):
    """Batch-process face images for efficient prediction."""
    model.eval().to(device)
    predictions = []

    # Get sorted list of image paths (faster than os.listdir + filtering)
    img_paths = sorted([
        os.path.join(cropped_faces_dir, f)
        for f in os.listdir(cropped_faces_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Found {len(img_paths)} face images for prediction")

    # Process in batches for maximum GPU utilization
    for i in tqdm(range(0, len(img_paths), batch_size), desc="🔍 Predicting identities"):
        batch_paths = img_paths[i:i+batch_size]
        batch_images = []

        # Load and transform all images in batch
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_images.append(transform(img))
            except Exception as e:
                print(f"⚠️ Error loading {os.path.basename(path)}: {str(e)}")
                continue

        if not batch_images:
            continue

        # Stack and move to device
        batch_tensor = torch.stack(batch_images).to(device)

        # Batch prediction
        with torch.inference_mode():
            logits = model(batch_tensor)
            if isinstance(logits, list):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            scores, pred_idxs = torch.max(probs, dim=1)

        # Store results
        for path, pred_idx, score in zip(batch_paths, pred_idxs.cpu(), scores.cpu()):
            predictions.append({
                "image": os.path.basename(path),
                "identity": class_names[pred_idx.item()],
                "score": round(score.item(), 4)
            })

    return predictions

# Get predictions on cropped faces
predictions = predict_faces(
    model=face_id_model,
    cropped_faces_dir=cropped_faces_dir,
    transform=transform,
    class_names=class_names,
    batch_size=64  # Adjust based on GPU memory
)

preds_df = pd.DataFrame(predictions)
preds_df.head(20)

# Attendance System Design

# Load fine-tuned face detector (YOLOv8)
face_detector = YOLO("runs/detect/face_detector_yolov8/weights/best.pt")

# Load fine-tuned gender classifer
gender_model = GenderClassifier.load_from_checkpoint("gender_classifier.ckpt")
gender_model.eval().to(device)

# Load fine tuned face identifier
face_identifer = FaceIDModel.load_from_checkpoint('best-facenet.ckpt').to(device).eval()

# Load Gallery Embeddings (from LFW identities)
gallery_npz = np.load("lfw_identity_gallery.npz", allow_pickle=True)
gallery = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in gallery_npz.items()}
gallery_names = list(gallery.keys())
gallery_stack = torch.stack(list(gallery.values()))

# Load Ground-Truth Genders
with open("DataLFW/gender_labels.json", "r") as f:
    gender_labels = json.load(f)

# Transforms
transform_gender = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_face = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def process_single_photo(image_path, output_dir=None):
    """Full pipeline for single photo processing"""
    # Initialize
    models = load_models()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or f"attendance_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Face Detection
    detections = models['face_detector'](image_path)[0]
    img = Image.open(image_path).convert('RGB')
    face_crops = []

    # Crop all detected faces
    for i, box in enumerate(detections.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        face_crops.append(img.crop((x1, y1, x2, y2)))

    if not face_crops:
        print("⚠️ No faces detected!")
        return None

    # 2. Batch Processing
    with torch.inference_mode():
        # Preprocess all faces
        face_tensors = torch.stack([transform_face(face) for face in face_crops]).to(DEVICE)
        gender_tensors = torch.stack([transform_gender(face) for face in face_crops]).to(DEVICE)

        # Get predictions
        gender_probs = torch.softmax(models['gender_model'](gender_tensors), dim=1)
        face_embs = models['facenet'](face_tensors)
        face_embs = torch.nn.functional.normalize(face_embs, p=2, dim=1)

        # Compare with gallery
        similarity_matrix = face_embs @ gallery_embeddings.T
        best_scores, best_indices = torch.max(similarity_matrix, dim=1)

    # 3. Generate Results
    results = []
    for i, (face, score, idx) in enumerate(zip(face_crops, best_scores, best_indices)):
        # Save face crop
        face_path = os.path.join(output_dir, f"face_{i}.jpg")
        face.save(face_path)

        # Get predictions
        gender = "male" if gender_probs[i][0] > GENDER_CONF_THRESH else "female"
        identity = gallery_names[idx] if score >= MIN_FACE_CONFIDENCE else "Unknown"

        results.append({
            'face_id': i,
            'image_path': face_path,
            'identity': identity,
            'confidence': round(score.item(), 4),
            'gender': gender,
            'gender_confidence': round(max(gender_probs[i]).item(), 4)
        })

    # 4. Create Attendance Log
    attendance = []
    for student in gallery_names:
        present = any(r['identity'] == student for r in results)
        attendance.append({
            'student': student,
            'present': int(present),
            'timestamp': timestamp
        })

    # Save outputs
    results_df = pd.DataFrame(results)
    attendance_df = pd.DataFrame(attendance)

    results_df.to_csv(os.path.join(output_dir, 'face_predictions.csv'), index=False)
    attendance_df.to_csv(os.path.join(output_dir, 'attendance.csv'), index=False)

    # Print summary
    print(f"\Detected {len(results)} faces:")
    print(results_df[['face_id', 'identity', 'confidence']])

    print(f"\n Attendance Summary:")
    print(attendance_df)

    print(f"\n Results saved to: {output_dir}")

    return results_df, attendance_df

# Fill Attendance sheet
process_single_photo("classroom_simulated.jpg")



