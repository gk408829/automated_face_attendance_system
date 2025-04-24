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
import shutil
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix,
                            ConfusionMatrixDisplay)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

# Constants
GENDER_CONF_THRESH = 0.7  # Confidence threshold for gender classification
MIN_FACE_CONFIDENCE = 0.6  # Minimum confidence for face recognition
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def setup_directories():
    """Initialize all required directories"""
    data_dir = "/Users/gauravkhanal/Documents/attendance_system/DataLFW"
    dirs = {
        'data': data_dir,
        'background': os.path.join(data_dir, "backgrounds", "classrooms"),
        'sim_output': "/Users/gauravkhanal/Documents/attendance_system/SimulatedData",
        'yolo_dataset': os.path.join(data_dir, "yolo_dataset"),
        'gender_dataset': os.path.join(data_dir, "gender_dataset"),
        'output_crop': os.path.join(data_dir, "yolo_dataset", "cropped_faces_for_gender")
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def load_lfw_data(min_faces=10, max_people=20):
    """Load and filter LFW dataset"""
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, resize=1.0)
    names = lfw.target_names
    counts = Counter(lfw.target)

    # Filter for people with sufficient samples
    qualified_indices = [i for i, c in counts.items() if c >= min_faces + 5]  # Require 15+ samples
    selected_names = [names[i] for i in qualified_indices][:max_people]

    return lfw, selected_names


def save_filtered_images(lfw_data, names, output_dir):
    """Save filtered images to organized directory structure"""
    for name in names:
        os.makedirs(os.path.join(output_dir, name.replace(" ", "_")), exist_ok=True)

    for img, label in tqdm(zip(lfw_data.images, lfw_data.target),
                           total=len(lfw_data.images),
                           desc="Saving images"):
        person = lfw_data.target_names[label]
        if person in names:
            path = os.path.join(output_dir, person.replace(" ", "_"))
            img_pil = Image.fromarray((img * 255).astype("uint8"))
            img_pil.save(os.path.join(path, f"{uuid.uuid4().hex}.jpg"))


def initialize_gender_labels(selected_names):
    """Define and save gender labels"""
    gender_labels = {
        "Abdullah_Gul": "male",
        "Alvaro_Uribe": "male",
        "Andy_Roddick": "male",
        "Ariel_Sharon": "male",
        "Atal_Bihari_Vajpayee": "male",
        "Bill_Simon": "male",
        "Colin_Powell": "male",
        "Dominique_de_Villepin": "male",
        "George_W_Bush": "male",
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

    # Filter to only include selected names
    gender_labels = {k: v for k, v in gender_labels.items() if k in selected_names}

    return gender_labels


def detect_faces(data_dir, device=DEVICE):
    """Perform face detection using MTCNN"""
    mtcnn = MTCNN(keep_all=False, device=device)
    bbox_data = {}

    # Get all image paths
    all_image_paths = []
    for person_folder in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_folder)
        if os.path.isdir(person_path):
            all_image_paths.extend(
                os.path.join(person_path, img_file)
                for img_file in os.listdir(person_path)
                if img_file.lower().endswith('.jpg')
            )

    # Detect faces
    for img_path in tqdm(all_image_paths, desc="Detecting faces"):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                boxes, _ = mtcnn.detect(img)
                if boxes is not None:
                    bbox_data[img_path] = boxes[0].tolist()
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue

    return bbox_data


def create_simulated_dataset(config):
    """Create simulated classroom dataset"""
    # Load face and background paths
    face_paths = []
    for person in os.listdir(config["face_base"]):
        person_path = os.path.join(config["face_base"], person)
        if os.path.isdir(person_path):
            face_paths.extend((os.path.join(person_path, f), person)
                              for f in os.listdir(person_path)
                              if f.lower().endswith('.jpg'))

    bg_paths = [os.path.join(config["bg_base"], f)
                for f in os.listdir(config["bg_base"])
                if f.lower().endswith(config["background_ext"])]

    if not bg_paths:
        raise ValueError("No background images found.")

    # Split by gender
    female_faces = []
    male_faces = []
    for path, label in face_paths:
        gender = config["gender_labels"].get(label, "male")
        if gender == "female":
            female_faces.append((path, label))
        else:
            male_faces.append((path, label))

    # Simulation
    annotations = {}
    total_female = total_male = 0

    for img_idx in tqdm(range(config["num_images"]), desc="Generating simulated images"):
        num_faces = random.randint(*config["faces_per_image"])
        num_female = int(num_faces * config["female_ratio"])
        num_male = num_faces - num_female

        # Sample faces
        selected_females = random.choices(female_faces, k=num_female) if len(female_faces) < num_female \
            else random.sample(female_faces, num_female)
        selected_males = random.sample(male_faces, num_male) if len(male_faces) >= num_male else []

        selected_faces = selected_females + selected_males
        random.shuffle(selected_faces)

        # Create composite image
        bg_path = random.choice(bg_paths)
        bg = Image.open(bg_path).convert("RGBA").resize(config["output_size"])
        canvas = bg.copy()
        image_bboxes = []

        for face_path, label in selected_faces:
            try:
                face = Image.open(face_path).convert("RGBA")

                # Apply transformations
                scale = random.uniform(*config["face_scale_range"])
                face_size = int(config["output_size"][0] * scale)
                face = face.resize((face_size, face_size))

                face = face.rotate(random.uniform(*config["rotation_range"]), expand=True)

                if config["apply_blur"] and random.random() < config["blur_chance"]:
                    face = face.filter(ImageFilter.GaussianBlur(
                        radius=random.uniform(*config["blur_radius_range"])))

                # Position face
                max_x = config["output_size"][0] - face.size[0]
                max_y = config["output_size"][1] - face.size[1]

                for attempt in range(10):  # Try 10 positions
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                    bbox = [x, y, x + face.size[0], y + face.size[1]]

                    if not has_overlap(bbox, [b["bbox"] for b in image_bboxes]):
                        canvas.paste(face, (x, y), face)
                        image_bboxes.append({"label": label, "bbox": bbox})
                        if config["gender_labels"][label] == "female":
                            total_female += 1
                        else:
                            total_male += 1
                        break

            except Exception as e:
                print(f"Skipping {face_path}: {e}")
                continue

        if image_bboxes:
            image_name = f"sim_{img_idx:03d}.jpg"
            image_path = os.path.join(config["output_dir"], image_name)
            canvas.convert("RGB").save(image_path)
            annotations[image_name] = image_bboxes

    return annotations, total_female, total_male


def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def has_overlap(bbox, existing_bboxes, threshold=0.01):
    """Check for significant overlap with existing boxes"""
    return any(calculate_iou(bbox, b) > threshold for b in existing_bboxes)


def prepare_yolo_dataset(annotations, gender_labels, output_base, train_split=0.8):
    """Prepare YOLO format dataset"""
    # Create directory structure
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_base, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(output_base, "labels", subset), exist_ok=True)

    # Shuffle and split
    image_names = list(annotations.keys())
    random.shuffle(image_names)
    split_index = int(len(image_names) * train_split)

    # Process images
    for i, image_name in enumerate(tqdm(image_names, desc="Preparing YOLO dataset")):
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
                    class_id = 0 if gender == "female" else 1

                    # Convert to YOLO format
                    x_center = ((x_min + x_max) / 2) / w
                    y_center = ((y_min + y_max) / 2) / h
                    box_width = (x_max - x_min) / w
                    box_height = (y_max - y_min) / h

                    f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

            copy2(img_path, img_output_path)

        except Exception as e:
            print(f"Skipping {image_name}: {e}")

    # Create YAML config
    yaml_path = os.path.join(output_base, "dataset.yaml")
    yaml_content = f"""path: {output_base}
    train: images/train
    val: images/val

    nc: 2
    names: ['female', 'male']
    """
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())

    print(f"YOLOv8 dataset created at: {output_base}")
    return yaml_path


def train_yolo_model(yaml_path, epochs=100, imgsz=640, batch=16, patience=10):
    """Train YOLO face detection model"""
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="face_detector_yolov8",
        patience=patience,
        device=0 if torch.cuda.is_available() else "cpu",
        pretrained=True,
    )

    # Load best model
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    return YOLO(best_model_path)


def evaluate_yolo_model(model, val_images):
    """Evaluate YOLO model performance"""
    print("Evaluating model on validation set")
    metrics = model.val()

    # Visualize metrics
    plot_files = ['PR_curve.png', 'R_curve.png', 'P_curve.png', 'F1_curve.png']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, plot_file in zip(axes.flat, plot_files):
        img_path = os.path.join(model.trainer.save_dir, plot_file)
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(plot_file.split('.')[0])

    plt.tight_layout()
    plt.show()
    return metrics


def crop_detected_faces(model, image_dir, output_dir, conf=0.3):
    """Crop faces from detected images"""
    os.makedirs(output_dir, exist_ok=True)
    results = model.predict(source=image_dir, conf=conf, save=True)

    for result in tqdm(results, desc="Cropping faces"):
        try:
            img = cv2.imread(result.path)
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                if x1 < x2 and y1 < y2:
                    cropped_face = img[y1:y2, x1:x2]
                    out_name = f"{os.path.splitext(os.path.basename(result.path))[0]}_face{i}.jpg"
                    cv2.imwrite(os.path.join(output_dir, out_name), cropped_face)
        except Exception as e:
            print(f"Error processing {result.path}: {e}")

    print(f"Cropped faces saved to: {output_dir}")


class GenderDataset(Dataset):
    """Dataset for gender classification"""

    def __init__(self, data_dir, transform=None, augment_female=False):
        self.samples = []
        self.transform = transform
        self.augment_female = augment_female

        # Load images with gender labels
        for gender in ["male", "female"]:
            label = 0 if gender == "male" else 1
            gender_path = os.path.join(data_dir, gender)
            for img in os.listdir(gender_path):
                img_path = os.path.join(gender_path, img)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.augment_female and label == 1:
            image = self._augment_image(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _augment_image(self, image):
        """Apply female-specific augmentation"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.9),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                    scale=(0.8, 1.2))
        ])(image)


class GenderClassifier(pl.LightningModule):
    """Gender classification model"""

    def __init__(self, class_weights=None):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        pred_labels = torch.argmax(preds, dim=1)
        acc = accuracy_score(y.cpu(), pred_labels.cpu())
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train_gender_classifier(data_dir, max_epochs=50, patience=3):
    """Train gender classification model"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset
    full_dataset = GenderDataset(data_dir, transform=transform, augment_female=True)
    labels = [label for _, label in full_dataset]

    # Stratified split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(split.split(np.zeros(len(labels)), labels))

    # Compute class weights
    class_counts = np.bincount(labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights /= class_weights.sum()

    # Create data loaders
    train_labels = [full_dataset[i][1] for i in train_idx]
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=32,
        sampler=sampler
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=32,
        shuffle=False
    )

    # Initialize model
    model = GenderClassifier(class_weights=class_weights)

    # Define callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[early_stop]
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)

    # Save model
    trainer.save_checkpoint("gender_classifier.ckpt")

    return model


class FaceIDModel(pl.LightningModule):
    """Face recognition model"""

    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()
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


def train_face_recognition(data_dir, max_epochs=30, val_split=0.2):
    """Train face recognition model"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create weighted sampler
    train_labels = [dataset.samples[i][1] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # Initialize model
    model = FaceIDModel(num_classes=num_classes)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-faceid"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)

    return model


def predict_faces(model, image_dir, transform, class_names, batch_size=32):
    """Batch-process face images for prediction"""
    model.eval().to(DEVICE)
    predictions = []

    # Get image paths
    img_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Found {len(img_paths)} face images for prediction")

    # Process in batches
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Predicting identities"):
        batch_paths = img_paths[i:i + batch_size]
        batch_images = []

        # Load and transform images
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_images.append(transform(img))
            except Exception as e:
                print(f"Error loading {os.path.basename(path)}: {str(e)}")
                continue

        if not batch_images:
            continue

        # Predict
        batch_tensor = torch.stack(batch_images).to(DEVICE)

        with torch.inference_mode():
            logits = model(batch_tensor)
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


def process_attendance(image_path, models, output_dir=None):
    """Full attendance processing pipeline"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or f"attendance_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Face Detection
    detections = models['face_detector'](image_path)[0]
    img = Image.open(image_path).convert('RGB')
    face_crops = []

    for i, box in enumerate(detections.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        face_crops.append(img.crop((x1, y1, x2, y2)))

    if not face_crops:
        print("No faces detected!")
        return None

    # 2. Batch Processing
    with torch.inference_mode():
        # Preprocess faces
        face_tensors = torch.stack([transforms.Resize((160, 160)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])](face) for face in face_crops]).to(DEVICE)

        gender_tensors = torch.stack([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ](face) for face in face_crops]).to(DEVICE)

        # Get predictions
        gender_probs = torch.softmax(models['gender_model'](gender_tensors), dim=1)
        face_logits = models['face_identifier'](face_tensors)
        face_preds = torch.argmax(face_logits, dim=1)
        face_probs = torch.softmax(face_logits, dim=1)

        # 3. Generate Results
        results = []
        for i, (face, gender_prob, face_pred, face_prob) in enumerate(
                zip(face_crops, gender_probs, face_preds, face_probs)):
        # Save face crop
            face_path = os.path.join(output_dir, f"face_{i}.jpg")
        face.save(face_path)

        # Get predictions
        gender = "male" if gender_prob[0] > GENDER_CONF_THRESH else "female"
        identity = models['class_names'][face_pred] if max(face_prob) > MIN_FACE_CONFIDENCE else "Unknown"

        results.append({
        'face_id': i,
        'image_path': face_path,
        'identity': identity,
        'confidence': round(max(face_prob).item(), 4),
        'gender': gender,
        'gender_confidence': round(max(gender_prob).item(), 4)
    })

    # 4. Create Attendance Log
    attendance = []
    for student in models['class_names']:
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
print(f"\nDetected {len(results)} faces:")
print(results_df[['face_id', 'identity', 'confidence']])

print(f"\nAttendance Summary:")
print(attendance_df)

print(f"\nResults saved to: {output_dir}")

return results_df, attendance_df


def main():
    """Main execution pipeline"""
    # 1. Setup
    dirs = setup_directories()

    # 2. Load and prepare LFW data
    lfw_data, selected_names = load_lfw_data()
    save_filtered_images(lfw_data, selected_names, dirs['data'])
    gender_labels = initialize_gender_labels(selected_names)

    # Save gender labels
    with open(os.path.join(dirs['data'], "gender_labels.json"), "w") as f:
        json.dump(gender_labels, f, indent=2)

    # 3. Face detection
    bbox_data = detect_faces(dirs['data'])
    with open(os.path.join(dirs['sim_output'], "bounding_boxes.json"), "w") as f:
        json.dump(bbox_data, f, indent=2)

    # 4. Create simulated dataset
    sim_config = {
        "face_base": dirs['data'],
        "bg_base": dirs['background'],
        "output_dir": dirs['sim_output'],
        "output_size": (640, 480),
        "num_images": 300,
        "faces_per_image": (3, 7),
        "apply_blur": True,
        "background_ext": ".jpg",
        "face_scale_range": (0.1, 0.3),
        "rotation_range": (-15, 15),
        "blur_chance": 0.3,
        "blur_radius_range": (0.5, 1.5),
        "female_ratio": 0.6,
        "gender_labels": gender_labels
    }

    annotations, total_female, total_male = create_simulated_dataset(sim_config)

    # Save annotations
    with open(os.path.join(dirs['sim_output'], "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nSimulated dataset created with {total_female} female and {total_male} male faces")

    # 5. Prepare YOLO dataset
    yaml_path = prepare_yolo_dataset(
        annotations,
        gender_labels,
        dirs['yolo_dataset']
    )

    # 6. Train YOLO model
    yolo_model = train_yolo_model(yaml_path)
    evaluate_yolo_model(yolo_model, os.path.join(dirs['yolo_dataset'], "images", "val"))

    # 7. Crop faces for gender classification
    crop_detected_faces(
        yolo_model,
        os.path.join(dirs['yolo_dataset'], "images", "val"),
        dirs['output_crop']
    )

    # 8. Prepare gender dataset
    for person in os.listdir(dirs['data']):
        person_path = os.path.join(dirs['data'], person)
        if not os.path.isdir(person_path) or person == "gender_dataset":
            continue

        gender = gender_labels.get(person)
        if gender not in ["male", "female"]:
            continue

        target_dir = os.path.join(dirs['gender_dataset'], gender)
        for img_name in os.listdir(person_path):
            src = os.path.join(person_path, img_name)
            dst = os.path.join(target_dir, f"{person}_{img_name}")
            if os.path.isfile(src):
                shutil.copy(src, dst)

    # 9. Train gender classifier
    gender_model = train_gender_classifier(dirs['gender_dataset'])

    # 10. Train face recognition
    face_model = train_face_recognition(dirs['data'])

    # 11. Load models for attendance system
    models = {
        'face_detector': yolo_model,
        'gender_model': gender_model,
        'face_identifier': face_model,
        'class_names': selected_names
    }

    # 12. Process sample image
    sample_image = os.path.join(dirs['sim_output'], "sim_000.jpg")
    if os.path.exists(sample_image):
        process_attendance(sample_image, models)
    else:
        print(f"Sample image not found at {sample_image}")


if __name__ == "__main__":
    main()