# SSD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib as plt
from datetime import datetime
from torchvision import transforms

# Load PASCAL VOC 2007 - both train and validation sets
voc_dataset = load_dataset("detection-datasets/pascal-voc", "detection-datasets--pascal-voc")

# Access train and validation splits
train_dataset = voc_dataset["train"]
val_dataset = voc_dataset["validation"]

# Define transforms for training data
train_transforms = A.Compose([
    A.RandomSizedBBoxSafeCrop(height=300, width=300, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RGBShift(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.Resize(height=300, width=300),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category']))

# Validation transforms with Albumentations
val_transforms = A.Compose([
    A.Resize(height=300, width=300),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category']))

# Function to convert Hugging Face dataset to PyTorch format
def convert_to_ssd_format(example, transform_fn):
    # Get original image
    img = np.array(example["image"])  # Albumentations expects numpy arrays
    
    # Get bounding boxes and class labels
    boxes = []
    categories = []
    
    for bbox, category in zip(example["objects"]["bbox"], example["objects"]["category"]):
        # Albumentations expects [x_min, y_min, x_max, y_max] format
        boxes.append(bbox)
        categories.append(category)
    
    # Apply transforms to image and bounding boxes
    transformed = transform_fn(image=img, bboxes=boxes, category=categories)
    
    # Extract transformed data
    image = transformed['image']  # Already a tensor from ToTensor
    transformed_boxes = transformed['bboxes']
    transformed_categories = transformed['category']
    
    # Convert to tensors (boxes already normalized by Albumentations)
    if transformed_boxes:  # Check if any boxes remain after transform
        boxes = torch.tensor(transformed_boxes, dtype=torch.float32)
        labels = torch.tensor(transformed_categories, dtype=torch.int64)
    else:
        # Handle case where all boxes were removed by transform
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.int64)
    
    return {
        "image": image,
        "boxes": boxes,
        "labels": labels
    }
# Create dataset mappers
def train_mapper(example):
    return convert_to_ssd_format(example, train_transforms)

def val_mapper(example):
    return convert_to_ssd_format(example, val_transforms)

# Define custom collate function to handle variable-sized boxes and labels
def custom_collate_fn(batch):
    images = []
    boxes = []
    labels = []
    
    for sample in batch:
        images.append(sample["image"])
        boxes.append(sample["boxes"])
        labels.append(sample["labels"])
    
    # Stack images into a single tensor
    images = torch.stack(images, 0)
    
    return {
        "images": images,
        "boxes": boxes,  # List of tensors with different shapes
        "labels": labels  # List of tensors with different shapes
    }

# Map the datasets
mapped_train_dataset = train_dataset.map(train_mapper)
mapped_val_dataset = val_dataset.map(val_mapper)

# Create DataLoaders
train_loader = DataLoader(
    mapped_train_dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=4,  # Using 4 parallel workers
    collate_fn=custom_collate_fn  
)

val_loader = DataLoader(
    mapped_val_dataset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=4,
    collate_fn=custom_collate_fn
)

# Define model 
class SSD(nn.Module):
    def __init__(self, num_classes=20):
        super(SSD, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)