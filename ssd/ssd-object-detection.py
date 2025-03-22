# SSD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
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
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # First feature map (38x38)
        self.conv1 = nn.Sequential(
            *features[:16],  # First 16 layers of VGG
            nn.BatchNorm2d(512)
        )

        # Second feature map (19x19)
        self.conv2 = nn.Sequential(
            *features[16:23],
            nn.BatchNorm2d(512)
        )

        # Additional convolution layers (10x10)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # (5x5)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # (3x3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # (1x1)
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Define anchor box configurations for each feature map
        self.feature_maps = [38, 19, 10, 5, 3, 1]  # sizes of feature maps
        self.steps = [8, 16, 32, 64, 100, 300]  # effective stride for each feature map
        self.scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]  # anchor box scales
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # aspect ratios for each feature map
        
        # Generate default boxes
        self.default_boxes = []
        for k, f in enumerate(self.feature_maps): # k=feature map index f=the size of the feature map
            for i in range(f):
                for j in range(f):
                    cx = (j + 0.5) / f # x coordinates
                    cy = (i + 0.5) / f # y coordinates
                    
                    # Aspect ratio: 1
                    s = self.scales[k]
                    self.default_boxes.append([cx, cy, s, s])
                    
                    # Additional scale for aspect ratio 1
                    s_prime = np.sqrt(s * self.scales[k + 1]) if k < len(self.feature_maps) - 1 else 1
                    self.default_boxes.append([cx, cy, s_prime, s_prime])
                    
                    # Other aspect ratios
                    for ar in self.aspect_ratios[k]:
                        self.default_boxes.extend([
                            [cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)],
                            [cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)]
                        ])
        
        self.default_boxes = torch.FloatTensor(self.default_boxes)  # Convert to tensor
        self.default_boxes.clamp_(0, 1)  # Clip to [0,1] 

        # Define location layers
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),  # For conv1
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),  # For conv2
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),  # For conv3
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),  # For conv4
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),  # For conv5
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)   # For conv6
        ])

        # Define confidence layers
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),  # For conv1
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),  # For conv2
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),  # For conv3
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),  # For conv4
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),  # For conv5
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)   # For conv6
        ])

    # Forward function
    def forward(self, x):
        # Extract feature maps
        feature_maps = []
        x = self.conv1(x)
        feature_maps.append(x)
        x = self.conv2(x)
        feature_maps.append(x)
        x = self.conv3(x)
        feature_maps.append(x)
        x = self.conv4(x)
        feature_maps.append(x)
        x = self.conv5(x)
        feature_maps.append(x)
        x = self.conv6(x)
        feature_maps.append(x)

        # Apply prediction layers
        loc_preds = []
        conf_preds = []
        for i, feature_map in enumerate(feature_maps):
            # Predicts bounding box ajustments
            loc_pred = self.loc_layers[i](feature_map)
            # Predicts class probabilites
            conf_pred = self.conf_layers[i](feature_map)
            loc_preds.append(loc_pred.permute(0, 2, 3, 1).contiguous())
            conf_preds.append(conf_pred.permute(0, 2, 3, 1).contiguous())

        # Reshape and concatenate predictions
        loc_preds = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf_preds = torch.cat([o.view(o.size(0), -1) for o in conf_preds], 1)

        return loc_preds, conf_preds

# Freeze first 10 layers of the VGG backbone
for idx, param in enumerate(model.conv1.parameters()):
    layer_idx = idx // 2  # Each layer has weights and biases, so divide by 2
    if layer_idx < 10:    # First 10 layers
        param.requires_grad = False