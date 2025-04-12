# SSD   
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import random
import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from torchvision.ops.boxes import box_convert, box_iou

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Pascal VOC class names
VOC_CLASSES = (
    'background',  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

num_classes = len(VOC_CLASSES)
print(f"Number of classes: {num_classes}")

# Define Pascal VOC Dataset with improved capabilities
class PascalVOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', transforms=None, use_mosaic=False, mosaic_prob=0.5):
        """
        Pascal VOC Dataset with enhanced augmentation
        
        Args:
            root (str): Path to VOCdevkit folder
            year (str): Dataset year ('2007' or '2012')
            image_set (str): Dataset type ('train', 'val', 'test')
            transforms (callable): Transforms to be applied
            use_mosaic (bool): Whether to use mosaic augmentation
            mosaic_prob (float): Probability of applying mosaic augmentation
        """
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transforms = transforms
        self.use_mosaic = use_mosaic and image_set == 'train'  # Only use mosaic for training
        self.mosaic_prob = mosaic_prob
        
        # Paths
        self.images_dir = os.path.join(root, f'VOC{year}', 'JPEGImages')
        self.annotations_dir = os.path.join(root, f'VOC{year}', 'Annotations')
        
        # Load image ids
        splits_dir = os.path.join(root, f'VOC{year}', 'ImageSets', 'Main')
        split_file = os.path.join(splits_dir, f'{image_set}.txt')
        
        with open(split_file, 'r') as f:
            self.ids = [x.strip() for x in f.readlines()]
        
        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}
    
    def __len__(self):
        return len(self.ids)
    
    def load_image_and_labels(self, index):
        """Load image and annotations for a given index"""
        img_id = self.ids[index]
        
        # Load image
        img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Load annotation
        anno_path = os.path.join(self.annotations_dir, f'{img_id}.xml')
        boxes, labels = self._parse_voc_xml(ET.parse(anno_path).getroot())
        
        return img, boxes, labels
    
    def __getitem__(self, index):
        # Decide whether to use mosaic
        if self.use_mosaic and random.random() < self.mosaic_prob:
            img, boxes, labels = self._load_mosaic(index)
        else:
            img, boxes, labels = self.load_image_and_labels(index)
        
        sample = {
            'image': img,
            'bboxes': boxes,
            'labels': labels
        }
        
        # Apply transformations
        if self.transforms:
            sample = self.transforms(**sample)
            
        return {
            'images': sample['image'],
            'boxes': torch.FloatTensor(sample['bboxes']) if len(sample['bboxes']) > 0 else torch.zeros((0, 4)),
            'labels': torch.LongTensor(sample['labels']) if len(sample['labels']) > 0 else torch.zeros(0, dtype=torch.long)
        }
    
    def _load_mosaic(self, index):
        """
        Loads 4 images and combines them in a mosaic pattern
        """
        # Get 3 more random indexes
        indices = [index] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]
        
        # Final image will be 2x the size before resizing
        img_size = 1024  # This will later be resized to target size
        
        # Center of the mosaic
        cx, cy = img_size // 2, img_size // 2
        
        # Initialize mosaic image and annotation lists
        mosaic_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []
        
        # Mosaic quadrants
        positions = [
            [0, 0, cx, cy],          # top-left
            [cx, 0, img_size, cy],   # top-right
            [0, cy, cx, img_size],   # bottom-left
            [cx, cy, img_size, img_size]  # bottom-right
        ]
        
        for i, idx in enumerate(indices):
            img, boxes, labels = self.load_image_and_labels(idx)
            h, w = img.shape[:2]
            
            # Place image in mosaic
            x1a, y1a, x2a, y2a = positions[i]  # mosaic position
            
            # Resize image to fit the quadrant
            h_scale, w_scale = (y2a - y1a) / h, (x2a - x1a) / w
            img_resized = cv2.resize(img, (x2a - x1a, y2a - y1a))
            mosaic_img[y1a:y2a, x1a:x2a] = img_resized
            
            # Adjust box coordinates
            if len(boxes) > 0:
                # Scale box coordinates
                boxes_scaled = boxes.copy()
                boxes_scaled[:, 0] = w_scale * boxes[:, 0] + x1a
                boxes_scaled[:, 1] = h_scale * boxes[:, 1] + y1a
                boxes_scaled[:, 2] = w_scale * boxes[:, 2] + x1a
                boxes_scaled[:, 3] = h_scale * boxes[:, 3] + y1a
                
                # Add boxes and labels
                mosaic_boxes.extend(boxes_scaled)
                mosaic_labels.extend(labels)
        
        # Clip boxes to image boundaries
        if len(mosaic_boxes) > 0:
            mosaic_boxes = np.array(mosaic_boxes)
            mosaic_boxes[:, 0] = np.clip(mosaic_boxes[:, 0], 0, img_size)
            mosaic_boxes[:, 1] = np.clip(mosaic_boxes[:, 1], 0, img_size)
            mosaic_boxes[:, 2] = np.clip(mosaic_boxes[:, 2], 0, img_size)
            mosaic_boxes[:, 3] = np.clip(mosaic_boxes[:, 3], 0, img_size)
            
            # Filter out invalid boxes
            valid_boxes = []
            valid_labels = []
            
            for box, label in zip(mosaic_boxes, mosaic_labels):
                if box[2] > box[0] and box[3] > box[1]:  # width and height > 0
                    valid_boxes.append(box)
                    valid_labels.append(label)
            
            mosaic_boxes = valid_boxes
            mosaic_labels = valid_labels
        
        return mosaic_img, mosaic_boxes, mosaic_labels
    
    def _parse_voc_xml(self, node):
        """Parse Pascal VOC annotation XML file"""
        boxes = []
        labels = []
        
        for obj in node.findall('object'):
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue
                
            label = self.class_to_idx[name]
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin or xmax <= 0 or ymax <= 0:
                continue
                
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        return boxes, labels

# Import required for mosaic augmentation
import cv2

# Define enhanced transforms with resolution 512x512
train_transforms = A.Compose([
    # Spatial transformations
    A.Resize(height=512, width=512),
    
    # Flips and rotations
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.3),
    
    # Color augmentations - enhanced
    A.OneOf([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    ], p=0.7),
    
    # Noise and blur - more aggressive
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.3),
    
    # Weather simulation
    A.OneOf([
        A.RandomShadow(p=1.0),
        A.RandomRain(drop_length=8, blur_value=3, p=1.0),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
    ], p=0.2),
    
    # Cutout/erasing
    A.CoarseDropout(max_holes=8, max_height=64, max_width=64, min_holes=1, 
                   min_height=32, min_width=32, fill_value=0, p=0.2),
    
    # Normalize and convert to tensor
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))

# Validation transformations
val_transforms = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Custom collate function to handle variable size boxes and labels
def custom_collate_fn(batch):
    images = torch.stack([item['images'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels
    }

# Create datasets with mosaic augmentation
voc_root = '/home/adrian/ml-rover/VOCdevkit/VOCdevkit'

# For 2007 dataset
train_dataset = PascalVOCDataset(voc_root, year='2007', image_set='train', 
                                transforms=train_transforms, use_mosaic=True, mosaic_prob=0.5)
val_dataset = PascalVOCDataset(voc_root, year='2007', image_set='val', 
                              transforms=val_transforms)
test_dataset = PascalVOCDataset(voc_root, year='2007', image_set='test', 
                               transforms=val_transforms)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=32,
    shuffle=True, 
    num_workers=4,
    collate_fn=custom_collate_fn,
    pin_memory=True
)   

val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=4,
    collate_fn=custom_collate_fn,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=4,
    collate_fn=custom_collate_fn,
    pin_memory=True
)

# Define FPN module for feature pyramid
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        
        # Lateral connections (1x1 convolutions for channel reduction)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features):
        """
        Forward function for FPN
        
        Args:
            features: List of feature maps [P3, P4, P5, P6, P7]
            
        Returns:
            List of output feature maps with same resolution but unified channels
        """
        results = []
        # Last feature map doesn't need upsampling
        last_inner = self.lateral_convs[-1](features[-1])
        results.append(self.output_convs[-1](last_inner))
        
        # Process other feature maps from bottom up
        for idx in range(len(features) - 2, -1, -1):
            # Inner features - lateral connection + upsampled features
            higher_res_features = self.lateral_convs[idx](features[idx])
            
            # Upsample previous results
            inner_top_down = nn.functional.interpolate(
                last_inner, size=higher_res_features.shape[-2:], 
                mode='nearest'
            )
            
            # Add features
            last_inner = higher_res_features + inner_top_down
            results.insert(0, self.output_convs[idx](last_inner))
            
        return results

# Define SSD model with EfficientNet backbone and FPN
class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        # Store num_classes as a class attribute
        self.num_classes = num_classes
        
        # Load EfficientNet-B1 with pretrained weights
        # Use pretrained ImageNet weights
        self.backbone = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        
        # Get the feature maps we need from EfficientNet-B1
        # Extract intermediate feature maps from EfficientNet
        self.feature_extractors = nn.ModuleList([
            # Extract first 2 blocks (P2)
            nn.Sequential(*list(self.backbone.features)[:2]),
            # Extract middle blocks (P3)
            nn.Sequential(*list(self.backbone.features)[2:3]),
            # Extract later blocks (P4)
            nn.Sequential(*list(self.backbone.features)[3:5]),
            # Extract deep features (P5)
            nn.Sequential(*list(self.backbone.features)[5:7]),
            # Extract deepest features (P6)
            nn.Sequential(*list(self.backbone.features)[7:])
        ])
        
        # Extract channel dimensions from EfficientNet blocks
        # EfficientNet-B1 feature dimensions: [32, 24, 40, 80, 112, 192, 320]
        self.feature_channels = [24, 40, 80, 192, 320]
        
        # Additional layers for deeper feature maps
        self.extra_layer1 = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

        self.extra_layer2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

        # Add FPN for feature fusion
        self.fpn = FPN(
            in_channels_list=[self.feature_channels[1], self.feature_channels[2], 
                            self.feature_channels[4], 512, 256],
            out_channels=256
        )
        
        # Define feature map sizes based on input size 512x512
        # These will be different due to EfficientNet architecture compared to VGG
        # These sizes are approximate for 512x512 input
        self.feature_maps = [64, 32, 16, 8, 4, 2]
        self.steps = [8, 16, 32, 64, 128, 256]  # effective stride for each feature map
        self.scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]  # anchor box scales
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # aspect ratios for each feature map
        
        # Calculate number of boxes per feature map cell
        self.num_anchors = []
        for ar in self.aspect_ratios:
            # 1 + extra scale for aspect ratio 1 + 2 for each additional aspect ratio
            self.num_anchors.append(2 + 2 * len(ar))
        
        # Define location layers with
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(256, self.num_anchors[0] * 4, kernel_size=3, padding=1),  # For first feature map
            nn.Conv2d(256, self.num_anchors[1] * 4, kernel_size=3, padding=1),  # For second feature map
            nn.Conv2d(256, self.num_anchors[2] * 4, kernel_size=3, padding=1),  # For third feature map
            nn.Conv2d(256, self.num_anchors[3] * 4, kernel_size=3, padding=1),  # For fourth feature map
            nn.Conv2d(256, self.num_anchors[4] * 4, kernel_size=3, padding=1),  # For fifth feature map
            nn.Conv2d(256, self.num_anchors[5] * 4, kernel_size=3, padding=1)   # For sixth feature map
        ])

        # Define confidence layers
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(256, self.num_anchors[0] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[1] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[2] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[3] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[4] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[5] * num_classes, kernel_size=3, padding=1)
        ])
        
        # Generate default boxes
        self._create_default_boxes()
        
    def _create_default_boxes(self):
        """Generate default (anchor) boxes for all feature map cells"""
        default_boxes = []
        
        # For each feature map
        for k, f in enumerate(self.feature_maps):
            # For each cell in the feature map
            for i in range(f):
                for j in range(f):
                    # Center of the cell (normalized coordinates)
                    cx = (j + 0.5) / f
                    cy = (i + 0.5) / f
                    
                    # Aspect ratio: 1
                    s = self.scales[k]
                    default_boxes.append([cx, cy, s, s])
                    
                    # Additional scale for aspect ratio 1
                    if k < len(self.feature_maps) - 1:
                        s_prime = np.sqrt(s * self.scales[k + 1])
                        default_boxes.append([cx, cy, s_prime, s_prime])
                    else:
                        s_prime = 1.0
                        default_boxes.append([cx, cy, s_prime, s_prime])
                    
                    # Other aspect ratios
                    for ar in self.aspect_ratios[k]:
                        default_boxes.append([cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)])
                        default_boxes.append([cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)])
        
        self.default_boxes = torch.FloatTensor(default_boxes)
        
        # Convert default boxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
        self.default_boxes_xyxy = self._center_to_corner(self.default_boxes)
        self.default_boxes.clamp_(0, 1)
        self.default_boxes_xyxy.clamp_(0, 1)

    def _center_to_corner(self, boxes):
        """Convert boxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)"""
        corner_boxes = boxes.clone()
        corner_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # xmin = cx - w/2
        corner_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # ymin = cy - h/2
        corner_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # xmax = cx + w/2
        corner_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # ymax = cy + h/2
        return corner_boxes

    # Forward function
    def forward(self, x):
        # Extract features from EfficientNet backbone
        features = []
        for extractor in self.feature_extractors:
            x = extractor(x)
            features.append(x)
        
        # Get P3, P4, P5
        selected_features = [features[1], features[2], features[4]]
        
        # Add extra layers for P6, P7
        p6 = self.extra_layer1(features[4])
        p7 = self.extra_layer2(p6)
        
        # Combine to get all features for FPN
        fpn_input = [selected_features[0], selected_features[1], selected_features[2], p6, p7]
        
        # Apply FPN to get unified feature maps
        fpn_features = self.fpn(fpn_input)
        
        # Apply prediction layers
        loc_preds = []
        conf_preds = []
        
        for i, (feature, loc_layer, conf_layer) in enumerate(zip(fpn_features, self.loc_layers, self.conf_layers)):
            # Get location predictions
            loc = loc_layer(feature)
            batch_size = loc.size(0)
            # Reshape to [batch_size, height, width, num_anchors * 4] then flatten to [batch_size, num_anchors_total, 4]
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(batch_size, -1, 4)
            loc_preds.append(loc)
            
            # Get confidence predictions
            conf = conf_layer(feature)
            # Reshape to [batch_size, height, width, num_anchors * num_classes] then flatten
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(batch_size, -1, self.num_classes)
            conf_preds.append(conf)

        # Concatenate predictions from different feature maps
        loc_preds = torch.cat(loc_preds, dim=1)
        conf_preds = torch.cat(conf_preds, dim=1)
        
        return loc_preds, conf_preds, self.default_boxes_xyxy

# Improved loss function with GIoU
def giou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Calculates the Generalized IoU loss
    
    Args:
        pred_boxes: (tensor) Predicted boxes, sized [N, 4]
        target_boxes: (tensor) Target boxes, sized [N, 4]
        
    Returns:
        GIoU loss value
    """
    # Calculate intersection area
    inter_min = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    inter_max = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
    inter_wh = torch.clamp(inter_max - inter_min, min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    
    # Calculate union area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = pred_area + target_area - inter
    
    # Calculate IoU
    iou = inter / (union + eps)
    
    # Calculate smallest enclosing box
    encl_min = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
    encl_max = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
    encl_wh = torch.max(encl_max - encl_min, torch.zeros_like(encl_max))
    encl_area = encl_wh[:, 0] * encl_wh[:, 1]
    
    # Calculate GIoU
    giou = iou - (encl_area - union) / (encl_area + eps)
    
    # GIoU Loss (1 - GIoU)
    loss = 1 - giou
    
    return loss

# Helper functions for box encoding/decoding
def box_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) between two sets of boxes.
    
    Args:
        boxes1 (torch.Tensor): Shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes2 (torch.Tensor): Shape (M, 4) in format (xmin, ymin, xmax, ymax)
    
    Returns:
        torch.Tensor: IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    return inter / union

def encode_boxes(matched_boxes, default_boxes):
    """
    Encode ground truth boxes relative to default boxes (anchor boxes).
    
    Args:
        matched_boxes (torch.Tensor): Ground truth boxes (N, 4) in corner format
        default_boxes (torch.Tensor): Default anchor boxes (N, 4) in corner format
    
    Returns:
        torch.Tensor: Encoded box locations (as used in the SSD paper)
    """
    # Convert from corner to center format
    def corner_to_center(boxes):
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        cx = boxes[:, 0] + width / 2
        cy = boxes[:, 1] + height / 2
        return torch.stack([cx, cy, width, height], dim=1)
    
    # Get centers, widths and heights
    g_boxes = corner_to_center(matched_boxes)
    d_boxes = corner_to_center(default_boxes)
    
    # Encode according to SSD paper
    encoded_boxes = torch.zeros_like(g_boxes)
    # (gx - dx) / dw -> cx
    encoded_boxes[:, 0] = (g_boxes[:, 0] - d_boxes[:, 0]) / (d_boxes[:, 2] + 1e-8)
    # (gy - dy) / dh -> cy
    encoded_boxes[:, 1] = (g_boxes[:, 1] - d_boxes[:, 1]) / (d_boxes[:, 3] + 1e-8)
    # log(gw / dw) -> width
    encoded_boxes[:, 2] = torch.log(g_boxes[:, 2] / (d_boxes[:, 2] + 1e-8) + 1e-8)
    # log(gh / dh) -> height
    encoded_boxes[:, 3] = torch.log(g_boxes[:, 3] / (d_boxes[:, 3] + 1e-8) + 1e-8)
    
    return encoded_boxes

# Improved SSD loss with GIoU
class SSD_loss(nn.Module):
    def __init__(self, num_classes, default_boxes, device, alpha=0.5, gamma=1.5):
        """
        Improved SSD Loss function with GIoU and Focal Loss

        Args:
            num_classes (int): Number of object classes
            default_boxes (torch.Tensor): Default anchor boxes (in corner format)
            device (torch.device): GPU or CPU
            alpha (float): Weighting factor in focal loss
            gamma (float): Focusing parameter in focal loss
        """
        super(SSD_loss, self).__init__()
        
        self.num_classes = num_classes
        self.default_boxes = default_boxes.to(device)
        self.device = device
        
        # Focal loss parameters
        self.alpha = alpha
        self.gamma = gamma
        
        self.threshold = 0.5  # IoU threshold for positive matches
        self.neg_pos_ratio = 3  # Ratio of negative to positive samples
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def focal_loss(self, pred, target):
        """
        Compute focal loss for classification
        
        Args:
            pred (torch.Tensor): Predicted class scores [N, num_classes]
            target (torch.Tensor): Target classes [N]
            
        Returns:
            torch.Tensor: Focal loss values
        """
        # Compute standard cross entropy loss
        ce_loss = self.cross_entropy(pred, target)
        
        # Get probability of the target class (p_t)
        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        p_t = pred_softmax.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Calculate focal weight: alpha * (1-p_t)^gamma
        focal_weight = self.alpha * (1 - p_t).pow(self.gamma)
        
        # Apply weight to cross entropy loss
        focal_loss = focal_weight * ce_loss
        
        return focal_loss
    
    def forward(self, predictions, targets):
        """
        Compute SSD loss with GIoU for localization and focal loss for classification.

        Args:
            predictions (tuple): (loc_preds, conf_preds, default_boxes)
                - loc_preds: Shape (batch_size, num_priors, 4)
                - conf_preds: Shape (batch_size, num_priors, num_classes)
                - default_boxes: Default boxes used in the model
            targets (dict): {"boxes": [list of GT boxes], "labels": [list of GT labels]}
        
        Returns:
            torch.Tensor: Total loss (scalar)
        """
        loc_preds, conf_preds, _ = predictions
        batch_size = loc_preds.size(0)
        num_priors = self.default_boxes.size(0)
        
        # Create empty tensors for the targets
        loc_t = torch.zeros(batch_size, num_priors, 4).to(self.device)
        conf_t = torch.zeros(batch_size, num_priors, dtype=torch.long).to(self.device)
        
        # For each image in the batch
        for idx in range(batch_size):
            truths = targets['boxes'][idx]  # Ground truth boxes
            labels = targets['labels'][idx]  # Ground truth labels
            
            if truths.size(0) == 0:  # Skip if no ground truth boxes
                continue
            
            # Calculate IoU between default boxes and ground truth boxes
            overlaps = box_iou(self.default_boxes, truths)
            
            # For each default box, find the best matching ground truth box
            best_truth_overlap, best_truth_idx = overlaps.max(1)
            
            # For each ground truth box, find the best matching default box
            best_prior_overlap, best_prior_idx = overlaps.max(0)
            # Make sure each ground truth box has at least one matching default box
            for j in range(best_prior_idx.size(0)):
                best_truth_idx[best_prior_idx[j]] = j
                best_truth_overlap[best_prior_idx[j]] = 2.0  # Ensure it's greater than threshold
            
            # Get matched ground truth boxes and labels
            matches = truths[best_truth_idx]
            match_labels = labels[best_truth_idx]
            
            # Set background label (0) for boxes with low overlap
            match_labels[best_truth_overlap < self.threshold] = 0
            
            # Encode the ground truth boxes relative to default boxes
            loc_t[idx] = encode_boxes(matches, self.default_boxes)
            conf_t[idx] = match_labels
        
        # Compute positive mask (where gt label > 0)
        pos = conf_t > 0
        num_pos = pos.sum().item()
        
        # Skip loss calculation if there are no positive examples
        if num_pos == 0:
            return torch.tensor(0.0).to(self.device)
        
        # Localization loss with GIoU loss
        # First, decode predicted boxes
        pos_idx = pos.unsqueeze(2).expand_as(loc_preds)
        
        # Get positive predictions and targets
        pos_loc_preds = loc_preds[pos_idx].view(-1, 4)
        pos_loc_targets = loc_t[pos_idx].view(-1, 4)
        
        # Decode predictions to get actual boxes
        decoded_boxes = decode_boxes(pos_loc_preds, self.default_boxes.repeat(batch_size, 1, 1)[pos_idx].view(-1, 4))
        decoded_targets = decode_boxes(pos_loc_targets, self.default_boxes.repeat(batch_size, 1, 1)[pos_idx].view(-1, 4))
        
        # Compute GIoU loss
        loc_loss = giou_loss(decoded_boxes, decoded_targets).mean()
        
        # Confidence loss with focal loss
        # Reshape confidence predictions to [batch_size * num_priors, num_classes]
        batch_conf = conf_preds.view(-1, self.num_classes)
        
        # Use focal loss for all examples
        # No need for hard negative mining since focal loss naturally handles class imbalance
        conf_loss = self.focal_loss(batch_conf, conf_t.view(-1))
        
        # Sum loss values to get a scalar
        conf_loss = conf_loss.sum()
        
        # Normalize by number of positive examples
        pos_count = max(1, num_pos)  # Avoid division by zero
        conf_loss /= pos_count
        
        # Return scalar total loss - balance localization and classification loss
        total_loss = loc_loss + conf_loss
        return total_loss

def decode_boxes(loc, default_boxes):
    """Decode predicted box coordinates from offsets"""
    # Ensure both tensors are on the same device
    device = loc.device
    default_boxes = default_boxes.to(device)
    
    # Convert default boxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h)
    def corner_to_center(boxes):
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        cx = boxes[:, 0] + width / 2
        cy = boxes[:, 1] + height / 2
        return torch.stack([cx, cy, width, height], dim=1)
    
    # Convert default boxes to center format
    default_boxes_center = corner_to_center(default_boxes)
    
    # Decode predictions
    pred_cx = loc[:, 0] * default_boxes_center[:, 2] + default_boxes_center[:, 0]
    pred_cy = loc[:, 1] * default_boxes_center[:, 3] + default_boxes_center[:, 1]
    pred_w = torch.exp(loc[:, 2]) * default_boxes_center[:, 2]
    pred_h = torch.exp(loc[:, 3]) * default_boxes_center[:, 3]
    
    # Convert back to corner format
    boxes = torch.zeros_like(loc)
    boxes[:, 0] = pred_cx - pred_w / 2
    boxes[:, 1] = pred_cy - pred_h / 2
    boxes[:, 2] = pred_cx + pred_w / 2
    boxes[:, 3] = pred_cy + pred_h / 2
    
    return boxes

# Instantiate the SSD model and send it to the GPU
model = SSD(num_classes=num_classes)
model.to(device)

# Initialize the SSD loss function
SSDLoss = SSD_loss(
    num_classes=num_classes, 
    default_boxes=model.default_boxes_xyxy,  # Use boxes in corner format
    device=device
)

# Freeze first 10 layers of the VGG backbone for faster training
frozen_layers = 10
layer_count = 0
for param in model.conv1.parameters():
    if layer_count < frozen_layers * 2:  # Each layer has weights and biases
        param.requires_grad = False
    layer_count += 1

# Define optimizer with learning rate scheduling and weight decay
optimizer = optim.AdamW([
    {'params': [p for p in model.conv1.parameters() if p.requires_grad], 'lr': 0.00005},
    {'params': model.conv2.parameters(), 'lr': 0.0002},
    {'params': model.conv3.parameters(), 'lr': 0.0005},
    {'params': model.conv4.parameters(), 'lr': 0.0005},
    {'params': model.conv5.parameters(), 'lr': 0.0005},
    {'params': model.conv6.parameters(), 'lr': 0.0005},
    {'params': model.loc_layers.parameters(), 'lr': 0.0005},
    {'params': model.conf_layers.parameters(), 'lr': 0.0005}
], lr=0.0005, weight_decay=1e-4)

# Learning rate scheduler with warmup
def get_lr_scheduler(optimizer, warmup_epochs=5, max_epochs=50):
    # Define warmup scheduler
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Cosine annealing scheduler
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs-warmup_epochs, eta_min=1e-6
    )
    
    return warmup_scheduler, cosine_scheduler

# Get schedulers
warmup_scheduler, cosine_scheduler = get_lr_scheduler(optimizer)

# Training loop
def train_model(model, loss_fn, optimizer, warmup_scheduler, lr_scheduler, 
                train_loader, val_dataset, epochs, warmup_epochs=5, checkpoint_dir='./checkpoints'):
    """
    Args:
        model: SSD model
        loss_fn: Loss function
        optimizer: Optimizer
        warmup_scheduler: Learning rate scheduler for warmup phase
        lr_scheduler: Main learning rate scheduler
        train_loader: Training data loader
        val_dataset: Validation dataset
        epochs: Number of epochs to train for
        warmup_epochs: Number of warmup epochs
        checkpoint_dir: Directory to save checkpoints
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize lists for tracking metrics
    train_losses = []
    val_losses = []
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Start training
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        t_start = datetime.now()
        print(f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['images'].to(device)
            boxes = batch['boxes']
            labels = batch['labels']
            
            # Skip batches with no ground truth boxes
            if all(b.size(0) == 0 for b in boxes):
                continue
                
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                loc_preds, conf_preds, default_boxes = model(images)
                
                # Compute loss
                loss = loss_fn((loc_preds, conf_preds, default_boxes), {
                    'boxes': [b.to(device) for b in boxes],
                    'labels': [l.to(device) for l in labels]
                })
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            
            # Gradient clipping with scaling
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Update weights with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Print batch progress
            if (batch_idx + 1) % 20 == 0 or batch_idx + 1 == len(train_loader):
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Update learning rate schedulers
        if epoch < warmup_epochs:
            # During warmup phase, use only warmup scheduler
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Warmup phase: learning rate set to {current_lr:.6f}")
        else:
            # After warmup, use cosine annealing scheduler
            lr_scheduler.step()  # Using lr_scheduler from function argument
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Cosine phase: learning rate set to {current_lr:.6f}")

        # Calculate average training loss
        avg_train_loss = epoch_loss / max(1, batch_count)
        train_losses.append(avg_train_loss)

        # Validation phase (only every few epochs)
        val_epoch = (epoch + 1) % 2 == 0 or epoch < 2 or epoch >= epochs - 3
        
        if val_epoch:
            # Validation phase for loss calculation
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            # Create a validation loader with no multiprocessing
            safe_val_loader = DataLoader(
                val_dataset,
                batch_size=4,
                shuffle=False, 
                num_workers=0,  # No multiprocessing
                collate_fn=custom_collate_fn
            )
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(safe_val_loader):
                    # Move data to device
                    images = batch['images'].to(device)
                    boxes = batch['boxes']
                    labels = batch['labels']
                    
                    # Skip batches with no ground truth boxes
                    if all(b.size(0) == 0 for b in boxes):
                        continue
                    
                    # Forward pass
                    loc_preds, conf_preds, default_boxes = model(images)
                    
                    # Compute loss
                    loss = loss_fn((loc_preds, conf_preds, default_boxes), {
                        'boxes': [b.to(device) for b in boxes],
                        'labels': [l.to(device) for l in labels]
                    })
                    
                    # Track metrics
                    val_loss += loss.item()
                    val_batch_count += 1
                    
                    # Limit validation to speed up training
                    if batch_idx >= 25:  # Check ~100 validation images
                        break
            
            # Calculate average validation loss
            avg_val_loss = val_loss / max(1, val_batch_count)
            val_losses.append(avg_val_loss)
            
            # Update main learning rate scheduler based on validation loss
            lr_scheduler.step(avg_val_loss)
            
            # Print epoch summary
            t_end = datetime.now()
            duration = t_end - t_start
            print(f"  Epoch {epoch+1} completed in {duration}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint if this is the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                print(f"  Saving best model with validation loss: {best_val_loss:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                print(f"  No improvement in validation loss for {patience_counter} epochs")
                
                # Save periodic checkpoint
                if (epoch + 1) % 5 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            # Skip validation this epoch
            print(f"  Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} (Validation skipped)")
            t_end = datetime.now()
            print(f"  Completed in {t_end - t_start}")
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_losses[-1] if val_losses else float('inf'),
    }, os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # Return training history
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }

# Function to plot training history
def plot_training_history(history):
    """Plot training and validation metrics without mAP"""
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation losses
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Function to visualize detections
def visualize_detections(model, image_path, transform=None, conf_threshold=0.5):
    """
    Visualize object detections on an image
    
    Args:
        model: Trained SSD model
        image_path: Path to the image
        transform: Transforms to apply to the image
        conf_threshold: Confidence threshold for detections
    """
    # Set model to evaluation mode
    model.eval()
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    orig_image = np.array(image)
    
    # Apply transforms if provided
    if transform:
        transformed = transform(image=np.array(image))
        image_tensor = transformed['image']
    else:
        # Default transform
        transform = A.Compose([
            A.Resize(height=300, width=300),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        transformed = transform(image=np.array(image))
        image_tensor = transformed['image']
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        loc_preds, conf_preds, default_boxes = model(image_tensor)
        
        # Ensure default_boxes is on the same device
        default_boxes = default_boxes.to(device)
        
        # Decode predictions
        decoded_boxes = decode_boxes(loc_preds[0], default_boxes)
        
        # Get confidence scores
        scores = torch.nn.functional.softmax(conf_preds[0], dim=1)
        
        # Get detections with confidence > threshold
        detections = []
        
        for class_idx in range(1, model.num_classes):  # Skip background class
            class_scores = scores[:, class_idx]
            mask = class_scores > conf_threshold
            
            if mask.sum() == 0:
                continue
                
            class_boxes = decoded_boxes[mask]
            class_scores = class_scores[mask]
            
            # Apply non-maximum suppression
            keep_idx = torchvision.ops.nms(class_boxes, class_scores, iou_threshold=0.45)
            
            for idx in keep_idx:
                detections.append({
                    'box': class_boxes[idx].cpu().numpy(),
                    'score': class_scores[idx].item(),
                    'class': class_idx
                })
    
    # Visualize detections
    plt.figure(figsize=(12, 8))
    plt.imshow(orig_image)
    
    # Plot each detection
    for det in detections:
        box = det['box']
        score = det['score']
        class_idx = det['class']
        
        # Scale box coordinates to original image size
        h, w, _ = orig_image.shape
        x1, y1, x2, y2 = box
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        
        # Add label
        class_name = VOC_CLASSES[class_idx]
        plt.text(x1, y1-5, f'{class_name}: {score:.2f}', 
                 fontsize=10, color='white', 
                 bbox=dict(facecolor='red', alpha=0.7))
    
    plt.title('Object Detections')
    plt.axis('off')
    plt.savefig('detections.png')
    plt.show()

# Updated main training function call
print("Starting model training...")
history = train_model(
    model=model,
    loss_fn=SSDLoss,
    optimizer=optimizer,
    warmup_scheduler=warmup_scheduler,
    lr_scheduler=cosine_scheduler,
    train_loader=train_loader,
    val_dataset=val_dataset, 
    epochs=100,
    warmup_epochs=5
)

# Plot training history
plot_training_history(history)

# Load the best model
best_model_path = os.path.join('checkpoints', 'best_model.pth')
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model with validation loss: {checkpoint['val_loss']:.4f}")
else:
    print("No best model checkpoint found. Using final trained model.")

# Save the final model in a format suitable for inference
final_model_path = 'ssd_pascal_voc_final.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'class_names': VOC_CLASSES
}, final_model_path)

print(f"Final model saved to {final_model_path}")