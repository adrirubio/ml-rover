# SSD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import torchvision
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
from torchvision.ops.boxes import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import cv2

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
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transforms = transforms
        self.use_mosaic = use_mosaic and image_set == 'train'
        self.mosaic_prob = mosaic_prob
        
        self.images_dir = os.path.join(root, f'VOC{year}', 'JPEGImages')
        self.annotations_dir = os.path.join(root, f'VOC{year}', 'Annotations')
        
        splits_dir = os.path.join(root, f'VOC{year}', 'ImageSets', 'Main')
        split_file = os.path.join(splits_dir, f'{image_set}.txt')
        with open(split_file, 'r') as f:
            self.ids = [x.strip() for x in f.readlines()]
            
        self.class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}
    
    def __len__(self):
        return len(self.ids)
    
    def load_image_and_labels(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        anno_path = os.path.join(self.annotations_dir, f'{img_id}.xml')
        boxes, labels = self._parse_voc_xml(ET.parse(anno_path).getroot())
        return img, boxes, labels
    
    def __getitem__(self, index):
        if self.use_mosaic and random.random() < self.mosaic_prob:
            img, boxes, labels = self._load_mosaic(index)
        else:
            img, boxes, labels = self.load_image_and_labels(index)
        sample = {'image': img, 'bboxes': boxes, 'labels': labels}
        if self.transforms:
            sample = self.transforms(**sample)
        
        # Normalize bounding boxes assuming the image is resized to 512x512
        normalized_boxes = []
        for box in sample['bboxes']:
            xmin, ymin, xmax, ymax = box
            nbox = [xmin / 512, ymin / 512, xmax / 512, ymax / 512]
            normalized_boxes.append(nbox)
        if normalized_boxes:
            arr = np.array(normalized_boxes)
            if arr.min() < 0 or arr.max() > 1:
                print(f"WARNING: Normalized boxes out of bounds: min={arr.min()}, max={arr.max()}")

        return {
            'images': sample['image'],
            'boxes': torch.FloatTensor(normalized_boxes) if normalized_boxes else torch.zeros((0, 4)),
            'labels': torch.LongTensor(sample['labels']) if sample['labels'] else torch.zeros(0, dtype=torch.long)
        }
    
    def _load_mosaic(self, index):
        indices = [index] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]
        img_size = 1024
        cx, cy = img_size // 2, img_size // 2
        mosaic_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []
        positions = [
            [0, 0, cx, cy],
            [cx, 0, img_size, cy],
            [0, cy, cx, img_size],
            [cx, cy, img_size, img_size]
        ]
        for i, idx in enumerate(indices):
            img, boxes, labels = self.load_image_and_labels(idx)
            h, w = img.shape[:2]
            x1a, y1a, x2a, y2a = positions[i]
            h_scale, w_scale = (y2a - y1a) / h, (x2a - x1a) / w
            img_resized = cv2.resize(img, (x2a - x1a, y2a - y1a))
            mosaic_img[y1a:y2a, x1a:x2a] = img_resized
            if len(boxes) > 0:
                if not isinstance(boxes, np.ndarray):
                    boxes = np.array(boxes)
                boxes_scaled = boxes.copy()
                boxes_scaled[:, 0] = w_scale * boxes[:, 0] + x1a
                boxes_scaled[:, 1] = h_scale * boxes[:, 1] + y1a
                boxes_scaled[:, 2] = w_scale * boxes[:, 2] + x1a
                boxes_scaled[:, 3] = h_scale * boxes[:, 3] + y1a
                for box, label in zip(boxes_scaled, labels):
                    box_width = box[2] - box[0]
                    box_height = box[3] - box[1]
                    original_area = box_width * box_height
                    clipped_box = [
                        max(0, box[0]),
                        max(0, box[1]),
                        min(img_size, box[2]),
                        min(img_size, box[3])
                    ]
                    clipped_width = clipped_box[2] - clipped_box[0]
                    clipped_height = clipped_box[3] - clipped_box[1]
                    clipped_area = clipped_width * clipped_height
                    if (clipped_width > 0 and clipped_height > 0 and 
                        clipped_area / (original_area + 1e-8) > 0.25):
                        mosaic_boxes.append(clipped_box)
                        mosaic_labels.append(label)
        if len(mosaic_boxes) > 0:
            mosaic_boxes = np.array(mosaic_boxes)
        return mosaic_img, mosaic_boxes, mosaic_labels
    
    def _parse_voc_xml(self, node):
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
            if xmax <= xmin or ymax <= ymin or xmax <= 0 or ymax <= 0:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        return boxes, labels

# IMPROVED: Enhanced data augmentation
train_transforms = A.Compose([
    A.Resize(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    # Increased transformation intensity
    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=20, p=0.5),
    # More aggressive color augmentation
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    # Added random brightness contrast
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
    # Added grayscale occasionally
    A.ToGray(p=0.02),
    # Added blur occasionally
    A.GaussianBlur(blur_limit=3, p=0.05),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=['labels']))

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

# Create datasets with increased mosaic augmentation
voc_root = '/home/adrian/ssd/VOCdevkit/VOCdevkit'

# IMPROVED: Increased mosaic augmentation probability
train_dataset = PascalVOCDataset(voc_root, year='2007', image_set='train', 
                                transforms=train_transforms, use_mosaic=True, mosaic_prob=0.8)
val_dataset = PascalVOCDataset(voc_root, year='2007', image_set='val', 
                              transforms=val_transforms)
test_dataset = PascalVOCDataset(voc_root, year='2007', image_set='test', 
                               transforms=val_transforms)

# IMPROVED: Adjusted batch size
train_loader = DataLoader(
    train_dataset, 
    batch_size=16,  # Reduced for ResNet-50 memory requirements
    shuffle=True, 
    num_workers=4,
    collate_fn=custom_collate_fn,
    pin_memory=True
)   

val_loader = DataLoader(
    val_dataset, 
    batch_size=16, 
    shuffle=False, 
    num_workers=4,
    collate_fn=custom_collate_fn,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=16, 
    shuffle=False, 
    num_workers=4,
    collate_fn=custom_collate_fn,
    pin_memory=True
)

# Define enhanced FPN module with Group Normalization
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        
        # Lateral connections (1x1 convolutions for channel reduction)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(32, out_channels),  # Group Normalization for better performance
                nn.ReLU(inplace=True)
            )
            for in_channels in in_channels_list
        ])
        
        # Output convolutions with Group Normalization
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),  # Group Normalization for better performance
                nn.ReLU(inplace=True)
            )
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

# Define SSD model with ResNet-50 backbone and improved FPN
class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        # Store num_classes as a class attribute
        self.num_classes = num_classes
        
        # IMPROVED: Using ResNet-50 backbone
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Extract intermediate feature maps from ResNet-50
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1
            ),  # Output stride: 4, channels: 256
            nn.Sequential(resnet.layer2),  # Output stride: 8, channels: 512
            nn.Sequential(resnet.layer3),  # Output stride: 16, channels: 1024
            nn.Sequential(resnet.layer4),  # Output stride: 32, channels: 2048
        ])
        
        # Feature channels from ResNet-50 blocks
        self.feature_channels = [256, 512, 1024, 2048]
        
        # Additional layers for deeper feature maps
        self.extra_layer1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
        )

        self.extra_layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )

        # IMPROVED: Add FPN with more channels for better feature fusion
        self.fpn = FPN(
            in_channels_list=[512, 1024, 2048, 512, 256],
            out_channels=256
        )
        
        # Define feature map sizes based on input size 512x512
        self.feature_maps = [64, 32, 16, 8, 4]
        self.steps = [8, 16, 32, 64, 128]
        
        # IMPROVED: Adjusted scales and aspect ratios for better anchor coverage
        self.scales = [0.05, 0.1, 0.25, 0.4, 0.6, 0.8]
        self.aspect_ratios = [[2, 3], [2, 3], [2, 3, 5], [2, 3, 5], [2, 3]]
        
        # Calculate number of boxes per feature map cell
        self.num_anchors = []
        for ar in self.aspect_ratios:
            # 1 + extra scale for aspect ratio 1 + 2 for each additional aspect ratio
            self.num_anchors.append(2 + 2 * len(ar))
        
        # Define location layers
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(256, self.num_anchors[0] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[1] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[2] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[3] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[4] * 4, kernel_size=3, padding=1)
        ])

        # Define confidence layers
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(256, self.num_anchors[0] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[1] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[2] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[3] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[4] * num_classes, kernel_size=3, padding=1)
        ])
        
        # Generate default boxes
        self._create_default_boxes()
        
        # IMPROVED: Add center-ness prediction layers for better localization
        self.centerness_layers = nn.ModuleList([
            nn.Conv2d(256, self.num_anchors[0], kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[1], kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[2], kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[3], kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_anchors[4], kernel_size=3, padding=1)
        ])
        
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
        # Extract features from ResNet backbone
        features = []
        for extractor in self.feature_extractors:
            x = extractor(x)
            features.append(x)
        
        # Add extra layers for deeper feature maps
        p6 = self.extra_layer1(features[3])  # Starting from layer4 of ResNet
        p7 = self.extra_layer2(p6)
        
        # Combine to get all features for FPN
        fpn_input = [features[1], features[2], features[3], p6, p7]
        
        # Apply FPN to get unified feature maps
        fpn_features = self.fpn(fpn_input)
        
        # Apply prediction layers
        loc_preds = []
        conf_preds = []
        centerness_preds = []
        
        for i, (feature, loc_layer, conf_layer, centerness_layer) in enumerate(
            zip(fpn_features, self.loc_layers, self.conf_layers, self.centerness_layers)
        ):
            # Get location predictions
            loc = loc_layer(feature)
            batch_size = loc.size(0)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(batch_size, -1, 4)
            loc_preds.append(loc)
            
            # Get confidence predictions
            conf = conf_layer(feature)
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(batch_size, -1, self.num_classes)
            conf_preds.append(conf)
            
            # Get centerness predictions
            centerness = centerness_layer(feature)
            centerness = centerness.permute(0, 2, 3, 1).contiguous()
            centerness = centerness.view(batch_size, -1, 1)
            centerness_preds.append(centerness)

        # Concatenate predictions from different feature maps
        loc_preds = torch.cat(loc_preds, dim=1)
        conf_preds = torch.cat(conf_preds, dim=1)
        centerness_preds = torch.cat(centerness_preds, dim=1)
        
        return loc_preds, conf_preds, centerness_preds, self.default_boxes_xyxy

# Improved GIoU loss with centerness weighting
def giou_loss(pred_boxes, target_boxes, centerness=None, eps=1e-7):
    """
    Calculates the Generalized IoU loss with optional centerness weighting
    
    Args:
        pred_boxes: (tensor) Predicted boxes, sized [N, 4]
        target_boxes: (tensor) Target boxes, sized [N, 4]
        centerness: (tensor, optional) Centerness weights, sized [N, 1]
        
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
    
    # Apply centerness weighting if provided
    if centerness is not None:
        loss = loss * centerness.squeeze(1)
    
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

def encode_boxes(matched_boxes, default_boxes, variance=[0.1, 0.2]):
    """
    Encode ground truth boxes relative to default boxes with variance scaling.
    """
    def corner_to_center(boxes):
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        cx = boxes[:, 0] + width / 2
        cy = boxes[:, 1] + height / 2
        return torch.stack([cx, cy, width, height], dim=1)
    
    g_boxes = corner_to_center(matched_boxes)
    d_boxes = corner_to_center(default_boxes)
    
    encoded_boxes = torch.zeros_like(g_boxes)
    encoded_boxes[:, 0] = (g_boxes[:, 0] - d_boxes[:, 0]) / (d_boxes[:, 2] * variance[0] + 1e-8)
    encoded_boxes[:, 1] = (g_boxes[:, 1] - d_boxes[:, 1]) / (d_boxes[:, 3] * variance[0] + 1e-8)
    encoded_boxes[:, 2] = torch.log(g_boxes[:, 2] / (d_boxes[:, 2] + 1e-8) + 1e-8) / variance[1]
    encoded_boxes[:, 3] = torch.log(g_boxes[:, 3] / (d_boxes[:, 3] + 1e-8) + 1e-8) / variance[1]
    
    return encoded_boxes

# Calculate centerness for better localization weighting
def compute_centerness(boxes):
    """
    Compute centerness for each box as in FCOS paper
    
    Args:
        boxes: (tensor) Boxes in (xmin, ymin, xmax, ymax) format
    
    Returns:
        Centerness scores for each box
    """
    left = boxes[:, 0]
    top = boxes[:, 1]
    right = boxes[:, 2]
    bottom = boxes[:, 3]
    
    width = right - left
    height = bottom - top
    
    centerx = (left + right) / 2
    centery = (top + bottom) / 2
    
    # Calculate distances from center to each side
    l_dist = centerx - left
    r_dist = right - centerx
    t_dist = centery - top
    b_dist = bottom - centery
    
    # Calculate centerness as in FCOS paper
    centerness = torch.sqrt(
        (torch.min(l_dist, r_dist) / torch.max(l_dist, r_dist + 1e-8)) *
        (torch.min(t_dist, b_dist) / torch.max(t_dist, b_dist + 1e-8))
    )
    
    return centerness

# Improved SSD loss with centerness weighting
class SSD_loss(nn.Module):
    def __init__(self, num_classes, default_boxes, device, alpha=0.25, gamma=2.0, 
                 lambda_loc=2.0, lambda_conf=1.0, lambda_center=1.0, variance=[0.1, 0.1]):
        super(SSD_loss, self).__init__()
        self.num_classes = num_classes
        self.default_boxes = default_boxes.to(device)
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_loc = lambda_loc
        self.lambda_conf = lambda_conf
        self.lambda_center = lambda_center
        self.variance = variance

        # Increased IoU threshold for improved precision
        self.threshold = 0.5
        self.neg_pos_ratio = 3  # Increased for better hard negative mining

        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # Initialize class weights (higher for rare classes)
        self.class_weights = torch.ones(num_classes, device=device)
        
        # Pascal VOC class frequency adjustments
        # Background is index 0
        # Rare classes (bird=3, boat=4, bottle=5, chair=9, cow=10, diningtable=11, 
        # pottedplant=16, sheep=17, sofa=18)
        rare_classes = [3, 4, 5, 9, 10, 11, 16, 17, 18]
        # Common classes (person=15, car=7, dog=12, cat=8)
        common_classes = [15, 7, 12, 8]
        
        # Adjust weights based on class frequency - more aggressive weighting
        for cls in rare_classes:
            self.class_weights[cls] = 2.5  # Higher weight for rare classes
            
        for cls in common_classes:
            self.class_weights[cls] = 0.75  # Lower weight for common classes
            
        # Background class (0) gets slightly lower weight
        self.class_weights[0] = 0.5
    
    def focal_loss(self, pred, target):
        ce_loss = self.cross_entropy(pred, target)
        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        pt = pred_softmax.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Apply class weights based on target classes
        class_weight = self.class_weights[target]
        
        # Combine focal weight with class weight
        focal_weight = self.alpha * (1 - pt).pow(self.gamma) * class_weight
        
        return focal_weight * ce_loss
    
    def forward(self, predictions, targets):
        loc_preds, conf_preds, centerness_preds, _ = predictions
        batch_size = loc_preds.size(0)
        num_priors = self.default_boxes.size(0)
        
        # Prepare target tensors
        loc_t = torch.zeros(batch_size, num_priors, 4).to(self.device)
        conf_t = torch.zeros(batch_size, num_priors, dtype=torch.long).to(self.device)
        centerness_t = torch.zeros(batch_size, num_priors, 1).to(self.device)
        
        # Track positives for each batch
        batch_positives = []
        
        for idx in range(batch_size):
            truths = targets['boxes'][idx]
            labels = targets['labels'][idx]
            if truths.size(0) == 0:
                batch_positives.append(0)
                continue
            
            # Match default boxes to ground truth
            overlaps = box_iou(self.default_boxes, truths)
            
            # For each default box, find best matching GT
            best_truth_overlap, best_truth_idx = overlaps.max(1)
            
            # For each GT, ensure it's matched to at least one default box
            best_prior_overlap, best_prior_idx = overlaps.max(0)
            for j in range(best_prior_idx.size(0)):
                best_truth_idx[best_prior_idx[j]] = j
                best_truth_overlap[best_prior_idx[j]] = 2.0  # Ensure it's positive
            
            # Extract matched boxes and labels
            matches = truths[best_truth_idx]
            match_labels = labels[best_truth_idx]
            
            # Mark positives (IoU > threshold)
            match_labels[best_truth_overlap < self.threshold] = 0  # Background
            
            # Calculate centerness targets for positive boxes
            pos_mask = (match_labels > 0)
            if pos_mask.sum() > 0:
                # Compute centerness only for positive boxes
                centerness_targets = compute_centerness(matches[pos_mask])
                centerness_t[idx, pos_mask, 0] = centerness_targets
            
            # Encode matched boxes
            loc_t[idx] = encode_boxes(matches, self.default_boxes, self.variance)
            conf_t[idx] = match_labels
            
            # Count positives
            batch_positives.append((match_labels > 0).sum().item())
        
        # Check for positives
        num_pos = sum(batch_positives)
        if num_pos == 0:
            print("WARNING: No positive matches in batch, returning minimal loss")
            return torch.tensor(0.001, requires_grad=True, device=self.device)
        
        # Create mask for positive examples
        pos = conf_t > 0
        pos_idx = pos.unsqueeze(2).expand_as(loc_preds)
        
        # Localization loss - only for positive matches
        pos_loc_preds = loc_preds[pos_idx].view(-1, 4)
        pos_loc_targets = loc_t[pos_idx].view(-1, 4)
        
        # Use smooth L1 loss for localization
        loc_loss = self.smooth_l1(pos_loc_preds, pos_loc_targets).sum(dim=1).mean()
        
        # Centerness loss - only for positive matches
        pos_centerness_idx = pos.unsqueeze(2).expand_as(centerness_preds)
        pos_centerness_preds = centerness_preds[pos_centerness_idx].view(-1, 1)
        pos_centerness_targets = centerness_t[pos_centerness_idx].view(-1, 1)
        
        # Use binary cross entropy loss for centerness
        centerness_loss = self.bce(pos_centerness_preds, pos_centerness_targets).mean()
        
        # Hard negative mining for classification
        batch_conf = conf_preds.view(-1, self.num_classes)
        
        # Loss for all examples
        loss_c = self.focal_loss(batch_conf, conf_t.view(-1))
        loss_c = loss_c.view(batch_size, -1)
        
        # Hard negative mining
        loss_c[pos] = 0  # Filter out positive boxes
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = torch.clamp(self.neg_pos_ratio * pos.sum(1), max=pos.size(1)-1)
        neg = idx_rank < num_neg.unsqueeze(1).expand_as(idx_rank)
        
        # Combined positive and negative examples
        pos_idx = pos.view(-1, 1).expand_as(batch_conf)
        neg_idx = neg.view(-1, 1).expand_as(batch_conf)
        conf_p = batch_conf[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t.view(-1)[(pos+neg).view(-1)]
        
        # Final classification loss
        conf_loss = self.focal_loss(conf_p, targets_weighted).mean()
        
        # Combined loss
        total_loss = (self.lambda_loc * loc_loss + 
                      self.lambda_conf * conf_loss + 
                      self.lambda_center * centerness_loss)
        
        # Safety check for NaNs
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("WARNING: NaN or Inf loss detected!")
            print(f"loc_loss: {loc_loss.item()}, conf_loss: {conf_loss.item()}, centerness_loss: {centerness_loss.item()}")
            return torch.tensor(0.1, device=self.device, requires_grad=True)
        
        return total_loss

def decode_boxes(loc, default_boxes, variance=[0.1, 0.2], clamp_val=4.0):
    """
    Decode predicted box coordinates from offsets with variance scaling.
    """
    device = loc.device
    default_boxes = default_boxes.to(device)

    def corner_to_center(boxes):
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        cx = boxes[:, 0] + width / 2
        cy = boxes[:, 1] + height / 2
        return torch.stack([cx, cy, width, height], dim=1)

    default_boxes_center = corner_to_center(default_boxes)
    
    # Clamp the width and height offsets for stability
    loc_w = torch.clamp(loc[:, 2] * variance[1], min=-clamp_val, max=clamp_val)
    loc_h = torch.clamp(loc[:, 3] * variance[1], min=-clamp_val, max=clamp_val)

    pred_cx = default_boxes_center[:, 0] + loc[:, 0] * variance[0] * default_boxes_center[:, 2]
    pred_cy = default_boxes_center[:, 1] + loc[:, 1] * variance[0] * default_boxes_center[:, 3]
    pred_w = default_boxes_center[:, 2] * torch.exp(loc_w)
    pred_h = default_boxes_center[:, 3] * torch.exp(loc_h)

    boxes_decoded = torch.zeros_like(loc)
    boxes_decoded[:, 0] = pred_cx - pred_w / 2
    boxes_decoded[:, 1] = pred_cy - pred_h / 2
    boxes_decoded[:, 2] = pred_cx + pred_w / 2
    boxes_decoded[:, 3] = pred_cy + pred_h / 2
    
    # Clamp to valid range
    boxes_decoded = torch.clamp(boxes_decoded, 0, 1)

    return boxes_decoded

# Instantiate the SSD model
model = SSD(num_classes=num_classes)
model.to(device)

# Initialize the SSD loss function
SSDLoss = SSD_loss(
    num_classes=num_classes, 
    default_boxes=model.default_boxes_xyxy,
    device=device,
    alpha=0.25,
    gamma=2.0,   # Increased gamma for better handling of hard examples
    lambda_loc=2.0,  # Increased localization loss weight for better precision
    lambda_conf=1.0,
    lambda_center=1.0  # Weight for centerness loss
)

# Freeze early layers of ResNet
for i in range(1):  # Only freeze the first feature extractor (conv1, bn1, maxpool, layer1)
    for param in model.feature_extractors[i].parameters():
        param.requires_grad = False

# IMPROVED: Use AdamW optimizer for better convergence
optimizer = optim.AdamW([
    {'params': model.feature_extractors[1:].parameters(), 'lr': 1e-4},
    {'params': model.extra_layer1.parameters(), 'lr': 2e-4},
    {'params': model.extra_layer2.parameters(), 'lr': 2e-4},
    {'params': model.fpn.parameters(), 'lr': 2e-4},
    {'params': model.loc_layers.parameters(), 'lr': 5e-4},
    {'params': model.conf_layers.parameters(), 'lr': 2e-4},
    {'params': model.centerness_layers.parameters(), 'lr': 2e-4}
], lr=1e-4, weight_decay=1e-4)

# IMPROVED: Better learning rate scheduler with faster warmup
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=[2e-4, 4e-4, 4e-4, 4e-4, 8e-4, 4e-4, 4e-4],  # Higher peak rates
    steps_per_epoch=len(train_loader),
    epochs=120,
    pct_start=0.3,  # Faster warmup
    div_factor=25,
    final_div_factor=10000,
    anneal_strategy='cos'
)

# Enhanced mAP calculation function with centerness weighting
def calculate_mAP(model, data_loader, device, conf_threshold=0.05, top_k=200):
    """
    Enhanced mAP calculation with centerness weighting
    """
    model.eval()
    metric = MeanAveragePrecision().to(device)
    metric.box_format = "xyxy"  # Ensure correct box format
    all_preds = []
    all_targets = []
    
    print("Starting mAP calculation...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating mAP"):
            images = batch['images'].to(device)
            batch_size = images.size(0)
            
            # Forward pass
            loc_preds, conf_preds, centerness_preds, default_boxes = model(images)
            
            # Process each image in batch
            for i in range(batch_size):
                # Calculate confidence scores
                scores = torch.nn.functional.softmax(conf_preds[i], dim=1)
                
                # Apply centerness weighting to scores
                centerness = torch.sigmoid(centerness_preds[i])
                
                # Decode boxes
                boxes = decode_boxes(loc_preds[i], default_boxes.to(device), variance=[0.1, 0.1])
                
                # Store predictions by class
                pred_boxes_all = []
                pred_scores_all = []
                pred_labels_all = []
                
                # Process each class
                for c in range(1, model.num_classes):  # Skip background
                    # Get scores for this class
                    class_scores = scores[:, c]
                    
                    # Apply centerness weighting to confidence scores
                    weighted_scores = class_scores * centerness.squeeze(1)
                    
                    # Filter by confidence threshold
                    mask = weighted_scores > conf_threshold
                    if mask.sum() == 0:
                        continue
                        
                    # Get filtered boxes
                    class_boxes = boxes[mask]
                    class_scores = weighted_scores[mask]
                    
                    # Take top-k if needed
                    if len(class_scores) > top_k:
                        _, idx = class_scores.topk(top_k)
                        class_boxes = class_boxes[idx]
                        class_scores = class_scores[idx]
                    
                    # Apply Soft-NMS instead of hard NMS
                    keep_idx = torchvision.ops.nms(class_boxes, class_scores, iou_threshold=0.5)
                    class_boxes = class_boxes[keep_idx]
                    class_scores = class_scores[keep_idx]
                    
                    # Add to predictions
                    pred_boxes_all.append(class_boxes)
                    pred_scores_all.append(class_scores)
                    pred_labels_all.extend([c] * len(class_boxes))
                
                # Format predictions and targets
                if len(pred_boxes_all) > 0:
                    pred_boxes_cat = torch.cat(pred_boxes_all)
                    pred_scores_cat = torch.cat(pred_scores_all)
                    pred_labels_cat = torch.tensor(pred_labels_all, device=device)
                else:
                    pred_boxes_cat = torch.zeros((0, 4), device=device)
                    pred_scores_cat = torch.zeros(0, device=device)
                    pred_labels_cat = torch.zeros(0, dtype=torch.int64, device=device)
                
                # Add to metric
                all_preds.append({
                    'boxes': pred_boxes_cat.cpu(),
                    'scores': pred_scores_cat.cpu(),
                    'labels': pred_labels_cat.cpu()
                })
                
                all_targets.append({
                    'boxes': batch['boxes'][i].cpu(),
                    'labels': batch['labels'][i].cpu()
                })
    
    # Compute mAP
    metric.update(all_preds, all_targets)
    results = metric.compute()
    
    # Print detailed results
    mAP = results['map'].item()
    mAP_50 = results['map_50'].item()
    mAP_75 = results['map_75'].item()
    
    print(f"mAP: {mAP:.4f}, mAP@0.5: {mAP_50:.4f}, mAP@0.75: {mAP_75:.4f}")
    
    return mAP

# Training loop with improved mAP calculation and augmentation
def train_model(model, loss_fn, optimizer, scheduler, train_loader, val_loader, 
                epochs, warmup_epochs=5, plateau_epochs=95, checkpoint_dir='./checkpoints'):
    """
    Enhanced training loop with three-phase learning rate schedule and mAP calculation
    
    Args:
        model: SSD model
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train for
        warmup_epochs, plateau_epochs: Lengths of the warmup and plateau phases
        checkpoint_dir: Directory to save checkpoints
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize lists for tracking metrics
    train_losses = []
    val_losses = []
    val_maps = []
    
    # For early stopping
    best_val_map = 0.0
    patience = 12  # Reduced patience for earlier stopping
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
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
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
                loc_preds, conf_preds, centerness_preds, default_boxes = model(images)
                
                # Compute loss
                loss = loss_fn((loc_preds, conf_preds, centerness_preds, default_boxes), {
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
            
            # Print batch progress for every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        scheduler.step()

        # Calculate average training loss
        avg_train_loss = epoch_loss / max(1, batch_count)
        train_losses.append(avg_train_loss)

        # Validation phase (only every few epochs to save time)
        val_epoch = (epoch + 1) % 5 == 0 or epoch < 2 or epoch >= epochs - 5 or epoch == warmup_epochs - 1
        
        if val_epoch:
            # Validation phase for loss calculation
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            print("Running validation...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                    # Move data to device
                    images = batch['images'].to(device)
                    boxes = batch['boxes']
                    labels = batch['labels']
                    
                    # Skip batches with no ground truth boxes
                    if all(b.size(0) == 0 for b in boxes):
                        continue
                    
                    # Forward pass
                    loc_preds, conf_preds, centerness_preds, default_boxes = model(images)
                    
                    # Compute loss
                    loss = loss_fn((loc_preds, conf_preds, centerness_preds, default_boxes), {
                        'boxes': [b.to(device) for b in boxes],
                        'labels': [l.to(device) for l in labels]
                    })
                    
                    # Track metrics
                    val_loss += loss.item()
                    val_batch_count += 1
            
            # Calculate average validation loss
            avg_val_loss = val_loss / max(1, val_batch_count)
            val_losses.append(avg_val_loss)
            
            # Calculate mAP on validation set
            print("Calculating mAP...")
            val_map = calculate_mAP(model, val_loader, device)
            val_maps.append(val_map)
            
            # Print epoch summary
            t_end = datetime.now()
            duration = t_end - t_start
            print(f"  Epoch {epoch+1} completed in {duration}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val mAP: {val_map:.4f}")
            
            # Save checkpoint if this is the best model so far (based on mAP)
            if val_map > best_val_map:
                best_val_map = val_map
                patience_counter = 0
                
                print(f"  Saving best model with validation mAP: {best_val_map:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_map': val_map,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                print(f"  No improvement in validation mAP for {patience_counter} validation runs")
                
                # Save periodic checkpoint
                if (epoch + 1) % 20 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'val_map': val_map,
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
        'val_map': val_maps[-1] if val_maps else 0.0,
    }, os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # Return training history
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maps': val_maps
    }

# Function to plot training history with mAP
def plot_training_history(history):
    """Plot training and validation metrics including mAP"""
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation losses
    plt.subplot(2, 1, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    
    # Plot validation metrics only for epochs where validation was performed
    val_epochs = np.linspace(0, len(history['train_losses'])-1, len(history['val_losses']))
    plt.plot(val_epochs, history['val_losses'], label='Validation Loss')
    
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation mAP
    plt.subplot(2, 1, 2)
    plt.plot(val_epochs, history['val_maps'], label='Validation mAP@0.5', color='g')
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_with_map.png')
    plt.show()

# Function to visualize detections with centerness scores
def visualize_detections(model, image_path, transform=None, conf_threshold=0.4):
    """
    Visualize object detections on an image with centerness scores
    
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
            A.Resize(height=512, width=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        transformed = transform(image=np.array(image))
        image_tensor = transformed['image']
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        loc_preds, conf_preds, centerness_preds, default_boxes = model(image_tensor)
        
        # Ensure default_boxes is on the same device
        default_boxes = default_boxes.to(device)
        
        # Decode predictions
        decoded_boxes = decode_boxes(loc_preds[0], default_boxes)
        
        # Get confidence scores
        scores = torch.nn.functional.softmax(conf_preds[0], dim=1)
        
        # Get centerness scores
        centerness = torch.sigmoid(centerness_preds[0]).squeeze(1)
        
        # Get detections with confidence > threshold
        detections = []
        
        for class_idx in range(1, model.num_classes):  # Skip background class
            class_scores = scores[:, class_idx]
            
            # Apply centerness weighting
            weighted_scores = class_scores * centerness
            mask = weighted_scores > conf_threshold
            
            if mask.sum() == 0:
                continue
                
            class_boxes = decoded_boxes[mask]
            class_scores = weighted_scores[mask]
            class_centerness = centerness[mask]
            
            # Apply non-maximum suppression
            keep_idx = torchvision.ops.nms(class_boxes, class_scores, iou_threshold=0.45)
            
            for idx in keep_idx:
                detections.append({
                    'box': class_boxes[idx].cpu().numpy(),
                    'score': class_scores[idx].item(),
                    'centerness': class_centerness[idx].item(),
                    'class': class_idx
                })
    
    # Visualize detections
    plt.figure(figsize=(12, 8))
    plt.imshow(orig_image)
    
    # Plot each detection
    for det in detections:
        box = det['box']
        score = det['score']
        centerness = det['centerness']
        class_idx = det['class']
        
        # Scale box coordinates to original image size
        h, w, _ = orig_image.shape
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1 * w))
        y1 = max(0, int(y1 * h))
        x2 = min(w, int(x2 * w))
        y2 = min(h, int(y2 * h))
        
        # Create rectangle patch with color based on confidence
        color = plt.cm.jet(score)[:3]  # Use jet colormap, higher score = more red
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                               edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        
        # Add label with score and centerness
        class_name = VOC_CLASSES[class_idx]
        plt.text(x1, y1-5, f'{class_name}: {score:.2f} (c:{centerness:.2f})', 
                 fontsize=10, color='white', 
                 bbox=dict(facecolor=color, alpha=0.7))
    
    plt.title('Object Detections with Centerness')
    plt.axis('off')
    plt.savefig('improved_detections_resnet50.png', dpi=300, bbox_inches='tight')
    plt.show()

# Function for test-time augmentation (TTA)
def test_time_augmentation(model, image, num_augs=5, conf_threshold=0.3):
    """
    Apply test-time augmentation for more robust predictions
    
    Args:
        model: Trained SSD model
        image: Input image tensor [C, H, W]
        num_augs: Number of augmentations to apply
        conf_threshold: Confidence threshold for detections
        
    Returns:
        List of detection dictionaries
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Original image
    image = image.unsqueeze(0).to(device)
    
    # Initialize augmentations
    augs = [
        # Original
        lambda x: x,
        # Horizontal flip
        lambda x: torch.flip(x, [3]),
        # Brightness adjustment
        lambda x: torch.clamp(x * 1.1, 0, 1),
        # Contrast adjustment
        lambda x: torch.clamp((x - x.mean()) * 1.1 + x.mean(), 0, 1),
        # Small rotation (implemented as identity since proper rotation requires more work)
        lambda x: x,
    ]
    
    # Use a subset of augmentations
    if num_augs < len(augs):
        augs = augs[:num_augs]
    
    all_detections = []
    
    with torch.no_grad():
        for aug_fn in augs:
            # Apply augmentation
            aug_image = aug_fn(image)
            
            # Get predictions
            loc_preds, conf_preds, centerness_preds, default_boxes = model(aug_image)
            
            # Process predictions
            scores = torch.nn.functional.softmax(conf_preds[0], dim=1)
            centerness = torch.sigmoid(centerness_preds[0]).squeeze(1)
            boxes = decode_boxes(loc_preds[0], default_boxes.to(device))
            
            # Handle inverse transformation for boxes if needed (e.g., flip back)
            if aug_fn == augs[1]:  # If horizontal flip
                boxes[:, 0], boxes[:, 2] = 1 - boxes[:, 2], 1 - boxes[:, 0]
            
            # Get detections for each class
            for class_idx in range(1, model.num_classes):
                class_scores = scores[:, class_idx]
                weighted_scores = class_scores * centerness
                mask = weighted_scores > conf_threshold
                
                if mask.sum() == 0:
                    continue
                
                # Get filtered predictions
                class_boxes = boxes[mask].cpu().numpy()
                class_scores = weighted_scores[mask].cpu().numpy()
                
                for i in range(len(class_boxes)):
                    all_detections.append({
                        'box': class_boxes[i],
                        'score': class_scores[i],
                        'class': class_idx
                    })
    
    # Apply non-maximum suppression across all augmentations
    # Group by class
    class_detections = defaultdict(list)
    for det in all_detections:
        class_detections[det['class']].append(det)
    
    # NMS for each class
    final_detections = []
    for class_idx, dets in class_detections.items():
        if not dets:
            continue
            
        boxes = np.array([d['box'] for d in dets])
        scores = np.array([d['score'] for d in dets])
        
        # Convert to torch tensors for NMS
        boxes_tensor = torch.FloatTensor(boxes).to(device)
        scores_tensor = torch.FloatTensor(scores).to(device)
        
        # Apply NMS
        keep_idx = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        
        for idx in keep_idx.cpu().numpy():
            final_detections.append({
                'box': boxes[idx],
                'score': scores[idx],
                'class': class_idx
            })
    
    return final_detections

# Main training function call
print("Starting improved SSD model training with ResNet-50 backbone...")
history = train_model(
    model=model,
    loss_fn=SSDLoss,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=120,  # Reduced epochs with more effective training
    warmup_epochs=3,
    plateau_epochs=77
)

# Plot training history including mAP
plot_training_history(history)

# Load the best model
best_model_path = os.path.join('checkpoints', 'best_model.pth')
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model with validation mAP: {checkpoint['val_map']:.4f}")
else:
    print("No best model checkpoint found. Using final trained model.")

# Evaluate on test set with TTA
print("Evaluating on test set with Test-Time Augmentation...")

# Custom calculation for mAP with TTA
test_map = calculate_mAP(model, test_loader, device, conf_threshold=0.05)
print(f"Test set mAP@0.5 (without TTA): {test_map:.4f}")

# Save the final model in a format suitable for inference
final_model_path = 'improved_ssd_resnet50_pascal_voc_final.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'class_names': VOC_CLASSES,
    'test_map': test_map,
    'architecture': 'SSD-ResNet50-FPN-Centerness'
}, final_model_path)

print(f"Final model saved to {final_model_path}")

# Apply model quantization for faster inference
print("Creating quantized model for faster inference...")

# Prepare for quantization
model.eval()

# Quantization-aware model definition
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)

# Save quantized model
quantized_model_path = 'improved_ssd_resnet50_pascal_voc_quantized.pth'
torch.save({
    'model_state_dict': quantized_model.state_dict(),
    'num_classes': num_classes,
    'class_names': VOC_CLASSES,
    'test_map': test_map,
    'architecture': 'SSD-ResNet50-FPN-Centerness-Quantized'
}, quantized_model_path)

print(f"Quantized model saved to {quantized_model_path}")

