# SSD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import random
import os
from PIL import Image
import json

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Pascal VOC Classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Number of classes (20 VOC classes + 1 background)
num_classes = len(VOC_CLASSES) + 1

# Define transforms with proper Albumentations pipeline
train_transforms = A.Compose([
    # Spatial transformations
    A.Resize(height=300, width=300),
    
    # Flips and rotations
    A.HorizontalFlip(p=0.5),
    
    # Color augmentations - simplified but effective
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    ], p=0.5),
    
    # Light noise and blur - helps with robustness
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
        A.GaussianBlur(blur_limit=3, p=1.0),
    ], p=0.2),
    
    # Occasional weather simulation
    A.RandomShadow(p=0.2),
    
    # Normalize and convert to tensor
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))

# Validation transformations
val_transforms = A.Compose([
    A.Resize(height=300, width=300),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Test transformations (for inference)
test_transforms = A.Compose([
    A.Resize(height=300, width=300),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Custom Pascal VOC Dataset
class VOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', transforms=None):
        self.voc = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=True
        )
        self.transforms = transforms
        
    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, idx):
        img, target = self.voc[idx]
        
        # Extract bounding boxes and labels from VOC annotation format
        boxes = []
        labels = []
        
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            
            # Get class label (convert class name to index)
            label = VOC_CLASSES.index(obj['name'])
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transformations
        if self.transforms:
            transformed = self.transforms(image=np.array(img), bboxes=boxes, labels=labels)
            img = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        
        # If no boxes left after transforms, return empty arrays
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0), dtype=np.int64)
        
        # Convert to torch tensors
        boxes = torch.from_numpy(boxes)
        labels = torch.from_numpy(labels) + 1  # Add 1 because 0 is reserved for background
        
        return {
            'images': img,
            'boxes': boxes,
            'labels': labels
        }

# Custom collate function for batching
def collate_fn(batch):
    """Custom collate function to handle variable sized boxes and labels."""
    images = []
    boxes = []
    labels = []
    
    for item in batch:
        images.append(item['images'])
        boxes.append(item['boxes'])
        labels.append(item['labels'])
    
    # Stack images into a batch
    images = torch.stack(images, 0)
    
    # No stacking for boxes and labels since they have variable sizes
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels
    }

# Define model 
class SSD(nn.Module):
    def __init__(self, num_classes):
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
        
        # Calculate number of boxes per feature map cell
        self.num_anchors = []
        for ar in self.aspect_ratios:
            # 1 + extra scale for aspect ratio 1 + 2 for each additional aspect ratio
            self.num_anchors.append(2 + 2 * len(ar))
        
        # Define location layers
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512, self.num_anchors[0] * 4, kernel_size=3, padding=1),  # For conv1
            nn.Conv2d(512, self.num_anchors[1] * 4, kernel_size=3, padding=1),  # For conv2
            nn.Conv2d(256, self.num_anchors[2] * 4, kernel_size=3, padding=1),  # For conv3
            nn.Conv2d(256, self.num_anchors[3] * 4, kernel_size=3, padding=1),  # For conv4
            nn.Conv2d(256, self.num_anchors[4] * 4, kernel_size=3, padding=1),  # For conv5
            nn.Conv2d(256, self.num_anchors[5] * 4, kernel_size=3, padding=1)   # For conv6
        ])

        # Define confidence layers
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(512, self.num_anchors[0] * num_classes, kernel_size=3, padding=1),  # For conv1
            nn.Conv2d(512, self.num_anchors[1] * num_classes, kernel_size=3, padding=1),  # For conv2
            nn.Conv2d(256, self.num_anchors[2] * num_classes, kernel_size=3, padding=1),  # For conv3
            nn.Conv2d(256, self.num_anchors[3] * num_classes, kernel_size=3, padding=1),  # For conv4
            nn.Conv2d(256, self.num_anchors[4] * num_classes, kernel_size=3, padding=1),  # For conv5
            nn.Conv2d(256, self.num_anchors[5] * num_classes, kernel_size=3, padding=1)   # For conv6
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
        # Extract feature maps
        sources = []
        
        # Apply VGG layers and extract feature maps
        x = self.conv1(x)
        sources.append(x)  # 38x38 feature map
        
        x = self.conv2(x)
        sources.append(x)  # 19x19 feature map
        
        x = self.conv3(x)
        sources.append(x)  # 10x10 feature map
        
        x = self.conv4(x)
        sources.append(x)  # 5x5 feature map
        
        x = self.conv5(x)
        sources.append(x)  # 3x3 feature map
        
        x = self.conv6(x)
        sources.append(x)  # 1x1 feature map

        # Apply prediction layers
        loc_preds = []
        conf_preds = []
        
        for i, (source, loc_layer, conf_layer) in enumerate(zip(sources, self.loc_layers, self.conf_layers)):
            # Get location predictions
            loc = loc_layer(source)
            batch_size = loc.size(0)
            # Reshape to [batch_size, height, width, num_anchors * 4] then flatten to [batch_size, num_anchors_total, 4]
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(batch_size, -1, 4)
            loc_preds.append(loc)
            
            # Get confidence predictions
            conf = conf_layer(source)
            # Reshape to [batch_size, height, width, num_anchors * num_classes] then flatten
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(batch_size, -1, num_classes)
            conf_preds.append(conf)

        # Concatenate predictions from different feature maps
        loc_preds = torch.cat(loc_preds, dim=1)
        conf_preds = torch.cat(conf_preds, dim=1)
        
        return loc_preds, conf_preds, self.default_boxes_xyxy

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

class SSD_loss(nn.Module):
    def __init__(self, num_classes, default_boxes, device):
        """
        SSD Loss function

        Args:
            num_classes (int): Number of object classes
            default_boxes (torch.Tensor): Default anchor boxes (in corner format)
            device (torch.device): GPU or CPU
        """
        super(SSD_loss, self).__init__()
        
        self.num_classes = num_classes
        self.default_boxes = default_boxes.to(device)
        self.device = device
        
        self.threshold = 0.5  # IoU threshold for positive matches
        self.neg_pos_ratio = 3  # Ratio of negative to positive samples
        
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Compute SSD loss.

        Args:
            predictions (tuple): (loc_preds, conf_preds, default_boxes)
                - loc_preds: Shape (batch_size, num_priors, 4)
                - conf_preds: Shape (batch_size, num_priors, num_classes)
                - default_boxes: Default boxes used in the model
            targets (dict): {"boxes": [list of GT boxes], "labels": [list of GT labels]}
        
        Returns:
            torch.Tensor: Total loss
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
        
        # Localization loss (only for positive matches)
        pos_idx = pos.unsqueeze(2).expand_as(loc_preds)
        loc_loss = self.smooth_l1(loc_preds[pos_idx].view(-1, 4), 
                                 loc_t[pos_idx].view(-1, 4))
        
        # Confidence loss
        # Reshape confidence predictions to [batch_size * num_priors, num_classes]
        batch_conf = conf_preds.view(-1, self.num_classes)
        # Compute softmax loss
        loss_c = self.cross_entropy(batch_conf, conf_t.view(-1))
        loss_c = loss_c.view(batch_size, -1)
        
        # Hard negative mining
        # Exclude positive examples from negative mining
        loss_c[pos] = 0
        # Sort confidence losses in descending order
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # Number of negative examples to keep
        num_pos_per_batch = pos.long().sum(1)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos_per_batch, min=1, max=num_priors - 1)
        # Keep only the selected negatives
        neg = idx_rank < num_neg.unsqueeze(1)
        
        # Combine positive and selected negative examples for confidence loss
        pos_idx = pos.unsqueeze(2).expand_as(conf_preds)
        neg_idx = neg.unsqueeze(2).expand_as(conf_preds)
        conf_loss = self.cross_entropy(conf_preds[pos_idx | neg_idx].view(-1, self.num_classes),
                                    conf_t[pos | neg].view(-1))
        
        # Normalize by number of positive examples
        pos_count = max(1, num_pos)  # Avoid division by zero
        loc_loss /= pos_count
        conf_loss /= pos_count
        
        return loc_loss + conf_loss

def decode_boxes(loc, default_boxes):
    """Decode predicted box coordinates from offsets"""
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

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-10)

# Function to calculate mAP (simplified version for training feedback)
def calculate_map(model, val_loader, device, iou_threshold=0.5, conf_threshold=0.5):
    model.eval()
    all_detections = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            
            # Forward pass
            loc_preds, conf_preds, default_boxes = model(images)
            
            # Convert predictions to detections
            batch_size = loc_preds.size(0)
            for i in range(batch_size):
                # Decode locations
                detected_boxes = decode_boxes(loc_preds[i], default_boxes)
                
                # Get scores for each class
                scores = torch.nn.functional.softmax(conf_preds[i], dim=1)
                
                # Keep track of detections and ground truths
                detections = []
                for c in range(1, num_classes):  # Skip background class
                    class_scores = scores[:, c]
                    mask = class_scores > conf_threshold
                    
                    if mask.sum() == 0:
                        continue
                        
                    class_boxes = detected_boxes[mask]
                    class_scores = class_scores[mask]
                    
                    # Non-maximum suppression
                    indices = torchvision.ops.nms(class_boxes, class_scores, iou_threshold)
                    
                    for idx in indices:
                        detections.append({
                            'box': class_boxes[idx].cpu().numpy(),
                            'score': class_scores[idx].item(),
                            'class': c
                        })
                
                all_detections.append(detections)
                
                # Ground truths
                gt_boxes_img = gt_boxes[i].cpu().numpy()
                gt_labels_img = gt_labels[i].cpu().numpy()
                ground_truths = []
                
                for box, label in zip(gt_boxes_img, gt_labels_img):
                    ground_truths.append({
                        'box': box,
                        'class': label.item()
                    })
                
                all_ground_truths.append(ground_truths)
    
    # Simple mAP calculation
    # This is a placeholder for efficiency during training
    correct_detections = 0
    total_detections = 1e-6  # Avoid division by zero
    
    for dets, gts in zip(all_detections, all_ground_truths):
        total_detections += len(dets)
        
        for det in dets:
            for gt in gts:
                if det['class'] == gt['class']:
                    # Calculate IoU
                    iou = calculate_iou(det['box'], gt['box'])
                    if iou >= iou_threshold:
                        correct_detections += 1
                        break
    
    return correct_detections / total_detections

# Set up dataset paths
data_root = 'data/VOC'  # This directory will be created if it doesn't exist

# Create datasets
train_dataset = VOCDataset(
    root=data_root,
    year='2007',
    image_set='train',
    transforms=train_transforms
)

val_dataset = VOCDataset(
    root=data_root,
    year='2007',
    image_set='val',
    transforms=val_transforms
)

# Create data loaders with appropriate batch size for H100
train_loader = DataLoader(
    train_dataset,
    batch_size=32,  # Larger batch size for H100
    shuffle=True,
    num_workers=8,  # More workers for faster data loading
    collate_fn=collate_fn,
    pin_memory=True,  # Faster data transfer to GPU
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_fn,
    pin_memory=True,
)

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
for idx, param in enumerate(model.conv1.parameters()):
    layer_idx = idx // 2  # Each layer has weights and biases, so divide by 2
    if layer_idx < 10:    # First 10 layers
        param.requires_grad = False

# Define optimizer with learning rate scheduling optimized for H100
optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 0.0002, 'weight_decay': 1e-4},
    {'params': model.conv2.parameters(), 'lr': 0.0002, 'weight_decay': 1e-4},
    {'params': model.conv3.parameters(), 'lr': 0.002, 'weight_decay': 1e-4},
    {'params': model.conv4.parameters(), 'lr': 0.002, 'weight_decay': 1e-4},
    {'params': model.conv5.parameters(), 'lr': 0.002, 'weight_decay': 1e-4},
    {'params': model.conv6.parameters(), 'lr': 0.002, 'weight_decay': 1e-4},
    {'params': model.loc_layers.parameters(), 'lr': 0.002, 'weight_decay': 1e-4},
    {'params': model.conf_layers.parameters(), 'lr': 0.002, 'weight_decay': 1e-4}
], betas=(0.9, 0.999))

# Scheduler for faster convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.7,  
    patience=2,  
    verbose=True
)

# Training loop
def train_model(model, SSDLoss, optimizer, scheduler, train_loader, val_loader, epochs):
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # For timing
    start_time = datetime.now()
    
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images']
            boxes = batch['boxes']
            labels = batch['labels']
            
            # Move data to the appropriate device
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            loc_preds, conf_preds, default_boxes = model(images)
            
            # Skip batches with no ground truth boxes
            if all(b.size(0) == 0 for b in boxes):
                continue
            
            # Compute loss
            loss = SSDLoss((loc_preds, conf_preds, default_boxes), {'boxes': boxes, 'labels': labels})
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()

            train_loss.append(loss.item())
            
            # Print progress
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch {it+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                # Print estimated remaining time
                elapsed = datetime.now() - start_time
                progress = (it * len(train_loader) + batch_idx + 1) / (epochs * len(train_loader))
                if progress > 0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    print(f"Elapsed: {elapsed}, Estimated remaining: {remaining}")
        
        # Get train loss mean
        train_loss_mean = np.mean(train_loss) if train_loss else 0
        train_losses[it] = train_loss_mean

        # Validation phase
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images']
                boxes = batch['boxes']
                labels = batch['labels']
                
                # Move data to the appropriate device
                images = images.to(device)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]
                
                # Skip batches with no ground truth boxes
                if all(b.size(0) == 0 for b in boxes):
                    continue
                
                # Forward pass
                loc_preds, conf_preds, default_boxes = model(images)
                
                # Compute loss
                loss = SSDLoss((loc_preds, conf_preds, default_boxes), {'boxes': boxes, 'labels': labels})

                val_loss.append(loss.item())
        
        # Get validation loss mean
        val_loss_mean = np.mean(val_loss) if val_loss else float('inf')
        val_losses[it] = val_loss_mean
        
        # Update learning rate
        scheduler.step(val_loss_mean)

        # Calculate mAP on validation set every 5 epochs or on the last epoch
        if (it + 1) % 5 == 0 or it == epochs - 1:
            mAP = calculate_map(model, val_loader, device)
            print(f"Epoch {it+1}, mAP: {mAP:.4f}")
            
            # If we've reached target accuracy, we can stop early
            if mAP >= 0.7:  # 70% mAP
                print(f"Reached target accuracy of {mAP:.4f} at epoch {it+1}")
                
                # Save the model
                torch.save({
                    'epoch': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mAP': mAP,
                }, 'ssd_target_accuracy.pth')
                
                if mAP >= 0.75:  # If we hit the upper target, definitely stop
                    print("Reached upper target accuracy, stopping training")
                    break
        
        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss_mean:.4f}, Validation Loss: {val_loss_mean:.4f}, Duration: {dt}')
        
        # Early stopping check
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            patience_counter = 0
            
            # Save the best model
            torch.save({
                'epoch': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_ssd_model.pth')
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {it+1} epochs")
                break
                
        # Check if we're reaching the time limit (1.5 hours)
        if (datetime.now() - start_time).total_seconds() > 5400:  # 1.5 hours
            print("Approaching time limit, stopping training")
            break

    total_time = datetime.now() - start_time
    print(f"Total training time: {total_time}")
    return train_losses, val_losses

# Train the model
num_epochs = 20 
train_losses, val_losses = train_model(model, SSDLoss, optimizer, scheduler, train_loader, val_loader, epochs=num_epochs)

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses[:len(train_losses[train_losses > 0])], label='train loss')
plt.plot(val_losses[:len(val_losses[val_losses > 0])], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig('ssd_training_loss.png')
plt.show()

# Load the best model
checkpoint = torch.load('best_ssd_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Calculate final mAP on validation set
mAP = calculate_map(model, val_loader, device)
print(f"Final mAP: {mAP:.4f}")

# Save the final model
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'mAP': mAP
}, 'ssd-object-detection-final.pth')

print(f"Final model saved to ssd-object-detection-final.pth")