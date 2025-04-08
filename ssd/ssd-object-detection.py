# SSD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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

# Define Pascal VOC Dataset
class PascalVOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', transforms=None):
        """
        Pascal VOC Dataset
        
        Args:
            root (str): Path to VOCdevkit folder
            year (str): Dataset year ('2007' or '2012')
            image_set (str): Dataset type ('train', 'val', 'test')
            transforms (callable, optional): Optional transforms to be applied
        """
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transforms = transforms
        
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
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # Load image
        img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Load annotation
        anno_path = os.path.join(self.annotations_dir, f'{img_id}.xml')
        boxes, labels = self._parse_voc_xml(ET.parse(anno_path).getroot())
        
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
    
    def _parse_voc_xml(self, node):
        """Parse Pascal VOC annotation XML file"""
        boxes = []
        labels = []
        
        for obj in node.findall('object'):
            # Skip difficult objects if needed
            # difficult = int(obj.find('difficult').text)
            # if difficult:
            #     continue
            
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

# Define the transforms
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
        A.GaussNoise(p=1.0),  # Fixed: removed var_limit parameter
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

# Create datasets
# Set the correct path to your VOCdevkit directory
voc_root = '/home/adrian/ml-rover/VOCdevkit/VOCdevkit'  # Update this path

# For 2007 dataset
train_dataset = PascalVOCDataset(voc_root, year='2007', image_set='train', transforms=train_transforms)
val_dataset = PascalVOCDataset(voc_root, year='2007', image_set='val', transforms=val_transforms)
# Optionally add the test set for final evaluation
test_dataset = PascalVOCDataset(voc_root, year='2007', image_set='test', transforms=val_transforms)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=16,
    shuffle=True, 
    num_workers=4,
    collate_fn=custom_collate_fn  
)   

val_loader = DataLoader(
    val_dataset, 
    batch_size=16, 
    shuffle=False, 
    num_workers=4,
    collate_fn=custom_collate_fn
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=16, 
    shuffle=False, 
    num_workers=4,
    collate_fn=custom_collate_fn
)

# Define SSD model with corrected feature map sizes
class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        # Store num_classes as a class attribute
        self.num_classes = num_classes
        
        # Load VGG16 with pretrained weights
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # Extract features directly without adding extra BatchNorm2d
        # First feature map (37x37)
        self.conv1 = nn.Sequential(*features[:23])  # All VGG layers up to conv4_3

        # Second feature map (18x18)
        self.conv2 = nn.Sequential(*features[23:30])  # Conv5 blocks

        # Additional convolution layers (18x18) - FC layers converted to Conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # (9x9)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

        # (5x5)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

        # (2x2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=2),  # No padding to get 2x2
            nn.ReLU(inplace=True),
        )

        # Define actual feature map sizes based on the network architecture
        self.feature_maps = [37, 18, 18, 9, 5, 2]  # Updated to match actual sizes
        self.steps = [8, 16, 16, 32, 64, 100]  # effective stride for each feature map
        self.scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]  # anchor box scales
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # aspect ratios for each feature map
        
        # Calculate number of boxes per feature map cell
        self.num_anchors = []
        for ar in self.aspect_ratios:
            # 1 + extra scale for aspect ratio 1 + 2 for each additional aspect ratio
            self.num_anchors.append(2 + 2 * len(ar))
        
        # Define location layers with corrected output channels
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512, self.num_anchors[0] * 4, kernel_size=3, padding=1),  # For conv1
            nn.Conv2d(512, self.num_anchors[1] * 4, kernel_size=3, padding=1),  # For conv2
            nn.Conv2d(1024, self.num_anchors[2] * 4, kernel_size=3, padding=1),  # For conv3
            nn.Conv2d(512, self.num_anchors[3] * 4, kernel_size=3, padding=1),  # For conv4
            nn.Conv2d(256, self.num_anchors[4] * 4, kernel_size=3, padding=1),  # For conv5
            nn.Conv2d(256, self.num_anchors[5] * 4, kernel_size=3, padding=1)   # For conv6
        ])

        # Define confidence layers with corrected input channels
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(512, self.num_anchors[0] * num_classes, kernel_size=3, padding=1),  # For conv1
            nn.Conv2d(512, self.num_anchors[1] * num_classes, kernel_size=3, padding=1),  # For conv2
            nn.Conv2d(1024, self.num_anchors[2] * num_classes, kernel_size=3, padding=1),  # For conv3
            nn.Conv2d(512, self.num_anchors[3] * num_classes, kernel_size=3, padding=1),  # For conv4
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
        sources.append(x)  # 37x37 feature map
        
        x = self.conv2(x)
        sources.append(x)  # 18x18 feature map
        
        x = self.conv3(x)
        sources.append(x)  # 18x18 feature map (different channels)
        
        x = self.conv4(x)
        sources.append(x)  # 9x9 feature map
        
        x = self.conv5(x)
        sources.append(x)  # 5x5 feature map
        
        x = self.conv6(x)
        sources.append(x)  # 2x2 feature map

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
            conf = conf.view(batch_size, -1, self.num_classes)
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
        
        # Sum loss values to get a scalar
        conf_loss = conf_loss.sum()
        
        # Normalize by number of positive examples
        pos_count = max(1, num_pos)  # Avoid division by zero
        loc_loss /= pos_count
        conf_loss /= pos_count
        
        # Return scalar total loss
        total_loss = loc_loss + conf_loss
        return total_loss

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

# Fix for the device issue in decode_boxes function

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

# Fix for calculate_ap function to ensure device consistency
def calculate_ap(model, val_loader, device, iou_threshold=0.5, conf_threshold=0.5):
    """
    Calculate Average Precision for Pascal VOC
    """
    model.eval()
    
    # Initialize metrics
    all_detections = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            
            # Forward pass
            loc_preds, conf_preds, default_boxes = model(images)
            
            # Ensure default_boxes is on the same device
            default_boxes = default_boxes.to(device)
            
            # Convert predictions to detections
            batch_size = loc_preds.size(0)
            for i in range(batch_size):
                # Decode locations
                detected_boxes = decode_boxes(loc_preds[i], default_boxes)
                
                # Get scores for each class
                scores = torch.nn.functional.softmax(conf_preds[i], dim=1)
                
                # Keep track of detections and ground truths
                image_detections = []
                for c in range(1, model.num_classes):  # Skip background class
                    class_scores = scores[:, c]
                    mask = class_scores > conf_threshold
                    
                    if mask.sum() == 0:
                        continue
                        
                    class_boxes = detected_boxes[mask]
                    class_scores = class_scores[mask]
                    
                    # Non-maximum suppression
                    indices = torchvision.ops.nms(class_boxes, class_scores, iou_threshold)
                    
                    for idx in indices:
                        image_detections.append({
                            'box': class_boxes[idx].cpu().numpy(),
                            'score': class_scores[idx].item(),
                            'class': c
                        })
                
                all_detections.append(image_detections)
                
                # Process ground truths
                gt_boxes_img = gt_boxes[i].cpu().numpy()
                gt_labels_img = gt_labels[i].cpu().numpy()
                ground_truths = []
                
                for box, label in zip(gt_boxes_img, gt_labels_img):
                    ground_truths.append({
                        'box': box,
                        'class': label.item()
                    })
                
                all_ground_truths.append(ground_truths)
    
    # Calculate AP for each class
    aps = []
    for c in range(1, model.num_classes):  # Skip background class
        # Collect all detections and ground truths for this class
        class_detections = []
        class_gt_count = 0
        
        for img_idx in range(len(all_detections)):
            # Get detections for this class
            img_detections = [d for d in all_detections[img_idx] if d['class'] == c]
            # Sort by confidence score (highest first)
            img_detections.sort(key=lambda x: x['score'], reverse=True)
            class_detections.extend(img_detections)
            
            # Count ground truths for this class
            class_gt_count += sum(1 for gt in all_ground_truths[img_idx] if gt['class'] == c)
        
        # Skip classes with no ground truth objects
        if class_gt_count == 0:
            continue
            
        # Initialize tp and fp arrays
        tp = np.zeros(len(class_detections))
        fp = np.zeros(len(class_detections))
        
        # Create list of gt matches for each image
        gt_matched = []
        for gt_list in all_ground_truths:
            gt_matched.append([False] * len(gt_list))
        
        # Match detections to ground truths
        for d_idx, detection in enumerate(class_detections):
            # Find which image this detection came from by looping through all_detections
            img_idx = None
            for i, img_dets in enumerate(all_detections):
                for d in img_dets:
                    if d is detection:  # Identity check (same object in memory)
                        img_idx = i
                        break
                if img_idx is not None:
                    break
            
            if img_idx is None:
                fp[d_idx] = 1
                continue
                
            gt_list = all_ground_truths[img_idx]
            max_iou = -1
            max_idx = -1
            
            # Find the best matching ground truth for this detection
            for gt_idx, gt in enumerate(gt_list):
                if gt['class'] != detection['class']:
                    continue
                    
                # Skip already matched ground truths
                if gt_matched[img_idx][gt_idx]:
                    continue
                    
                # Calculate IoU
                iou = calculate_iou(detection['box'], gt['box'])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = gt_idx
            
            # If IoU exceeds threshold, it's a true positive
            if max_iou >= iou_threshold and max_idx >= 0:
                tp[d_idx] = 1
                gt_matched[img_idx][max_idx] = True
            else:
                fp[d_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / max(1, class_gt_count)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Add point at recall=0, precision=1 for AP calculation
        precisions = np.concatenate(([1], precisions))
        recalls = np.concatenate(([0], recalls))
        
        # Compute AP as area under precision-recall curve (VOC method)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):  # 11-point interpolation
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
            
        aps.append(ap)
    
    # Mean AP across all classes
    mAP = np.mean(aps) if aps else 0
    return mAP

# Instantiate the SSD model and send it to the GPU
model = SSD(num_classes=num_classes)
model.to(device)

# Debug - verify the anchor box and prediction shape match
with torch.no_grad():
    dummy_input = torch.zeros(1, 3, 300, 300).to(device)
    loc_preds, conf_preds, default_boxes = model(dummy_input)
    print(f"Default boxes shape: {model.default_boxes.shape}")
    print(f"Predictions shape: {loc_preds.shape}")
    
    # Calculate total anchor boxes from feature maps
    total_boxes = 0
    for i, f in enumerate(model.feature_maps):
        num_boxes = f*f*model.num_anchors[i]
        print(f"Feature map {i}: {f}x{f} with {model.num_anchors[i]} anchors = {num_boxes} boxes")
        total_boxes += num_boxes
    print(f"Total calculated boxes: {total_boxes}")

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
    {'params': [p for p in model.conv1.parameters() if p.requires_grad], 'lr': 0.0001},
    {'params': model.conv2.parameters(), 'lr': 0.0005},
    {'params': model.conv3.parameters(), 'lr': 0.001},
    {'params': model.conv4.parameters(), 'lr': 0.001},
    {'params': model.conv5.parameters(), 'lr': 0.001},
    {'params': model.conv6.parameters(), 'lr': 0.001},
    {'params': model.loc_layers.parameters(), 'lr': 0.001},
    {'params': model.conf_layers.parameters(), 'lr': 0.001}
], lr=0.001, weight_decay=5e-4)

# Learning rate scheduler with warmup
def get_lr_scheduler(optimizer, warmup_epochs=3, max_epochs=35):
    # Define warmup scheduler (linear warmup)
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Main scheduler (reduce on plateau)
    reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    return warmup_scheduler, reduce_scheduler

# Get schedulers
warmup_scheduler, lr_scheduler = get_lr_scheduler(optimizer)

# Training loop with improved logging and early stopping
def train_model(model, loss_fn, optimizer, warmup_scheduler, lr_scheduler, 
                train_loader, val_loader, epochs, checkpoint_dir='./checkpoints'):
    """
    Training loop for SSD model
    
    Args:
        model: SSD model
        loss_fn: Loss function
        optimizer: Optimizer
        warmup_scheduler: Learning rate scheduler for warmup phase
        lr_scheduler: Main learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train for
        checkpoint_dir: Directory to save checkpoints
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize lists for tracking metrics
    train_losses = []
    val_losses = []
    val_maps = []
    
    # For early stopping
    best_val_map = 0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
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
            
            # Forward pass
            loc_preds, conf_preds, default_boxes = model(images)
            
            # Compute loss
            loss = loss_fn((loc_preds, conf_preds, default_boxes), {
                'boxes': [b.to(device) for b in boxes],
                'labels': [l.to(device) for l in labels]
            })
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Print batch progress
            if (batch_idx + 1) % 20 == 0 or batch_idx + 1 == len(train_loader):
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Update learning rate with warmup scheduler
        if epoch < 3:  # Warmup phase
            warmup_scheduler.step()
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / max(1, batch_count)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
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
        
        # Calculate average validation loss
        avg_val_loss = val_loss / max(1, val_batch_count)
        val_losses.append(avg_val_loss)
        
        # Calculate mAP on validation set (using a subset for speed)
        print("Calculating validation mAP...")
        val_map = calculate_ap(model, val_loader, device)
        val_maps.append(val_map)
        
        # Update main learning rate scheduler based on validation loss
        lr_scheduler.step(avg_val_loss)
        
        # Print epoch summary
        t_end = datetime.now()
        duration = t_end - t_start
        print(f"  Epoch {epoch+1} completed in {duration}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val mAP: {val_map:.4f}")
        
        # Save checkpoint if this is the best model so far
        if val_map > best_val_map:
            best_val_map = val_map
            patience_counter = 0
            
            print(f"  Saving best model with mAP: {best_val_map:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_map': val_map,
                'val_loss': avg_val_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            print(f"  No improvement in val_map for {patience_counter} epochs")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_map': val_map,
                    'val_loss': avg_val_loss,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_map': val_maps[-1],
        'val_loss': val_losses[-1],
    }, os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # Return training history
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maps': val_maps
    }

# Function to plot training history
def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation losses
    plt.subplot(2, 1, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation mAP
    plt.subplot(2, 1, 2)
    plt.plot(history['val_maps'], label='Validation mAP', color='green')
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
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
        
        # Decode predictions
        decoded_boxes = decode_boxes(loc_preds[0], default_boxes)
        
        # Get confidence scores
        scores = torch.nn.functional.softmax(conf_preds[0], dim=1)
        
        # Get detections with confidence > threshold
        detections = []
        
        for class_idx in range(1, num_classes):  # Skip background class
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

# Train the model
print("Starting model training...")
history = train_model(
    model=model,
    loss_fn=SSDLoss,
    optimizer=optimizer,
    warmup_scheduler=warmup_scheduler,
    lr_scheduler=lr_scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=35
)

# Plot training history
plot_training_history(history)

# Load the best model and evaluate
best_model_path = os.path.join('checkpoints', 'best_model.pth')
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model with validation mAP: {checkpoint['val_map']:.4f}")
    
    # Calculate final mAP on full validation set
    final_map = calculate_ap(model, val_loader, device)
    print(f"Final mAP on validation set: {final_map:.4f}")
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

# Example of how to use the model for inference
# Replace 'path/to/test/image.jpg' with an actual image path
# visualize_detections(model, 'path/to/test/image.jpg')