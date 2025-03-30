# SSD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the Pascal VOC dataset
voc_dataset = load_dataset("EduardoLawson1/Pascal_voc")
 
# Split the train split into 90% train and 10% validation
split_dataset = voc_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# Define the VOC class labels
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Define transforms with proper Albumentations pipeline
train_transforms = A.Compose([
    A.Resize(300, 300),  # Simple resize instead of RandomResizedCrop
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

val_transforms = A.Compose([
    A.Resize(width=300, height=300),  # For resize, width and height are separate parameters
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def convert_to_ssd_format(example, transform_fn):
    # Convert image to numpy array
    img = np.array(example["image"])
    
    # Extract label
    label = example.get("label", 0)
    
    # Create bounding box for the whole image
    height, width = img.shape[:2]
    boxes = [[0, 0, width, height]]  # Default box covering whole image
    labels = [label]
    
    # Apply transform
    transformed = transform_fn(image=img, bboxes=boxes, labels=labels)
    
    # Extract and convert results to tensors
    image = transformed['image']  # Already a tensor from ToTensorV2
    
    # Fast tensor conversion with proper handling of empty results
    if transformed['bboxes']:
        boxes_tensor = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        labels_tensor = torch.tensor(transformed['labels'], dtype=torch.int64)
    else:
        # Fallback for cases where transforms removed all boxes
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros(0, dtype=torch.int64)
    
    return {
        "image": image,
        "boxes": boxes_tensor,
        "labels": labels_tensor
    }

# Create mapping functions for training and validation datasets
def train_mapper(example):
    return convert_to_ssd_format(example, train_transforms)

def val_mapper(example):
    return convert_to_ssd_format(example, val_transforms)

# Define a collate function that handles variable-sized objects properly
def custom_collate_fn(batch):
    images = []
    boxes = []
    labels = []
    
    for sample in batch:
        images.append(sample["image"])
        boxes.append(sample["boxes"])
        labels.append(sample["labels"])
    
    # Stack images into a single tensor - move to device later for efficiency
    images = torch.stack(images, 0)
    
    return {
        "images": images,
        "boxes": boxes,  # List of tensors with different shapes per image
        "labels": labels  # List of tensors with different shapes per image
    }

# Map the datasets
mapped_train_dataset = train_dataset.map(
    train_mapper,  
    with_indices=False,
    remove_columns=["image", "label"],
    num_proc=8
)

mapped_val_dataset = val_dataset.map(
    val_mapper,
    with_indices=False,
    remove_columns=["image", "label"],
    num_proc=8
)

# Create DataLoaders
train_loader = DataLoader(
    mapped_train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,  # Using 4 parallel workers
    collate_fn=custom_collate_fn  
)   

val_loader = DataLoader(
    mapped_val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=4,
    collate_fn=custom_collate_fn
)

# Define model 
class SSD(nn.Module):
    def __init__(self, num_classes=21):
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
        
        # Define location layers (with correct output channels)
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512, self.num_anchors[0] * 4, kernel_size=3, padding=1),  # For conv1
            nn.Conv2d(512, self.num_anchors[1] * 4, kernel_size=3, padding=1),  # For conv2
            nn.Conv2d(256, self.num_anchors[2] * 4, kernel_size=3, padding=1),  # For conv3
            nn.Conv2d(256, self.num_anchors[3] * 4, kernel_size=3, padding=1),  # For conv4
            nn.Conv2d(256, self.num_anchors[4] * 4, kernel_size=3, padding=1),  # For conv5
            nn.Conv2d(256, self.num_anchors[5] * 4, kernel_size=3, padding=1)   # For conv6
        ])

        # Define confidence layers (with correct output channels)
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
            conf = conf.view(batch_size, -1, self.num_classes)
            conf_preds.append(conf)

        # Concatenate predictions from different feature maps
        loc_preds = torch.cat(loc_preds, dim=1)
        conf_preds = torch.cat(conf_preds, dim=1)
        
        return loc_preds, conf_preds, self.default_boxes_xyxy

# Instantiate the SSD model and send it to the GPU
num_classes = 21  # 20 object classes + 1 background class
model = SSD(num_classes=num_classes)
model.to(device)

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

# Initialize the SSD loss function
SSDLoss = SSD_loss(
    num_classes=num_classes, 
    default_boxes=model.default_boxes_xyxy,  # Use boxes in corner format
    device=device
)

# Freeze first 10 layers of the VGG backbone
for idx, param in enumerate(model.conv1.parameters()):
    layer_idx = idx // 2  # Each layer has weights and biases, so divide by 2
    if layer_idx < 10:    # First 10 layers
        param.requires_grad = False

# Define optimizer with lower learning rate in conv 1 and 2 (backbone)
optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 0.0001, 'weight_decay': 1e-4},  # Lower learning rate
    {'params': model.conv2.parameters(), 'lr': 0.0001, 'weight_decay': 1e-4},  # Lower learning rate
    {'params': model.conv3.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': model.conv4.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': model.conv5.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': model.conv6.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': model.loc_layers.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': model.conf_layers.parameters(), 'lr': 0.001, 'weight_decay': 1e-4}
], betas=(0.9, 0.999))

# Training loop
def batch_gd(model, SSDLoss, optimizer, train_loader, val_loader, epochs):
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []

        for batch in train_loader:
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
            
            # Compute loss
            loss = SSDLoss((loc_preds, conf_preds, default_boxes), {'boxes': boxes, 'labels': labels})
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        
        # Get train loss mean
        train_loss = np.mean(train_loss)

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
                
                # Forward pass
                loc_preds, conf_preds, default_boxes = model(images)
                
                # Compute loss
                loss = SSDLoss((loc_preds, conf_preds, default_boxes), {'boxes': boxes, 'labels': labels})

                val_loss.append(loss.item())
        
        # Get validation loss mean
        val_loss = np.mean(val_loss)

        # Save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Duration: {dt}')

    return train_losses, val_losses

# Train the model
train_losses, val_losses = batch_gd(model, SSDLoss, optimizer, train_loader, val_loader, epochs=70)

# Plot training and validation losses
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='test loss')
plt.legend()
plt.show()

# Save the model
model_save_path = "ssd-object-detection.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")