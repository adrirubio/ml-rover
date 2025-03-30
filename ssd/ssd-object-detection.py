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

# Load PASCAL VOC 2007 - both train and validation sets
voc_dataset = load_dataset("voc", "2007")

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
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        boxes.append([x_min, y_min, x_max, y_max])
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
mapped_train_dataset = train_dataset.map(train_mapper, remove_columns=["image", "objects"])
mapped_val_dataset = val_dataset.map(val_mapper, remove_columns=["image", "objects"])

# Create DataLoaders
train_loader = DataLoader(
    mapped_train_dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=4,  # Using 4 parallel workers
    collate_fn=custom_collate_fn  
)   

val_loader = DataLoader(
    mapped_val_dataset, 
    batch_size=16, 
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

# Instantiate the SSD model
num_classes = 21
model = SSD(num_classes=num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def box_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) between two sets of boxes.
    
    Args:
        boxes1 (torch.Tensor): Shape (N, 4)
        boxes2 (torch.Tensor): Shape (M, 4)
    
    Returns:
        torch.Tensor: IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter  # [N, M]
    
    return inter / union  # IoU

def encode_boxes(matched_boxes, default_boxes):
    """
    Encode ground truth boxes relative to default boxes.
    
    Args:
        matched_boxes (torch.Tensor): Ground truth boxes (N, 4)
        default_boxes (torch.Tensor): Default anchor boxes (N, 4)
    
    Returns:
        torch.Tensor: Encoded box locations
    """
    def box_to_centerwidth(boxes):
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + width / 2
        ctr_y = boxes[:, 1] + height / 2
        return torch.stack([ctr_x, ctr_y, width, height], dim=1)
    
    g_cxcy = box_to_centerwidth(matched_boxes)
    d_cxcy = box_to_centerwidth(default_boxes)
    
    encoded_boxes = torch.zeros_like(g_cxcy)
    encoded_boxes[:, :2] = (g_cxcy[:, :2] - d_cxcy[:, :2]) / (d_cxcy[:, 2:] + 1e-8)
    encoded_boxes[:, 2:] = torch.log(g_cxcy[:, 2:] / (d_cxcy[:, 2:] + 1e-8))
    
    return encoded_boxes

class SSD_loss(nn.Module):
    def __init__(self, num_classes, default_boxes, device):
        """
        SSD Loss function

        Args:
            num_classes (int): Number of object classes
            default_boxes (torch.Tensor): Default anchor boxes
            device (torch.device): GPU or CPU
        """
        super(SSD_loss, self).__init__()
        
        self.num_classes = num_classes
        self.default_boxes = default_boxes.to(device)
        self.device = device
        
        self.threshold = 0.5  # IoU threshold for positive matches
        self.neg_pos_ratio = 3  # Ratio of negative to positive samples
        
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Compute SSD loss.

        Args:
            predictions (tuple): (loc_preds, conf_preds)
                - loc_preds: Shape (batch_size, num_priors, 4) (Predicted box offsets)
                - conf_preds: Shape (batch_size, num_priors, num_classes) (Class probabilities)
            targets (dict): {"boxes": [list of GT boxes], "labels": [list of GT labels]}
        
        Returns:
            torch.Tensor: Total loss
        """
        loc_preds, conf_preds = predictions
        batch_size = loc_preds.size(0)
        num_priors = self.default_boxes.size(0)
        
        loc_t = torch.zeros(batch_size, num_priors, 4).to(self.device)
        conf_t = torch.zeros(batch_size, num_priors, dtype=torch.long).to(self.device)
        
        for idx in range(batch_size):
            truths = targets['boxes'][idx]
            labels = targets['labels'][idx]
            
            if truths.size(0) == 0:  # No objects in image
                continue
            
            overlaps = box_iou(self.default_boxes, truths)
            best_truth_overlap, best_truth_idx = overlaps.max(1)  
            
            best_prior_overlap, best_prior_idx = overlaps.max(0)  
            for j in range(best_prior_idx.size(0)):
                best_truth_idx[best_prior_idx[j]] = j
                best_truth_overlap[best_prior_idx[j]] = 2  

            matches = truths[best_truth_idx]  
            conf = labels[best_truth_idx]  
            conf[best_truth_overlap < self.threshold] = 0  
            
            loc_t[idx] = encode_boxes(matches, self.default_boxes)
            conf_t[idx] = conf
        
        pos = conf_t > 0
        loc_loss = self.smooth_l1(loc_preds[pos], loc_t[pos]).sum()
        
        conf_loss = self.cross_entropy(conf_preds.view(-1, self.num_classes), conf_t.view(-1))
        conf_loss = conf_loss.view(batch_size, -1)
        conf_loss[pos] = 0  

        _, loss_idx = conf_loss.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=pos.size(1) - 1)
        
        neg = idx_rank < num_neg.unsqueeze(1)
        
        pos_conf_loss = conf_loss[pos].sum()
        neg_conf_loss = conf_loss[neg].sum()
        
        epsilon = 1e-6
        loss = (loc_loss + pos_conf_loss + neg_conf_loss) / (num_pos.sum().float() + epsilon)
        
        return loss

# Initialize the SSD loss function
SSDLoss = SSD_loss(
    num_classes=num_classes, 
    default_boxes=model.default_boxes, 
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
            loc_preds, conf_preds = model(images)
            
            # Compute loss
            loss = SSDLoss((loc_preds, conf_preds), {'boxes': boxes, 'labels': labels})
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        
        # Get train loss mean
        train_loss = np.mean(train_loss)

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
                loc_preds, conf_preds = model(images)
                
                # Compute loss
                loss = SSDLoss((loc_preds, conf_preds), {'boxes': boxes, 'labels': labels})

                val_loss.append(loss.item())
        
        # Get validation loss mean
        val_loss = np.mean(val_loss)

        # Save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
            Validation Loss: {val_loss:.4f}, Duration: {dt}')

    return train_losses, val_losses

train_losses, val_losses = batch_gd(model, SSDLoss, optimizer, train_loader, val_loader, epochs=70)

# Loss train and test loss
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='test loss')
plt.legend()
plt.show()

# Save model
model_save_path = "ssd-object-detection.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")