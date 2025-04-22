# SSD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
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
    def __init__(self, root, year='2007', image_set='train', transforms=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transforms = transforms
        
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
        img, boxes, labels = self.load_image_and_labels(index)
        
        sample = {'image': img, 'bboxes': boxes, 'labels': labels}
        if self.transforms:
            sample = self.transforms(**sample)
        
        # Normalize bounding boxes assuming the image is resized to 300x300
        normalized_boxes = []
        for box in sample['bboxes']:
            xmin, ymin, xmax, ymax = box
            nbox = [xmin / 300, ymin / 300, xmax / 300, ymax / 300]
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

# Simplified SSD300-style augmentations (from the original paper)
train_transforms = A.Compose([
    A.Resize(300, 300),
    A.HorizontalFlip(p=0.5),                             
    A.ColorJitter(brightness=0.125, contrast=0.125,
                  saturation=0.125, hue=0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),               
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    min_visibility=0.5,
    label_fields=['labels']
))

val_transforms = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['labels']
))

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

# Combine VOC2007 + VOC2012 trainval
voc_root = '/home/adrian/ssd/VOCdevkit/VOCdevkit'

dataset_07 = PascalVOCDataset(
    voc_root, year='2007', image_set='trainval',
    transforms=train_transforms
)
dataset_12 = PascalVOCDataset(
    voc_root, year='2012', image_set='trainval',
    transforms=train_transforms
)
train_dataset = ConcatDataset([dataset_07, dataset_12])

# Validation and test sets remain VOC2007-val/test

val_dataset = PascalVOCDataset(
    voc_root, year='2007', image_set='val',
    transforms=val_transforms
)
test_dataset = PascalVOCDataset(
    voc_root, year='2007', image_set='test',
    transforms=val_transforms
)

# Dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True
)

class L2Norm(nn.Module):
    """L2‐normalization layer (SSD style)"""
    def __init__(self, n_channels, scale=20.0, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        # x: (N, C, H, W)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=self.eps)
        x = x / norm
        return x * self.weight.view(1, -1, 1, 1)

class SSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # 1) Backbone: ResNet-50
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.stage1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )  # 75×75 → C=256
        self.stage2 = resnet.layer2  # 38×38 → C=512
        self.stage3 = resnet.layer3  # 19×19 → C=1024
        self.stage4 = resnet.layer4  # 10×10 → C=2048

        # L2Norm on the 38×38 map
        self.l2norm = L2Norm(n_channels=512, scale=20.0)

        # 2) Extra layers for smaller maps
        self.extra1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True)
        )  # → 5×5
        self.extra2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True)
        )  # → 3×3
        self.extra3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True)
        )  # → 1×1

        # 3) Placeholder for heads; build them once we know map sizes
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        # Build heads *and* anchor buffers
        self._build_heads_and_anchors()

    def _build_heads_and_anchors(self):
        # use a dummy forward to get feature‐map shapes
        with torch.no_grad():
            x = torch.zeros(1, 3, 300, 300)
            f1 = self.stage1(x)
            f2 = self.stage2(f1); f2 = self.l2norm(f2)
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)
            e1 = self.extra1(f4)
            e2 = self.extra2(e1)
            e3 = self.extra3(e2)
            feats = [f2, f3, f4, e1, e2, e3]

        # feature map sizes (H=W)
        self.feature_maps = [f.shape[2] for f in feats]
        # SSD300 scales & ratios
        self.scales       = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]
        self.aspect_ratios= [[2, 0.5]] * 6

        # 3a) create & register default boxes
        boxes = []
        for k, f in enumerate(self.feature_maps):
            s_k  = self.scales[k]
            s_k1 = self.scales[k+1] if k+1 < len(self.scales) else 1.0
            for i in range(f):
                for j in range(f):
                    cx = (j + .5)/f; cy = (i + .5)/f
                    # two scales
                    boxes.append([cx, cy, s_k, s_k])
                    boxes.append([cx, cy, (s_k*s_k1)**.5, (s_k*s_k1)**.5])
                    # aspect ratios
                    for ar in self.aspect_ratios[k]:
                        boxes.append([cx, cy, s_k*ar**.5, s_k/ar**.5])
                        boxes.append([cx, cy, s_k/ar**.5, s_k*ar**.5])
        db = torch.tensor(boxes)  # (M,4) center‐size
        db = db.clamp(0,1)
        # center→corner
        cx, cy, w, h = db.unbind(dim=1)
        db_xyxy = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=1).clamp(0,1)
        # register as buffers so .to(device) moves them
        self.register_buffer('default_boxes', db)
        self.register_buffer('default_boxes_xyxy', db_xyxy)

        # 3b) build prediction heads
        num_anchors = [2 + 2*len(ar) for ar in self.aspect_ratios]
        channels    = [f.shape[1] for f in feats]
        for ch, a in zip(channels, num_anchors):
            self.loc_layers.append (nn.Conv2d(ch, a*4,          kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(ch, a*self.num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        # backbone
        x1 = self.stage1(x)
        x2 = self.l2norm(self.stage2(x1))
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # extras
        e1 = self.extra1(x4)
        e2 = self.extra2(e1)
        e3 = self.extra3(e2)
        feats = [x2, x3, x4, e1, e2, e3]

        # heads
        locs, confs = [], []
        for f, l, c in zip(feats, self.loc_layers, self.conf_layers):
            loc = l(f).permute(0,2,3,1).reshape(x.size(0), -1, 4)
            conf= c(f).permute(0,2,3,1).reshape(x.size(0), -1, self.num_classes)
            locs.append(loc); confs.append(conf)

        # return (N,ΣA·HW,4), (N,ΣA·HW,C), priors
        return torch.cat(locs,  dim=1), \
               torch.cat(confs, dim=1), \
               self.default_boxes_xyxy

# SSD Box Encoding and Decoding
def encode_boxes(matched_boxes, default_boxes, variances=(0.1, 0.2)):
    """
    Encode ground truth boxes relative to default boxes (SSD paper).
    matched_boxes, default_boxes: tensors of shape (N,4) in corner form [xmin,ymin,xmax,ymax]
    variances: tuple(float,int)
    returns: offsets tensor (N,4) [dx, dy, dw, dh]
    """
    # Convert to center-size
    g_cx = (matched_boxes[:, 0] + matched_boxes[:, 2]) / 2
    g_cy = (matched_boxes[:, 1] + matched_boxes[:, 3]) / 2
    g_w  = matched_boxes[:, 2] - matched_boxes[:, 0]
    g_h  = matched_boxes[:, 3] - matched_boxes[:, 1]

    d_cx = (default_boxes[:, 0] + default_boxes[:, 2]) / 2
    d_cy = (default_boxes[:, 1] + default_boxes[:, 3]) / 2
    d_w  = default_boxes[:, 2] - default_boxes[:, 0]
    d_h  = default_boxes[:, 3] - default_boxes[:, 1]

    # encode offsets
    dx = (g_cx - d_cx) / (variances[0] * d_w)
    dy = (g_cy - d_cy) / (variances[0] * d_h)
    dw = torch.log(g_w / d_w) / variances[1]
    dh = torch.log(g_h / d_h) / variances[1]

    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(loc_preds, default_boxes, variances=(0.1, 0.2)):
    """
    Decode predicted offsets to bounding boxes (SSD paper).
    loc_preds: tensor (N,4) [dx,dy,dw,dh]
    default_boxes: tensor (N,4) priors in corner form
    returns: decoded boxes (N,4) corner form
    """
    # default center-size
    d_cx = (default_boxes[:, 0] + default_boxes[:, 2]) / 2
    d_cy = (default_boxes[:, 1] + default_boxes[:, 3]) / 2
    d_w  = default_boxes[:, 2] - default_boxes[:, 0]
    d_h  = default_boxes[:, 3] - default_boxes[:, 1]

    # apply offsets
    p_cx = loc_preds[:, 0] * variances[0] * d_w + d_cx
    p_cy = loc_preds[:, 1] * variances[0] * d_h + d_cy
    p_w  = torch.exp(loc_preds[:, 2] * variances[1]) * d_w
    p_h  = torch.exp(loc_preds[:, 3] * variances[1]) * d_h

    # to corner form
    xmin = p_cx - p_w / 2
    ymin = p_cy - p_h / 2
    xmax = p_cx + p_w / 2
    ymax = p_cy + p_h / 2
    decoded = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    return decoded.clamp(0, 1)


# SSD MultiBox Loss
class MultiBoxLoss(nn.Module):
    """
    SSD Multibox loss: Smooth L1 for loc + CrossEntropy for conf
    with hard negative mining (3:1 ratio).
    """
    def __init__(self, default_boxes, iou_threshold=0.5, neg_pos_ratio=3):
        super().__init__()
        self.default_boxes = default_boxes  # (M,4) tensor corner form
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.loc_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.conf_loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets):
        # predictions: (loc_preds, conf_preds)
        loc_preds, conf_preds = predictions
        batch_size = loc_preds.size(0)
        num_priors = self.default_boxes.size(0)
        num_classes = conf_preds.size(2)

        # Prepare target tensors
        loc_t = torch.zeros((batch_size, num_priors, 4), device=loc_preds.device)
        conf_t = torch.zeros((batch_size, num_priors), dtype=torch.long, device=loc_preds.device)

        for b in range(batch_size):
            truths = targets['boxes'][b]  # (n_obj,4)
            labels = targets['labels'][b]  # (n_obj,)
            if truths.numel() == 0:
                continue
            # match priors to truths
            ious = box_iou(self.default_boxes, truths)
            best_truth_iou, best_truth_idx = ious.max(dim=1)
            best_prior_iou, best_prior_idx = ious.max(dim=0)
            best_truth_iou[best_prior_idx] = 1.0
            best_truth_idx[best_prior_idx] = torch.arange(best_prior_idx.size(0), device=ious.device)

            matches = truths[best_truth_idx]
            conf = labels[best_truth_idx]
            conf[best_truth_iou < self.iou_threshold] = 0

            loc_t[b] = encode_boxes(matches, self.default_boxes)
            conf_t[b] = conf

        # localization loss (positives)
        pos_mask = conf_t > 0
        num_pos = pos_mask.sum().clamp(min=1)
        loc_p = loc_preds[pos_mask].view(-1, 4)
        loc_g = loc_t[pos_mask].view(-1, 4)
        loc_loss = self.loc_loss_fn(loc_p, loc_g).sum()

        # confidence loss (all)
        conf_flat = conf_preds.view(-1, num_classes)
        conf_t_flat = conf_t.view(-1)
        conf_loss_all = self.conf_loss_fn(conf_flat, conf_t_flat)
        conf_loss_all = conf_loss_all.view(batch_size, num_priors)

        # hard negative mining
        conf_loss_all[pos_mask] = 0
        _, idx_rank = conf_loss_all.sort(dim=1, descending=True)
        _, idx_rank = idx_rank.sort(dim=1)
        num_neg = (self.neg_pos_ratio * pos_mask.sum(dim=1)).unsqueeze(1)
        neg_mask = idx_rank < num_neg

        conf_mask = pos_mask | neg_mask
        conf_p = conf_flat[conf_mask.view(-1)]
        conf_t2 = conf_t_flat[conf_mask.view(-1)]
        conf_loss = self.conf_loss_fn(conf_p, conf_t2).sum()

        total_loss = (loc_loss + conf_loss) / num_pos.float()
        return total_loss

# Instantiate model & loss
model = SSD(num_classes=num_classes).to(device)
criterion = MultiBoxLoss(
    default_boxes=model.default_boxes_xyxy.to(device),
    iou_threshold=0.5,
    neg_pos_ratio=3
)

# Optimizer and iteration‐based LR schedule
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# Compute iterations per epoch
iters_per_epoch = len(train_loader)
# Milestones at 80k and 100k iterations
milestones = [80_000 // iters_per_epoch, 100_000 // iters_per_epoch]

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=milestones,
    gamma=0.1
)

# mAP@0.50 evaluator
def evaluate_map50(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], box_format="xyxy").to(device)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch['images'].to(device)
            loc_preds, conf_preds, priors = model(imgs)

            # per‑image unpack
            for i in range(imgs.size(0)):
                # decode boxes
                boxes   = decode_boxes(loc_preds[i], priors)
                scores  = torch.softmax(conf_preds[i], dim=1)
                
                # gather per‑class detections
                preds = {'boxes': [], 'scores': [], 'labels': []}
                for c in range(1, num_classes):
                    mask = scores[:,c] > 0.01
                    if mask.sum()==0: continue
                    cls_boxes  = boxes[mask]
                    cls_scores = scores[mask,c]
                    keep       = torchvision.ops.nms(cls_boxes, cls_scores, iou_threshold=0.45)
                    preds['boxes'].append(cls_boxes[keep].cpu())
                    preds['scores'].append(cls_scores[keep].cpu())
                    preds['labels'].append(torch.full((len(keep),), c, dtype=torch.int64))

                if preds['boxes']:
                    all_preds.append({
                        'boxes': torch.cat(preds['boxes'],dim=0),
                        'scores': torch.cat(preds['scores'],dim=0),
                        'labels': torch.cat(preds['labels'],dim=0)
                    })
                else:
                    all_preds.append({'boxes':torch.zeros(0,4), 'scores':torch.zeros(0), 'labels':torch.zeros(0,dtype=torch.int64)})

                all_targets.append({
                    'boxes': batch['boxes'][i].cpu(),
                    'labels': batch['labels'][i].cpu()
                })

    metric.update(all_preds, all_targets)
    mAP50 = metric.compute()['map_50'].item()
    return mAP50

def train_model(
    model,
    criterion,        # instance of MultiBoxLoss
    optimizer,
    scheduler,        # MultiStepLR(milestones=[80,100], gamma=0.1)
    train_loader,
    val_loader,
    device,
    epochs=120,
    checkpoint_dir='checkpoints'
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_map50 = 0.0

    def evaluate_map50():
        model.eval()
        metric = MeanAveragePrecision(iou_thresholds=[0.5], box_format="xyxy").to(device)
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['images'].to(device)
                loc_preds, conf_preds, priors = model(imgs)

                for i in range(imgs.size(0)):
                    # decode & nms per class
                    boxes  = decode_boxes(loc_preds[i], priors)
                    scores = torch.softmax(conf_preds[i], dim=1)
                    preds  = {'boxes':[], 'scores':[], 'labels':[]}
                    for c in range(1, model.num_classes):
                        mask = scores[:,c] > 0.01
                        if not mask.any(): continue
                        cls_b = boxes[mask]
                        cls_s = scores[mask,c]
                        keep  = torchvision.ops.nms(cls_b, cls_s, iou_threshold=0.45)
                        preds['boxes'].append(cls_b[keep].cpu())
                        preds['scores'].append(cls_s[keep].cpu())
                        preds['labels'].append(torch.full((len(keep),), c, dtype=torch.int64))
                    if preds['boxes']:
                        all_preds.append({
                            'boxes':  torch.cat(preds['boxes'], dim=0),
                            'scores': torch.cat(preds['scores'],dim=0),
                            'labels': torch.cat(preds['labels'],dim=0)
                        })
                    else:
                        all_preds.append({'boxes':torch.zeros(0,4), 'scores':torch.zeros(0), 'labels':torch.zeros(0,dtype=torch.int64)})
                    all_targets.append({
                        'boxes': batch['boxes'][i].cpu(),
                        'labels': batch['labels'][i].cpu()
                    })
        metric.update(all_preds, all_targets)
        return metric.compute()['map_50'].item()

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            imgs = batch['images'].to(device)
            # move each target tensor to the same device
            targets = {
                'boxes':  [b.to(device) for b in batch['boxes']],
                'labels': [l.to(device) for l in batch['labels']]
            }
            optimizer.zero_grad()
            loc_preds, conf_preds, priors = model(imgs)
            loss = criterion((loc_preds, conf_preds), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        msg = f"Epoch {epoch:3d}/{epochs}  train_loss: {avg_loss:.4f}"

        if epoch % 5 == 0 or epoch == epochs:
            map50 = evaluate_map50()
            msg += f"  val_mAP@0.50: {map50:.4f}"
            # save best
            if map50 > best_map50:
                best_map50 = map50
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_dir, 'best_model.pth'))
        print(msg)

    # final save
    torch.save(model.state_dict(),
               os.path.join(checkpoint_dir, 'final_model.pth'))

# Evaluation (mAP@0.50)
def evaluate_map50(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], box_format="xyxy").to(device)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['images'].to(device)
            loc_preds, conf_preds, priors = model(imgs)
            for i in range(imgs.size(0)):
                boxes = decode_boxes(loc_preds[i], priors)
                scores = torch.softmax(conf_preds[i], dim=1)
                preds = {'boxes': [], 'scores': [], 'labels': []}
                for c in range(1, model.num_classes):
                    mask = scores[:,c] > 0.01
                    if not mask.any(): continue
                    b = boxes[mask]; s = scores[mask,c]
                    keep = torchvision.ops.nms(b, s, iou_threshold=0.45)
                    preds['boxes'].append(b[keep].cpu())
                    preds['scores'].append(s[keep].cpu())
                    preds['labels'].append(torch.full((len(keep),), c, dtype=torch.int64))
                if preds['boxes']:
                    all_preds.append({
                        'boxes': torch.cat(preds['boxes'], dim=0),
                        'scores':torch.cat(preds['scores'],dim=0),
                        'labels':torch.cat(preds['labels'],dim=0)
                    })
                else:
                    all_preds.append({'boxes':torch.zeros(0,4), 'scores':torch.zeros(0), 'labels':torch.zeros(0, dtype=torch.int64)})
                all_targets.append({
                    'boxes': batch['boxes'][i].cpu(),
                    'labels': batch['labels'][i].cpu()
                })
    metric.update(all_preds, all_targets)
    return metric.compute()['map_50'].item()

# Training Loop
def train_model(
    model, criterion, optimizer, scheduler,
    train_loader, val_loader, device,
    epochs=120, checkpoint_dir='checkpoints'
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_map50 = 0.0
    history = {'train_losses': [], 'val_map50': [], 'val_epochs': []}

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            imgs = batch['images'].to(device)
            targets = {
                'boxes':  [b.to(device) for b in batch['boxes']],
                'labels': [l.to(device) for l in batch['labels']]
            }
            optimizer.zero_grad()
            loc_preds, conf_preds, priors = model(imgs)
            loss = criterion((loc_preds, conf_preds), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        history['train_losses'].append(avg_loss)

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            map50 = evaluate_map50(model, val_loader, device)
            history['val_map50'].append(map50)
            history['val_epochs'].append(epoch)
            if map50 > best_map50:
                best_map50 = map50
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, val_mAP@0.50={map50:.4f}")
        else:
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}")

    # final save
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_model.pth'))
    return history

# Plotting
def plot_training_history(history):
    epochs = list(range(1, len(history['train_losses'])+1))
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history['train_losses'], label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.legend(); plt.grid(True)
    plt.show()

    if history['val_map50']:
        plt.figure(figsize=(8,5))
        plt.plot(history['val_epochs'], history['val_map50'], marker='o', label='Val mAP@0.50')
        plt.xlabel('Epoch'); plt.ylabel('mAP@0.50'); plt.title('Validation mAP@0.50'); plt.legend(); plt.grid(True)
        plt.show()

# Visualization
def visualize_detections(model, image_path, transform=None, conf_threshold=0.5):
    import numpy as np
    import matplotlib.patches as patches
    from PIL import Image
    model.eval()
    img = Image.open(image_path).convert('RGB')
    orig = np.array(img)
    if transform is None:
        transform = A.Compose([A.Resize(300,300), ToTensorV2()])
    inp = transform(image=orig)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        locs, confs, priors = model(inp)
        boxes = decode_boxes(locs[0], priors)
        scores = torch.softmax(confs[0], dim=1)

    plt.figure(figsize=(8,8)); ax = plt.gca(); ax.imshow(orig)
    for c in range(1, num_classes):
        mask = scores[:,c] > conf_threshold
        if not mask.any(): continue
        b = boxes[mask]; s = scores[mask,c]
        keep = torchvision.ops.nms(b, s, 0.45)
        for idx in keep:
            xmin,ymin,xmax,ymax = b[idx].cpu().numpy()
            h,w = orig.shape[:2]
            rect = patches.Rectangle((xmin*w, ymin*h), (xmax-xmin)*w, (ymax-ymin)*h,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin*w, ymin*h-2, f"{VOC_CLASSES[c]}:{s[idx]:.2f}",
                    color='white', fontsize=8, backgroundcolor='r')
    plt.axis('off'); plt.show()

print("Starting SSD300 training...")
history = train_model(
    model, criterion, optimizer, scheduler,
    train_loader, val_loader, device,
    epochs=120
)
plot_training_history(history)
# load best
if os.path.exists('checkpoints/best_model.pth'):
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    print("Loaded best model.")

# test
test_map50 = evaluate_map50(model, test_loader, device)
print(f"Test mAP@0.50: {test_map50:.4f}")