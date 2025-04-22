"""
SSD300 Random VOC2007 Inference

This script runs inference on a random image from the Pascal VOC 2007 dataset
using the pre-trained SSD300 model.

Usage:
    python Pascal-VOC-inference.py

Requirements:
    - Pascal VOC 2007 dataset (trainval + test)
    - Pre-trained SSD model weights
    - Install dependencies

How to set up:

1. Download the Pascal VOC 2007 dataset:

    Create a directory for the dataset:
        mkdir -p /path/to/VOCdevkit
        cd /path/to/VOCdevkit

    Download and extract the dataset:
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        tar -xf VOCtrainval_06-Nov-2007.tar
        tar -xf VOCtest_06-Nov-2007.tar

    Set the dataset path in the script:
        VOC_ROOT = "/path/to/VOCdevkit/VOCdevkit"

2. Download SSD300 model weights:

    Download the 'ssd_weights.pth' file from:
        https://huggingface.co/pro-grammer/SSD/

    Set the weights path in the script:
        WEIGHTS_PATH = "/path/to/your/ssd_weights.pth"

3. Install dependencies:
    - **Python Libraries**: The following libraries are required for running this script:
        - `torch`
        - `torchvision`
        - `albumentations`
        - `matplotlib`
        - `PIL`
  
  You can install the necessary dependencies using `pip`:

  ```bash
  pip install torch torchvision albumentations matplotlib pillow

Make sure to update VOC_ROOT and WEIGHTS_PATH accordingly before running the script.
"""

import os, random
import torch, torch.nn as nn, torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import nms

# Paths & classes
VOC_ROOT = "/home/adrian/Documents/VOCdevkit/VOCdevkit" # Modify for your own use case
WEIGHTS_PATH = "/home/adrian/Documents/model-weights/ssd_weights.pth" # Modify for your own use case
VOC_CLASSES = (
    'background','aeroplane','bicycle','bird','boat','bottle','bus',
    'car','cat','chair','cow','diningtable','dog','horse','motorbike',
    'person','pottedplant','sheep','sofa','train','tvmonitor'
)
NUM_CLASSES = len(VOC_CLASSES)

# L2Norm layer
class L2Norm(nn.Module):
    def __init__(self, C, scale=20.0, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(C)*scale)
    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt().clamp(min=self.eps)
        return x / norm * self.weight.view(1,-1,1,1)

# SSD300‑ResNet50 backbone + heads
class SSD300(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.stage1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1
        )
        self.stage2, self.stage3, self.stage4 = (
            resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.l2norm = L2Norm(512)
        self.extra1 = nn.Sequential(
            nn.Conv2d(2048,512,1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,stride=2,padding=1), nn.ReLU(inplace=True)
        )
        self.extra2 = nn.Sequential(
            nn.Conv2d(512,256,1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,stride=2,padding=1), nn.ReLU(inplace=True)
        )
        self.extra3 = nn.Sequential(
            nn.Conv2d(256,256,3,stride=2,padding=1), nn.ReLU(inplace=True)
        )
        self._build_heads_and_priors()

    def _build_heads_and_priors(self):
        # Dummy forward to get feature sizes
        with torch.no_grad():
            x = torch.zeros(1,3,300,300)
            f1 = self.stage1(x)
            f2 = self.l2norm(self.stage2(f1))
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)
            e1 = self.extra1(f4)
            e2 = self.extra2(e1)
            e3 = self.extra3(e2)
            feats = [f2,f3,f4,e1,e2,e3]

        # Build priors
        scales = [0.2,0.34,0.48,0.62,0.76,0.9]
        ars    = [[2,0.5]]*6
        priors = []
        for k, f in enumerate(feats):
            sz = f.shape[2]
            s_k, s_k1 = scales[k], scales[k+1] if k+1<len(scales) else 1.0
            for i in range(sz):
                for j in range(sz):
                    cx,cy = (j+0.5)/sz, (i+0.5)/sz
                    priors += [[cx,cy,s_k,s_k],
                               [cx,cy,(s_k*s_k1)**0.5,(s_k*s_k1)**0.5]]
                    for ar in ars[k]:
                        priors += [[cx,cy,s_k*ar**0.5, s_k/ar**0.5],
                                   [cx,cy,s_k/ar**0.5,s_k*ar**0.5]]
        p = torch.tensor(priors).clamp(0,1)
        cx,cy,w,h = p.unbind(1)
        p_xy = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2],1).clamp(0,1)
        self.register_buffer('priors', p_xy)

        # Build heads
        num_anchors = [2+2*len(a) for a in ars]
        channels = [f.shape[1] for f in feats]
        self.loc_layers = nn.ModuleList()
        self.conf_layers= nn.ModuleList()
        for ch, na in zip(channels, num_anchors):
            self.loc_layers.append (nn.Conv2d(ch, na*4,    3, padding=1))
            self.conf_layers.append(nn.Conv2d(ch, na*self.num_classes, 3, padding=1))

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.l2norm(self.stage2(x1))
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        e1 = self.extra1(x4)
        e2 = self.extra2(e1)
        e3 = self.extra3(e2)
        feats = [x2,x3,x4,e1,e2,e3]

        locs, confs = [], []
        for f, l, c in zip(feats, self.loc_layers, self.conf_layers):
            loc = l(f).permute(0,2,3,1).reshape(x.size(0),-1,4)
            conf= c(f).permute(0,2,3,1).reshape(x.size(0),-1,self.num_classes)
            locs.append(loc); confs.append(conf)
        return torch.cat(locs,1), torch.cat(confs,1), self.priors

# Decode offsets to boxes
def decode_boxes(loc, priors, vars=(0.1,0.2)):
    cx = (priors[:,0]+priors[:,2])/2
    cy = (priors[:,1]+priors[:,3])/2
    w  = priors[:,2]-priors[:,0]
    h  = priors[:,3]-priors[:,1]

    dx,dy,dw,dh = loc.t()
    px = dx*vars[0]*w + cx
    py = dy*vars[0]*h + cy
    pw = torch.exp(dw*vars[1])*w
    ph = torch.exp(dh*vars[1])*h

    return torch.stack([px-pw/2, py-ph/2, px+pw/2, py+ph/2],1).clamp(0,1)

# Main inference
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SSD300(NUM_CLASSES).to(device).eval()
    ckpt   = torch.load(WEIGHTS_PATH, map_location=device)
    # rename old default_boxes keys if present
    new_ckpt = {
        ('priors' if k=='default_boxes_xyxy' else k):v
        for k,v in ckpt.items()
        if k!='default_boxes'
    }
    model.load_state_dict(new_ckpt)
    
    tf = A.Compose([
        A.Resize(300,300),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    # pick random VOC2007 val image
    ids = open(os.path.join(VOC_ROOT,"VOC2007/ImageSets/Main/val.txt")).read().split()
    img_id = random.choice(ids)
    img_path = os.path.join(VOC_ROOT,f"VOC2007/JPEGImages/{img_id}.jpg")
    img = np.array(Image.open(img_path).convert('RGB'))
    inp = tf(image=img)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        locs,confs,pri = model(inp)
        locs,confs,pri = locs[0], confs[0], pri.to(device)
        boxes  = decode_boxes(locs, pri)
        scores = torch.softmax(confs,1)

    # collect detections
    dets = []
    for c in range(1,NUM_CLASSES):
        m = scores[:,c]>0.5
        if not m.any(): continue
        b, s = boxes[m], scores[m,c]
        keep = nms(b, s, 0.45)
        for i in keep: dets.append((VOC_CLASSES[c], s[i].item(), b[i].cpu().numpy()))

    # plot
    os.makedirs('inference_results', exist_ok=True)
    H, W = img.shape[:2]
    fig,ax = plt.subplots(figsize=(6,6))
    ax.imshow(img); ax.axis('off')
    for label,score,box in dets:
        x1,y1,x2,y2 = (box * np.array([W,H,W,H])).astype(int)
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,
                                 edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.text(x1,y1-2,f"{label}:{score:.2f}",
                color='white',fontsize=8,backgroundcolor='r')

    out = f"inference_results/{img_id}_pred.png"
    plt.savefig(out,bbox_inches='tight',pad_inches=0)
    plt.close(fig)
    print("Saved:", out)

if __name__ == "__main__":
    main()