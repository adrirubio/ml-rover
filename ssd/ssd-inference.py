"""
SSD300 Inference

Usage:
    python ssd-inference.py \
      --image /path/to/image.jpg \
      --weights /path/to/ssd_weights.pth \
      --output /path/to/save/dir
"""

import os, argparse
import torch, torchvision, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Config
VOC_CLASSES = (
    'background','aeroplane','bicycle','bird','boat','bottle','bus',
    'car','cat','chair','cow','diningtable','dog','horse','motorbike',
    'person','pottedplant','sheep','sofa','train','tvmonitor'
)

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=20.0, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_channels) * scale)
    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt().clamp(min=self.eps)
        return x / norm * self.weight.view(1, -1, 1, 1)

# SSD300 Model
class SSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.stage1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1
        )  # 75×75
        self.stage2 = resnet.layer2  # 38×38
        self.stage3 = resnet.layer3  # 19×19
        self.stage4 = resnet.layer4  # 10×10
        self.l2norm = L2Norm(512)
        # extra layers
        self.extra1 = nn.Sequential(
            nn.Conv2d(2048,512,1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,stride=2,padding=1), nn.ReLU(inplace=True)
        )  # →5×5
        self.extra2 = nn.Sequential(
            nn.Conv2d(512,256,1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,stride=2,padding=1), nn.ReLU(inplace=True)
        )  # →3×3
        self.extra3 = nn.Sequential(
            nn.Conv2d(256,256,3,stride=2,padding=1), nn.ReLU(inplace=True)
        )  # →1×1
        self.loc_layers, self.conf_layers = nn.ModuleList(), nn.ModuleList()
        self._build_heads_and_anchors()

    def _build_heads_and_anchors(self):
        # dummy forward to get shapes
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
        # build priors
        scales = [0.2,0.34,0.48,0.62,0.76,0.9]
        ars    = [[2,0.5]]*6
        boxes = []
        for k, f in enumerate(feats):
            sz = f.shape[2]
            s_k, s_k1 = scales[k], scales[k+1] if k+1<len(scales) else 1.0
            for i in range(sz):
                for j in range(sz):
                    cx,cy = (j+0.5)/sz, (i+0.5)/sz
                    boxes += [[cx,cy,s_k,s_k],
                              [cx,cy,(s_k*s_k1)**.5,(s_k*s_k1)**.5]]
                    for ar in ars[k]:
                        boxes += [[cx,cy,s_k*ar**.5,s_k/ar**.5],
                                  [cx,cy,s_k/ar**.5,s_k*ar**.5]]
        db = torch.tensor(boxes).clamp(0,1)
        cx,cy,w,h = db.unbind(1)
        db_xyxy = torch.stack([cx-w/2,cy-h/2, cx+w/2,cy+h/2],1).clamp(0,1)
        self.register_buffer('default_boxes', db_xyxy)
        # build heads
        num_anchors = [2+2*len(a) for a in ars]
        channels    = [f.shape[1] for f in feats]
        for ch,a in zip(channels, num_anchors):
            self.loc_layers.append (nn.Conv2d(ch,a*4, padding=1, kernel_size=3))
            self.conf_layers.append(nn.Conv2d(ch,a*self.num_classes, padding=1, kernel_size=3))

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
        return torch.cat(locs,1), torch.cat(confs,1), self.default_boxes

# Box decoding
def decode_boxes(loc_preds, priors, variances=(0.1,0.2)):
    d_cx = (priors[:,0]+priors[:,2])/2
    d_cy = (priors[:,1]+priors[:,3])/2
    d_w  = priors[:,2]-priors[:,0]
    d_h  = priors[:,3]-priors[:,1]
    p_cx = loc_preds[:,0]*variances[0]*d_w + d_cx
    p_cy = loc_preds[:,1]*variances[0]*d_h + d_cy
    p_w  = torch.exp(loc_preds[:,2]*variances[1])*d_w
    p_h  = torch.exp(loc_preds[:,3]*variances[1])*d_h
    xmin = p_cx-p_w/2; ymin = p_cy-p_h/2
    xmax = p_cx+p_w/2; ymax = p_cy+p_h/2
    return torch.stack([xmin,ymin,xmax,ymax],1).clamp(0,1)

# Inference entry
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image',   required=True)
    p.add_argument('--weights', default='ssd_weights.pth')
    p.add_argument('--output',  default='inference_results')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SSD(len(VOC_CLASSES)).to(device).eval()
    state  = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=False)

    tf = A.Compose([A.Resize(300,300),
                    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
                    ToTensorV2()])
    img = np.array(Image.open(args.image).convert('RGB'))
    inp = tf(image=img)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        locs, confs, priors = model(inp)
        locs, confs = locs[0], confs[0]
        boxes  = decode_boxes(locs, priors)
        scores = torch.softmax(confs, dim=1)

    # per-class NMS @0.3, conf>0.5
    final_b, final_s, final_l = [], [], []
    for c in range(1, len(VOC_CLASSES)):
        m = scores[:,c]>0.5
        if not m.any(): continue
        b = boxes[m].cpu(); s = scores[m,c].cpu()
        keep = torchvision.ops.nms(b, s, 0.3)
        final_b.append(b[keep]); final_s.append(s[keep])
        final_l.append(torch.full((len(keep),), c, dtype=torch.int64))
    if not final_b:
        print("No detections.")
        return
    final_b = torch.cat(final_b,0)
    final_s = torch.cat(final_s,0)
    final_l = torch.cat(final_l,0)

    # top‑20
    topk = final_s.argsort(descending=True)[:20]
    final_b, final_s, final_l = final_b[topk], final_s[topk], final_l[topk]

    # draw
    os.makedirs(args.output, exist_ok=True)
    H,W = img.shape[:2]
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.imshow(img); ax.axis('off')
    for box, score, lbl in zip(final_b, final_s, final_l):
        x1,y1,x2,y2 = (box * torch.tensor([W,H,W,H])).int().tolist()
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-2, f'{VOC_CLASSES[lbl]}:{score:.2f}',
                color='white', fontsize=8, backgroundcolor='r')
    out = os.path.join(args.output,
                       os.path.basename(args.image).rsplit('.',1)[0]+'_pred.png')
    plt.savefig(out, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print("Saved:", out)

if __name__ == '__main__':
    main()