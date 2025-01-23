# CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from datasets import load_dataset
import numpy as np
import matplotlib as plt
from datetime import datetime

# Load COCO dataset
train_dataset = load_dataset('coco_detection', 'train')
test_dataset = load_dataset('coco_detection', 'train')

# Load the feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/detr-resnet-50')

