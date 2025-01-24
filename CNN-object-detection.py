# CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from datasets import load_dataset
import numpy as np
import matplotlib as plt
from datetime import datetime
from torchvision import transforms

# Load COCO dataset
train_dataset = load_dataset("detection-datasets/coco", split="train")
test_dataset = load_dataset("detection-datasets/coco", split="validation")

# Load the feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/detr-resnet-50')

# Print an example
print(f"First image: {train_dataset[0]}")