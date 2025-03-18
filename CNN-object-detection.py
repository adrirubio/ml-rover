# CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from datasets import load_dataset
import numpy as np
import matplotlib as plt
from datetime import datetime
from torchvision import transforms

# Load PASCAL VOC 2007 - both train and validation sets
voc_dataset = load_dataset("detection-datasets/pascal-voc", "detection-datasets--pascal-voc")

# Access train and validation splits
train_dataset = voc_dataset["train"]
val_dataset = voc_dataset["validation"]