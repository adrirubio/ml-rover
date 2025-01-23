# CNN
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
import numpy as np
import matplotlib as plt
from datetime import datetime

feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
model = AutoModelForObjectDetection.from_pretrained('facebook/detr-resnet-50')