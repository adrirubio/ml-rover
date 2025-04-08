import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Pascal VOC class names
VOC_CLASSES = (
    'background',  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

def load_model(model_path):
    """Load the trained SSD model"""
    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Import SSD class from your training script
    # (Make sure the SSD class definition is available)
    from ssd_object_detection import SSD
    
    # Create model and load weights
    model = SSD(num_classes=len(VOC_CLASSES))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

def detect_objects(model, image_path, device, conf_threshold=0.5):
    """Detect objects in an image using the trained SSD model"""
    # Load and prepare the image
    image = Image.open(image_path).convert('RGB')
    orig_image = np.array(image)
    
    # Transform the image
    transform = A.Compose([
        A.Resize(height=300, width=300),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=np.array(image))
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Helper function to decode boxes
    def decode_boxes(loc, default_boxes):
        # Ensure same device
        default_boxes = default_boxes.to(device)
        
        # Convert default boxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h)
        def corner_to_center(boxes):
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            cx = boxes[:, 0] + width / 2
            cy = boxes[:, 1] + height / 2
            return torch.stack([cx, cy, width, height], dim=1)
        
        # Convert boxes to center format
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
    
    # Perform detection
    with torch.no_grad():
        # Forward pass
        loc_preds, conf_preds, default_boxes = model(image_tensor)
        
        # Decode box coordinates
        decoded_boxes = decode_boxes(loc_preds[0], default_boxes)
        
        # Get confidence scores
        scores = torch.nn.functional.softmax(conf_preds[0], dim=1)
        
        # Collect detections
        detections = []
        
        for class_idx in range(1, len(VOC_CLASSES)):  # Skip background class
            class_scores = scores[:, class_idx]
            mask = class_scores > conf_threshold
            
            if mask.sum() == 0:
                continue
                
            # Get boxes and scores for this class
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
    
    return orig_image, detections

def visualize_detections(image, detections):
    """Visualize detected objects on the image"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Different colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, len(VOC_CLASSES)))
    
    # Draw each detection
    for det in detections:
        box = det['box']
        score = det['score']
        class_idx = det['class']
        color = colors[class_idx]
        
        # Scale box coordinates to original image size
        h, w, _ = image.shape
        x1, y1, x2, y2 = box
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        
        # Add label
        class_name = VOC_CLASSES[class_idx]
        plt.text(x1, y1-5, f'{class_name}: {score:.2f}', 
                 fontsize=10, color='white', 
                 bbox=dict(facecolor=color, alpha=0.7))
    
    plt.title('Object Detections')
    plt.axis('off')
    plt.savefig('detection_result.png')
    plt.show()
    
    # Also print detected objects
    class_counts = {}
    for det in detections:
        class_name = VOC_CLASSES[det['class']]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    print("Detected objects:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count}")

# Main function
def main():
    # Load the model
    model_path = 'ssd_pascal_voc_final.pth'
    model, device = load_model(model_path)
    
    # Path to the image you want to test
    image_path = 'test_image.jpg'  
    
    # Detect objects
    image, detections = detect_objects(model, image_path, device, conf_threshold=0.4)
    
    # Visualize and print results
    visualize_detections(image, detections)

if __name__ == "__main__":
    main()
