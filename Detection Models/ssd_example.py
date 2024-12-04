import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import ToTensor
from PIL import Image
from utils.visualize import draw_boxes

# Load model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

# Load and preprocess image
image_path = "../test_images/example.jpg"
image = Image.open(image_path).convert("RGB")
transform = ToTensor()
input_tensor = transform(image).unsqueeze(0)

# Perform detection
with torch.no_grad():
    outputs = model(input_tensor)

# Draw bounding boxes
boxes = outputs[0]['boxes']
scores = outputs[0]['scores']
result_image = draw_boxes(image, boxes, scores, threshold=0.5)

# Save result
result_image.save("../test_images/ssd_result.jpg")
