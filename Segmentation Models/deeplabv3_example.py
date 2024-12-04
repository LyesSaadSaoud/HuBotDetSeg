import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# Load model
model = deeplabv3_resnet50(pretrained=True)
model.eval()

# Load and preprocess image
image_path = "../test_images/example.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = ToTensor()(image).unsqueeze(0)

# Perform segmentation
with torch.no_grad():
    output = model(input_tensor)["out"][0]
    mask = output.argmax(0).byte().cpu().numpy()

# Save segmentation mask
Image.fromarray(mask * 255).save("../test_images/deeplabv3_mask.png")
