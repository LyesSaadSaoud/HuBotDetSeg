# HuBot: A Biomimicking Mobile Robot for Non-Disruptive Bird Behavior Study

Welcome to the official repository for **HuBot**, a biomimetic mobile robot designed for non-invasive wildlife monitoring and ecological research. This repository hosts the dataset, methodologies, results, and code associated with the HuBot project.

---

## Overview

The **Houbara bustard** is a critically endangered bird species that poses challenges for traditional research due to its elusive nature and sensitivity to human disturbances. To overcome these limitations, we introduce **HuBot**, a biomimetic robot designed to seamlessly integrate into the Houbara's natural habitat. Leveraging cutting-edge object detection algorithms and robust hardware, HuBot enables unprecedented insights into the Houbara's movement patterns, social interactions, and habitat use.

---

## Dataset

### **Key Details**
- **Size:** 5,000 images annotated for object detection.
- **Sources:** In-house footage, online repositories, and globally deployed camera traps.
- **Dataset Split:**
  - Training: 4,000 images
  - Validation: 500 images
  - Testing: 500 images
- **Annotations:** Bounding boxes localizing Houbara bustards in various environments.

### **Features**
- **Bias Mitigation:** Diverse lighting, backgrounds, and challenging scenarios.
- **Data Augmentation:** Horizontal flips, cropping, blurring, and noise addition.
- **Future Expansion:** Scaling to 50,000 images covering rare behaviors and subspecies.

The dataset supports the training and evaluation of object detection algorithms for non-invasive ecological monitoring.

---

## Experimental Setup and Methodology

### **Hardware**
- **Platform:** NVIDIA Jetson AGX Xavier
- **Specifications:**
  - NVIDIA Volta™ GPU with 10 TeraFLOPS
  - 48 GB LPDDR4 memory
- **Environment:** Ubuntu 18.04 LTS, PyTorch 2.2.1 with CUDA 11.8, OpenCV-Python

### **Software Pipeline**
- **Data Management:** Real-time wireless transmission, preprocessing, and storage on a secure local server.
- **Algorithms:**
  - Object Detection: YOLOv9 for precise localization
  - Real-Time Performance: Optimized for embedded applications
- **Testing Environment:**
  - Locations: Desert landscapes, coastal regions, and semi-urban environments.
  - Climate: Extreme temperatures (10°C to 45°C) and high humidity (up to 95%).

---

## Key Features and Contributions

1. **Biomimetic Design:** Life-like appearance and movement minimize disturbance to wildlife.
2. **Advanced Algorithms:**
   - Real-time detection with YOLOv9.
   - Enhanced environmental analysis with deep learning models.
3. **Environmental Resilience:**
   - Thermal insulation for extreme temperatures.
   - Moisture-resistant materials for high humidity and precipitation.
4. **Non-Invasive Observation:** Demonstrated ability to approach Houbara bustards without altering their behavior.

---

## Code Examples

### **Object Detection**
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image

# Load the model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess the image
image = Image.open("data/input/example.jpg").convert("RGB")
transform = T.Compose([T.ToTensor()])
input_tensor = transform(image).unsqueeze(0)

# Perform detection
with torch.no_grad():
    outputs = model(input_tensor)

# Print detections
for box, score in zip(outputs[0]['boxes'], outputs[0]['scores']):
    if score > 0.8:  # Threshold
        print(f"Box: {box.numpy()}, Score: {score.numpy()}")

---

## Experimental Setup and Methodology

### **Hardware**
- **Platform:** NVIDIA Jetson AGX Xavier
- **Specifications:**
  - NVIDIA Volta™ GPU with 10 TeraFLOPS
  - 48 GB LPDDR4 memory
- **Environment:** Ubuntu 18.04 LTS, PyTorch 2.2.1 with CUDA 11.8, OpenCV-Python

### **Software Pipeline**
- **Data Management:** Real-time wireless transmission, preprocessing, and storage on a secure local server.
- **Algorithms:**
  - Object Detection: YOLOv9 for precise localization.
  - Real-Time Performance: Optimized for embedded applications.
- **Testing Environment:**
  - Locations: Desert landscapes, coastal regions, and semi-urban environments.
  - Climate: Extreme temperatures (10°C to 45°C) and high humidity (up to 95%).

---

## Key Features and Contributions

1. **Biomimetic Design:** Life-like appearance and movement minimize disturbance to wildlife.
2. **Advanced Algorithms:**
   - Real-time detection with YOLOv9.
   - Enhanced environmental analysis with deep learning models.
3. **Environmental Resilience:**
   - Thermal insulation for extreme temperatures.
   - Moisture-resistant materials for high humidity and precipitation.
4. **Non-Invasive Observation:**
   - Demonstrated ability to approach Houbara bustards without altering their behavior.

---

## Results

### **Object Detection Performance**
- **Model:** YOLOv9
- **Metrics:**
  - mAP50: 98.56%
  - Precision: 98.33%
  - Recall: 94.55%
  - F1-Score: 96.40%
  - Inference Time: 0.0568 seconds

### **Environmental Validation**
- **Temperature:** Operated stably in temperatures ranging from 10°C to 45°C.
- **Humidity:** Maintained functionality under 95% humidity.
- **Live Trials:** Successfully interacted with Houbara bustards in controlled and natural environments.

---

## Future Work

1. Expanding the dataset to include:
   - Rare Houbara behaviors.
   - Multiple subspecies.
   - Geographically diverse locations.
2. Enhancing HuBot's hardware for:
   - Improved waterproofing.
   - Extended battery life.
3. Refining algorithms to further reduce detection and localization errors.

---

## Repository Structure

/static/ # Contains media assets such as images and videos /images/ # Images used in the dataset and visualizations /videos/ # Videos showcasing HuBot's performance index.html # Main webpage for the project README.md # This file


---

## Citation

If you use HuBot or its dataset, please cite our work:

Lyes Saad Saoud, Loïc Lesobre, Enrico Sorato, Saud Al Qaydi, Yves Hingrat, Lakmal Seneviratne, Irfan Hussain. "HuBot: A Biomimicking Mobile Robot for Non-Disruptive Bird Behavior Study and Ecological Conservation."


---

## Contact

For questions or collaboration inquiries, please contact:

- **Lyes Saad Saoud** (Primary Contributor): `lyes.saoud@kustar.ac.ae`
- **Irfan Hussain** (Corresponding Author): `irfan.hussain@ku.ac.ae`

---

### License

This repository is shared under the **CC BY 4.0 License**, allowing reuse for non-commercial purposes with proper attribution.
