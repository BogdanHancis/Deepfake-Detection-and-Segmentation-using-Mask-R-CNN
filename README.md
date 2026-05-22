# Deepfake Detection and Segmentation using Mask R-CNN

This project implements a deep learning-based system for detecting and segmenting manipulated facial regions in images using **Mask R-CNN** with PyTorch. The model performs pixel-level segmentation to identify forged areas in facial images.

---

## 🚀 Project Overview

Deepfake manipulation is a growing challenge in digital media authenticity. This project focuses on:

- Detecting manipulated facial regions
- Performing instance segmentation of forged areas
- Evaluating model performance using IoU and segmentation metrics
- Building a full ML pipeline (training, validation, evaluation, visualization)

---

## 📚 Dataset

This project uses the **OpenForensics dataset**, a large-scale benchmark for multi-face forgery detection and segmentation.

- 📌 Dataset: https://zenodo.org/records/5528418  
- 📄 DOI: 10.5281/zenodo.5528418  
- 🧠 Paper: OpenForensics: Multi-Face Forgery Detection and Segmentation In-The-Wild (ICCV 2021)

The dataset contains real and manipulated facial images with pixel-level annotations for segmentation tasks.

---

## 🧠 Key Features

- Mask R-CNN-based instance segmentation model
- COCO-style dataset integration
- Image preprocessing and augmentation pipeline
- Training and validation loops in PyTorch
- IoU-based evaluation metrics
- Visualization of predictions (bounding boxes + masks)
- Performance analysis and error inspection

---

## 🛠️ Technologies Used

- Python
- PyTorch
- OpenCV
- Mask R-CNN
- COCO Dataset Format
- Matplotlib / Visualization tools

---

---

## ⚙️ Pipeline

1. Load COCO-style dataset
2. Apply preprocessing and augmentation
3. Train Mask R-CNN model
4. Validate on held-out dataset
5. Evaluate using IoU and segmentation metrics
6. Visualize predictions and analyze errors

---

## 📊 Evaluation Metrics

- Intersection over Union (IoU)
- Precision / Recall
- Mask Accuracy
- Loss curves (classification, bounding box, mask loss)

---

## 📸 Results

- Successfully detects manipulated facial regions
- Produces accurate pixel-level segmentation masks
- Performs robustly across different image conditions
- Visual outputs highlight forged regions effectively

---

## 📌 Future Improvements

- Integration with transformer-based vision backbones
- Real-time deepfake detection system
- Video-based temporal forgery detection
- Improved generalization across datasets

## 👨‍💻 Author

Bogdan Hancis
