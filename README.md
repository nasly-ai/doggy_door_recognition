# Doggy Door Recognition

![Bo — Presidential Doggy Door](images/bo_10.jpg)

### Personalized Computer Vision Using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange) ![NVIDIA](https://img.shields.io/badge/NVIDIA-DLI%20Certified-green) ![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-VGG16-purple)

---

## What This Project Does

This project builds a personalized doggy door that uses computer vision to recognize a specific dog — Bo, the White House dog — and only opens for him. Every other animal stays outside.

The challenge: train an accurate image classifier using only 30 photographs. Training a deep learning model from scratch on 30 images would fail immediately due to overfitting. The solution is transfer learning — leveraging a model already trained on millions of images to recognize the specific subject using a fraction of the data.

---

## The Problem It Solves

Standard image classifiers can recognize broad categories — dogs, cats, cars. They cannot distinguish one specific dog from another without being retrained. This project demonstrates how to take a general-purpose vision model and specialize it for a specific recognition task using a small, real-world dataset.

---

## How It Works

**Base Model:** VGG16 pre-trained on ImageNet — a model that has already learned to recognize thousands of object categories including animals, features, and shapes.

**Transfer Learning Strategy:**
1. Load the pre-trained VGG16 model with its learned weights intact
2. Freeze all base layers — preserving 1.2 million learned image features
3. Add a custom classification layer on top — one output neuron for binary classification (Bo or not Bo)
4. Train only the new layers on the 30-image dataset
5. Fine-tune by selectively unfreezing base layers for improved accuracy

**Classification Type:** Binary — the model outputs a single probability score determining whether the subject in the image is Bo.

---

## Technical Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Base Model | VGG16 (torchvision.models) |
| Pre-training Dataset | ImageNet |
| Loss Function | Binary Cross-Entropy (BCELoss) |
| Image Processing | torchvision.transforms |
| Hardware Acceleration | CUDA (GPU) / CPU fallback |
| Environment | Google Colab / Jupyter Notebook |

---

## Key Concepts Demonstrated

**Transfer Learning** — Reusing learned representations from a large model to solve a related problem with limited data.

**Model Freezing** — Preventing pre-trained weights from being updated during initial training to preserve ImageNet learning.

**Fine-Tuning** — Selectively unfreezing base model layers after initial training for deeper optimization.

**Binary Classification** — Reducing a complex multi-class model to a single yes/no decision using a custom output layer.

**Small Dataset Optimization** — Achieving accurate classification with only 30 training images by leveraging pre-trained feature representations.

---

## Image Preprocessing Pipeline

```python
preprocess = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## Model Architecture

```
Sequential(
  VGG16 (frozen) — 1000 output features
  Linear(1000 → 1) — binary classification output
)
```

---

## Real-World Applications

- Security and access control — recognizing authorized individuals
- Healthcare imaging — personalizing diagnostic models for specific patients
- Retail and inventory — training product recognition on small datasets
- Wildlife monitoring — identifying specific animals across camera networks
- Identity verification — building specialized biometric recognition systems

---

## Curriculum Context

Completed as part of the **NVIDIA Deep Learning Institute (DLI)** curriculum — a hands-on certification program covering computer vision, deep learning, and transfer learning using PyTorch and GPU-accelerated computing.

---

## Author

**Nasly Duarte**
Applied AI Student | Financial Operations Architect | Builder
Miami Dade College — BS in Applied Artificial Intelligence
[github.com/nduarte215](https://github.com/nduarte215) | [mindfuldollar.blogspot.com](https://mindfuldollar.blogspot.com)

---

## Repository Structure

```
doggy-door-recognition/
│
├── README.md
├── 05b_presidential_doggy_door__1_.ipynb
│
└── images/
    ├── bo_10.jpg
    ├── bo_14.jpg
    └── bo_15.jpg
```
