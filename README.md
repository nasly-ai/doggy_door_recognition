


What This Project Does
This project builds a personalized doggy door that uses computer vision to recognize a specific dog — Bo, the White House dog — and only opens for him. Every other animal stays outside.
The challenge: train an accurate image classifier using only 30 photographs. Training a deep learning model from scratch on 30 images would fail immediately due to overfitting. The solution is transfer learning — leveraging a model already trained on millions of images to recognize the specific subject using a fraction of the data.

The Problem It Solves
Standard image classifiers can recognize broad categories — dogs, cats, cars. They cannot distinguish one specific dog from another without being retrained. This project demonstrates how to take a general-purpose vision model and specialize it for a specific recognition task using a small, real-world dataset.

How It Works
Base Model: VGG16 pre-trained on ImageNet — a model that has already learned to recognize thousands of object categories including animals, features, and shapes.
Transfer Learning Strategy:

Load the pre-trained VGG16 model with its learned weights intact
Freeze all base layers — preserving 1.2 million learned image features
Add a custom classification layer on top — one output neuron for binary classification (Bo or not Bo)
Train only the new layers on the 30-image dataset
Fine-tune by selectively unfreezing base layers for improved accuracy

Classification Type: Binary — the model outputs a single probability score determining whether the subject in the image is Bo.

Technical Stack
ComponentTechnologyDeep Learning FrameworkPyTorchBase ModelVGG16 (torchvision.models)Pre-training DatasetImageNetLoss FunctionBinary Cross-Entropy (BCELoss)Image Processingtorchvision.transformsHardware AccelerationCUDA (GPU) / CPU fallbackEnvironmentGoogle Colab / Jupyter Notebook

Key Concepts Demonstrated
Transfer Learning — Reusing learned representations from a large model to solve a related problem with limited data. The VGG16 model learned to detect edges, textures, shapes, and animal features from ImageNet. Those learned features transfer directly to recognizing a specific dog.
Model Freezing — Preventing pre-trained weights from being updated during initial training. This preserves the valuable representations learned from ImageNet while allowing the new classification layer to learn from the small dataset.
Fine-Tuning — Selectively unfreezing base model layers after initial training to allow deeper optimization on the target dataset.
Binary Classification — Reducing a complex multi-class model to a single yes/no decision using a custom output layer and binary loss function.
Small Dataset Optimization — Achieving accurate classification results with only 30 training images by leveraging pre-trained feature representations rather than learning from scratch.

Image Preprocessing Pipeline
pythonpreprocess = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
Images are resized, cropped, and normalized to match the input format VGG16 was originally trained on — ensuring the pre-trained weights are applied correctly.

Model Architecture
Sequential(
  VGG16 (frozen) — 1000 output features
  Linear(1000 → 1) — binary classification output
)
The frozen VGG16 acts as a feature extractor. The custom Linear layer learns to map those 1000 features to a single Bo/not-Bo prediction.

Results
Transfer learning allows accurate personalized recognition from a 30-image dataset — a task that would be impossible with a model trained from scratch on the same data. The model generalizes to new images of Bo it has never seen, distinguishing him from other dogs and animals.

Curriculum Context
This project was completed as part of the NVIDIA Deep Learning Institute (DLI) curriculum — a hands-on certification program covering computer vision, deep learning, and transfer learning using PyTorch and GPU-accelerated computing.

Real-World Applications
The techniques demonstrated in this project apply directly to:

Security and access control — recognizing authorized individuals or animals
Healthcare imaging — personalizing diagnostic models for specific patient profiles
Retail and inventory — training product recognition models on small SKU datasets
Wildlife monitoring — identifying specific animals across camera trap networks
Identity verification — building specialized biometric recognition systems


Author
Nasly Duarte
Applied AI Student | Financial Operations Architect | Builder
Miami Dade College — BS in Applied Artificial Intelligence
github.com/nduarte215 | mindfuldollar.blogspot.com
