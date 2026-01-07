# Distinguishing JJ-Lin from Look-Alikes

A machine learning project that implements Convolutional Neural Networks (CNN) and Fully Connected Neural Networks (NN) using PyTorch to distinguish JJ Lin (林俊杰, Lim Jun Jie) from non-JJ Lin individuals in unconstrained images. This study investigates the feasibility of building an automated binary classifier capable of identifying a specific public figure from individuals who share similar facial characteristics.

## 📋 Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [File Descriptions](#file-descriptions)
- [Notes](#notes)
- [Contributing](#contributing)

## ✨ Features

- **PyTorch Implementation**: Complete CNN and NN implementations using PyTorch framework
- **Binary Classification**: Distinguishes between JJ Lin and non-JJ Lin individuals (including look-alikes)
- **RGB Color Preservation**: Maintains texture, shadow, and color information crucial for facial recognition
- **Data Augmentation**: Comprehensive augmentation techniques including horizontal flipping, rotation, brightness, and contrast adjustments
- **High Performance**: CNN achieves 97.27% validation accuracy, NN baseline achieves 95.70%
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1 score, and confusion matrices

## 📊 Dataset

The dataset consists of approximately **7,326 images** collected from diverse sources including:
- Music videos and concert recordings
- Online image repositories
- Publicly available media resources

### Dataset Composition

| Category | Count | Percentage |
|----------|-------|------------|
| Real JJ Lin | 4,720 | 64.4% |
| Fake / Non-JJ Lin | 2,606 | 35.6% |
| **Total** | **7,326** | **100%** |

The dataset is split into:
- **Training set**: 5,860 samples (80%)
- **Validation set**: 1,466 samples (20%)

Stratified sampling is used to maintain class balance across splits.

## 📁 Project Structure

```
ML_Distinguishing_JJLin/
├── Activation.py          # Activation functions (sigmoid, relu, softmax)
├── Dense.py              # Fully connected layer implementation
├── Loss.py               # Loss functions (BCE, CCE, MSE)
├── Predict.py            # Prediction utility functions
├── preprocess_image.py   # Image preprocessing functions
├── preprocess_image_test.py  # Test preprocessing functions
├── testme_CNN.py         # CNN model testing script
├── testme_NN.py          # Neural Network testing script
├── CNN.ipynb             # CNN model training notebook
├── NN.ipynb              # Neural Network training notebook
├── SVM.ipynb             # SVM model notebook
├── project.ipynb         # Main project notebook
├── photo/                # Test images directory
│   ├── test1.png
│   ├── test2.png
│   └── ...
├── README.md             # This file
└── REBUTTAL.md           # Rebuttal to reviewer comments
```

## 🔧 Requirements

- Python 3.7+
- PyTorch
- NumPy
- PIL (Pillow)
- Matplotlib
- Pandas
- scikit-learn
- torchvision

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/cccalan1108/ML_Distinguishing_JJLin_from_Look_Alikes.git
cd ML_Distinguishing_JJLin
```

2. Install required packages:
```bash
pip install torch torchvision numpy pillow matplotlib pandas scikit-learn
```

3. Download the test dataset:
   - Download the testing set from [Google Drive](https://drive.google.com/drive/folders/1HG7B5WC1nAFCIQp1F4_H1STJFt9Jwpi5?usp=sharing)
   - Place the images in the `photo/` directory (or create the directory if it doesn't exist)

## 🚀 Usage

### Testing Pre-trained Models

You can use the programs we provided `testme_NN.py` and `testme_CNN.py` to test our models.

#### Test CNN Model:
```bash
python testme_CNN.py
```

#### Test Neural Network Model:
```bash
python testme_NN.py
```

### Training Models

#### Using Jupyter Notebooks:

1. **CNN Model Training**:
   - Open `CNN.ipynb`
   - Follow the cells to train the CNN model
   - The model will train for 20 epochs with the configured hyperparameters

2. **Neural Network Training**:
   - Open `NN.ipynb`
   - Follow the cells to train the fully connected NN
   - The model will train for 15 epochs with PCA dimensionality reduction

### Image Preprocessing

The preprocessing pipeline includes:

1. **RGB Color Preservation**: All images maintained in RGB color space to preserve texture details and shadow information
2. **Face Cropping**: Images cropped to include only faces positioned centrally
3. **Resizing**: Images resized to 128×128 pixels
4. **Normalization**: Mean-std normalization (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) to scale pixel values to [-1, 1]
5. **Data Augmentation** (training only):
   - Horizontal flipping (probability=0.5)
   - Random rotation (±15 degrees)
   - Brightness adjustment (factor range [0.8, 1.2])
   - Contrast adjustment (factor range [0.8, 1.2])

For the Neural Network model, RGB images are flattened to 49,152-dimensional vectors (128×128×3), then reduced to 256 dimensions using PCA with whitening.

```python
from preprocess_image import preprocess_image

data, labels = preprocess_image()
```

## 🔬 Methodology

### Training Configuration

Both models were trained with the following settings:

- **Optimizer**: Adam
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **Epochs**: 
  - Neural Network: 15 epochs
  - CNN: 20 epochs
- **Regularization**: 
  - Dropout rate: 0.5 (applied in fully connected layers)
  - Batch Normalization (CNN only, after each convolutional layer)

## 🏗️ Model Architecture

### CNN Architecture (Deep CNN)

The CNN model leverages hierarchical spatial feature extraction through multiple convolutional blocks:

| Layer Type | Filter Size | Channels | Stride | Padding | Activation |
|------------|-------------|----------|--------|---------|------------|
| Conv2D + BatchNorm | 3×3 | 3→32 | 1 | 1 | ReLU |
| MaxPool2D | 2×2 | – | 2 | – | – |
| Conv2D + BatchNorm | 3×3 | 32→64 | 1 | 1 | ReLU |
| MaxPool2D | 2×2 | – | 2 | – | – |
| Conv2D + BatchNorm | 3×3 | 64→128 | 1 | 1 | ReLU |
| MaxPool2D | 2×2 | – | 2 | – | – |
| Conv2D + BatchNorm | 3×3 | 128→256 | 1 | 1 | ReLU |
| MaxPool2D | 2×2 | – | 2 | – | – |
| Flatten | – | – | – | – | – |
| Dense + Dropout(0.5) | – | 16,384→512 | – | – | ReLU |
| Dense | – | 512→1 | – | – | Sigmoid |

**Key Features:**
- **Input**: 128×128×3 RGB images
- **4 Convolutional Blocks**: Progressive feature extraction from 32 to 256 filters
- **Batch Normalization**: Applied after each convolutional layer for stable training
- **Max Pooling**: 2×2 pooling with stride 2 after each convolutional block
- **Fully Connected Layers**: 16,384 → 512 → 1 with dropout regularization

### Neural Network Architecture (Baseline)

| Layer | Units | Activation |
|-------|-------|------------|
| Input (after PCA) | 256 | – |
| Hidden Layer 1 | 1,024 | ReLU |
| Dropout | 0.5 | – |
| Hidden Layer 2 | 512 | ReLU |
| Dropout | 0.5 | – |
| Output | 1 | Sigmoid |

**Key Features:**
- **Input**: 256-dimensional PCA-reduced feature vectors (from 49,152 flattened RGB pixels)
- **PCA Dimensionality Reduction**: Top 256 principal components with whitening
- **Fully Connected Architecture**: Two hidden layers with dropout regularization
- **Activation**: ReLU for hidden layers, Sigmoid for binary classification output

## 📊 Results

### Model Performance

Both models achieved strong performance on the validation set:

| Model | Validation Accuracy | F1 Score | Epochs |
|-------|---------------------|----------|--------|
| **CNN** | **97.27%** | **0.9786** | 20 |
| **NN Baseline (PCA)** | **95.70%** | **0.9673** | 15 |

### Key Findings

1. **High Performance**: Both models achieved accuracy above 95%, demonstrating strong classification capability
2. **CNN Advantage**: The CNN's hierarchical spatial feature extraction provides a modest 1.5% improvement over the PCA-based NN baseline
3. **RGB Preservation**: Maintaining RGB color information (rather than grayscale/binarization) preserves crucial texture and shadow details essential for facial recognition
4. **Effective Preprocessing**: The combination of RGB preservation, data augmentation, and appropriate normalization contributed to high model performance
5. **Competitive NN Baseline**: Despite receiving only 256-dimensional PCA-reduced features, the NN baseline achieved competitive performance, indicating successful feature extraction

### Confusion Matrices

The confusion matrices reveal detailed classification patterns for both models:
- CNN demonstrates slightly better discrimination between JJ Lin and non-JJ Lin samples
- Both models show strong precision and recall across both classes
- F1 scores above 0.96 indicate balanced precision and recall performance

## 📄 File Descriptions

- **Activation.py**: Implements activation functions (sigmoid, ReLU, softmax) with forward and backward passes
- **Dense.py**: Fully connected layer implementation with forward propagation, backpropagation, and parameter updates
- **Loss.py**: Loss function implementations (Binary Cross-Entropy, Categorical Cross-Entropy, MSE)
- **Predict.py**: Utility functions for model prediction and accuracy calculation
- **preprocess_image.py**: Image loading, preprocessing, and augmentation functions
- **testme_CNN.py**: Complete CNN model implementation and testing script
- **testme_NN.py**: Complete NN model implementation and testing script
- **CNN.ipynb**: CNN model training and evaluation notebook
- **NN.ipynb**: Neural Network training and evaluation notebook
- **SVM.ipynb**: Support Vector Machine implementation notebook

## ⚠️ Notes

### Important Considerations

1. **Model Files**: Pre-trained model files (`.pth` or `.pkl`) are excluded from the repository due to size limitations. You may need to train the models yourself or use Git LFS for large files.

2. **Image Preprocessing**: 
   - Images are maintained in RGB color space to preserve texture and shadow information
   - All images are resized to 128×128 pixels
   - Mean-std normalization is applied to scale pixel values to [-1, 1]
   - Data augmentation is applied only during training

3. **Computational Requirements**: 
   - Training on the full dataset requires significant computational resources
   - Recommended: GPU acceleration (CUDA-compatible) for faster training
   - Training was performed on Google Colab and VSCode environments
   - For the NN model, PCA dimensionality reduction is necessary to make training tractable

4. **Dataset**: 
   - Training dataset paths are hardcoded in `preprocess_image.py`
   - Update the paths according to your local setup:
     ```python
     preprocess_folder_real = "path/to/JJLin/images/"
     preprocess_folder_fake = "path/to/non-JJLin/images/"
     ```

5. **Test Images**: 
   - The `photo/` directory should contain test images
   - Images should be in PNG or JPG format
   - Images will be automatically preprocessed (resized to 128×128, normalized) if needed

## 🤝 Contributing

This is a research project. Contributions, suggestions, and improvements are welcome! Please feel free to:
- Report bugs or issues
- Suggest enhancements
- Submit pull requests
- Share your results and findings

## 📝 License

This project is for educational and research purposes. Please ensure you have proper permissions for any images used.

## 📚 References

- Lawrence, S., Giles, C. L., Tsoi, A. C., & Back, A. D. (1997). Face recognition: A convolutional neural-network approach. IEEE Transactions on Neural Networks, 8(1), 98–113.
- Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). DeepFace: Closing the gap to human-level performance in face verification. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. IEEE Conference on Computer Vision and Pattern Recognition.
- PyTorch Documentation: https://pytorch.org/docs/

## 👥 Authors

- **Ting-Chen Cho** (B11801004@ntu.edu.tw)
- **Chun-Chieh Chang** (B10801011@ntu.edu.tw)

---

**Note**: This project investigates the feasibility of binary facial recognition for distinguishing JJ Lin from look-alikes. The results demonstrate that both CNN and NN approaches achieve strong performance (>95% accuracy) when combined with appropriate preprocessing techniques including RGB color preservation, data augmentation, and comprehensive normalization. The CNN's hierarchical spatial feature extraction provides modest advantages over the PCA-based NN baseline, with both models demonstrating viability for celebrity face recognition tasks.
