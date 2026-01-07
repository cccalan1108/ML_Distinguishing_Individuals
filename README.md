# ML Distinguishing JJ Lin from Look-Alikes

A machine learning project that implements Convolutional Neural Networks (CNN) and Fully Connected Neural Networks (NN) from scratch using only NumPy to distinguish JJ-Lin (林俊傑 Lin Jun Jie) from look-alike individuals. This educational project demonstrates fundamental deep learning concepts through complete implementation without relying on high-level frameworks.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [File Descriptions](#file-descriptions)
- [Notes](#notes)
- [Contributing](#contributing)

## ✨ Features

- **From-scratch Implementation**: Complete CNN and NN implementations using only NumPy
- **Binary Classification**: Distinguishes between Jay Chou and look-alike individuals
- **Multiple Models**: Comparison between CNN and Fully Connected NN architectures
- **Image Preprocessing**: Binarization and resizing pipeline for image data
- **Educational Focus**: Demonstrates understanding of:
  - Convolution operations
  - Backpropagation through convolutional and pooling layers
  - Gradient computation for multi-dimensional tensors
  - Batch processing and optimization

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
- NumPy
- PIL (Pillow)
- Matplotlib
- Pandas
- scikit-learn

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/cccalan1108/ML_Distinguishing_JJLin_from_Look_Alikes.git
cd ML_Distinguishing_JJLin
```

2. Install required packages:
```bash
pip install numpy pillow matplotlib pandas scikit-learn
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
   - The model architecture includes:
     - Convolutional layer
     - Max pooling layer
     - Flatten layer
     - Fully connected layers

2. **Neural Network Training**:
   - Open `NN.ipynb`
   - Follow the cells to train the fully connected NN
   - The model uses multiple dense layers with ReLU and sigmoid activations

### Image Preprocessing

The preprocessing pipeline includes:
- Image resizing to 180×180 pixels
- Grayscale conversion
- Binarization (threshold=0.4) to reduce computational complexity

```python
from preprocess_image import preprocess_image

data, labels = preprocess_image()
```

## 🏗️ Model Architecture

### CNN Architecture
- **Input**: 180×180×1 grayscale images
- **Conv Layer**: Filter size 4×4, 8 output channels, padding 2, stride 3
- **Max Pooling**: Pool size 2×2, stride 2
- **Flatten**: Reshape to 1D vector
- **Dense Layer 1**: 7200 → 1024 units (ReLU activation)
- **Dense Layer 2**: 1024 → 1 unit (Sigmoid activation)
- **Output**: Binary classification (0 or 1)

### Neural Network Architecture
- **Input**: Flattened 180×180 images (32,400 features)
- **Hidden Layer 1**: 32,400 → 128 units (ReLU)
- **Hidden Layer 2**: 128 → 64 units (ReLU)
- **Hidden Layer 3**: 64 → 32 units (ReLU)
- **Output Layer**: 32 → 1 unit (Sigmoid)
- **Output**: Binary classification (0 or 1)

## 📊 Results

### Model Performance

- **CNN Validation Accuracy**: 66.03%
- **Neural Network Test Accuracy**: 58.04%

### Key Findings

1. CNN outperforms fully connected NN for image classification tasks
2. Even shallow CNN architectures provide benefits over dense layers for spatial data
3. Binarization preprocessing reduces computational complexity but limits feature richness
4. Models demonstrate learning above random chance (50%)

## 📄 File Descriptions

- **Activation.py**: Implements activation functions (sigmoid, ReLU, softmax) with forward and backward passes
- **Dense.py**: Fully connected layer implementation with forward propagation, backpropagation, and parameter updates
- **Loss.py**: Loss function implementations (Binary Cross-Entropy, Categorical Cross-Entropy, MSE)
- **Predict.py**: Utility functions for model prediction and accuracy calculation
- **preprocess_image.py**: Image loading, resizing, and binarization functions
- **testme_CNN.py**: Complete CNN model implementation and testing script
- **testme_NN.py**: Complete NN model implementation and testing script
- **CNN.ipynb**: CNN model training and evaluation notebook
- **NN.ipynb**: Neural Network training and evaluation notebook
- **SVM.ipynb**: Support Vector Machine implementation notebook

## ⚠️ Notes

### Important Considerations

1. **Model Files**: Pre-trained model files (`.npy`) are excluded from the repository due to size limitations. You may need to train the models yourself or use Git LFS for large files.

2. **Image Preprocessing**: 
   - Images are binarized (threshold=0.4) to reduce computational complexity
   - This preprocessing choice trades feature richness for computational feasibility
   - For better performance, consider using grayscale or RGB inputs

3. **Computational Constraints**: 
   - Implementing deep learning from scratch in NumPy is computationally intensive
   - Training may take significant time depending on your hardware
   - Consider using smaller batch sizes or fewer iterations if needed

4. **Dataset**: 
   - Training dataset paths are hardcoded in `preprocess_image.py`
   - Update the paths according to your local setup:
     ```python
     preprocess_folder_real = "path/to/Jay/images/"
     preprocess_folder_fake = "path/to/look-alike/images/"
     ```

5. **Test Images**: 
   - The `photo/` directory should contain test images
   - Images should be in PNG or JPG format
   - Images will be automatically resized to 180×180 if needed

## 🤝 Contributing

This is an educational project. Contributions, suggestions, and improvements are welcome! Please feel free to:
- Report bugs or issues
- Suggest enhancements
- Submit pull requests
- Share your results and findings

## 📝 License

This project is for educational purposes. Please ensure you have proper permissions for any images used.

## 📚 References

- Deep Learning fundamentals (convolution, backpropagation, gradient descent)
- NumPy documentation for array operations
- Image processing techniques

## 👥 Authors

- Project contributors

---

**Note**: This project is designed for educational purposes to demonstrate understanding of deep learning fundamentals through complete from-scratch implementation. Performance metrics are modest compared to state-of-the-art systems but demonstrate clear learning and validate core concepts.
