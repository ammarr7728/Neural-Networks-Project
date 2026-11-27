# Model Design Plan

## 1. Project Overview

The goal of this project is to develop neural network models that classify chest X-ray images into two categories:

- NORMAL (0)
- PNEUMONIA (1)

The project focuses on designing, training, and comparing different models to identify the most suitable architecture for this binary image classification task.

## 2. Input and Output Specifications

### Input
- Image size: 224 × 224 pixels.
- Color channels: 1 (Grayscale).
- Data type: float32.
- Pixel normalization: [0, 1].
- Input shape: (224, 224, 1).

### Output
The model produces a single probability using a sigmoid activation function.

- Probability below 0.5: NORMAL.
- Probability equal to or above 0.5: PNEUMONIA.

The classification threshold will be evaluated and may be adjusted using validation data.

## 3. Planned Models

The project will compare two baseline models and multiple neural network architectures.

### Model A: Traditional Machine Learning Baseline

A traditional machine learning classifier will be used as a baseline for comparison.

The planned approach is to:
1. Extract image features using a suitable feature extraction method.
2. Convert images into numerical feature vectors.
3. Train a traditional classifier.
4. Evaluate its performance using the validation data.

The final algorithm and feature extraction method will be documented after implementation.

### Model B: Simple CNN

A basic Convolutional Neural Network will be implemented to establish an initial deep learning baseline.

The planned architecture includes:

1. Input Layer: 224 × 224 × 1.
2. First Convolutional Layer: 32 filters, 3 × 3 kernel, ReLU activation.
3. Max Pooling Layer.
4. Second Convolutional Layer: 64 filters, 3 × 3 kernel, ReLU activation.
5. Max Pooling Layer.
6. Third Convolutional Layer: 128 filters, 3 × 3 kernel, ReLU activation.
7. Max Pooling Layer.
8. Global Average Pooling.
9. Dense Layer: 64 units with ReLU activation.
10. Dropout Layer.
11. Output Layer: 1 unit with sigmoid activation.

### Model C: Improved CNN Architecture

An improved neural network architecture will be designed and evaluated against the simple CNN.

Possible improvements include:

- Adjusting the number of convolutional filters.
- Experimenting with different dropout rates.
- Applying regularization techniques.
- Tuning learning rates and batch sizes.
- Testing alternative architectural configurations.

The final improved architecture will be selected based on experimental results rather than assumptions.

## 4. Regularization Techniques

The following techniques will be considered to reduce overfitting:

### Dropout
Randomly disables a proportion of neurons during training to improve generalization.

### Data Augmentation
Training images will undergo:
- Random Rotation.
- Random Translation.

Augmentation will only be applied to training data.

### Early Stopping
Training will stop when the monitored validation metric stops improving for a specified number of epochs.

## 5. Training Configuration

The initial training configuration is:

| Parameter | Initial Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 30 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Output Activation | Sigmoid |
| Main Activation | ReLU |
| Dropout | 0.5 |
| Early Stopping Patience | 5 |

These values represent the initial configuration and may be modified during experimentation.

## 6. Class Imbalance Handling

The dataset contains more PNEUMONIA images than NORMAL images.

To reduce the effect of class imbalance, class weights will be calculated from the training set and used during model training.

The positive class is defined as:

**PNEUMONIA = 1**

## 7. Evaluation Strategy

The models will be evaluated using:

- Accuracy.
- Precision.
- Recall.
- F1-Score.
- ROC-AUC.
- Confusion Matrix.

### Primary Metric: Pneumonia Recall

Recall is selected as the primary metric because identifying pneumonia cases is particularly important.

The team will also monitor ROC-AUC and other metrics to ensure that a high recall value is not achieved simply by predicting every image as PNEUMONIA.

Model selection will consider the overall validation performance, not recall alone.

The test set will only be used for final evaluation after model selection.

## 8. Experimentation and Reproducibility

To ensure reproducibility, the project will use the following random seeds:

- 42
- 123
- 2026

Experiments will record:

- Model architecture.
- Hyperparameter configuration.
- Random seed.
- Training and validation metrics.
- Training duration.
- Best model checkpoint.

## 9. Model Selection Strategy

The final model will be selected based on:

1. Validation Pneumonia Recall.
2. ROC-AUC and F1-Score.
3. Confusion matrix analysis.
4. Generalization performance.
5. Consistency across different random seeds.

A model that predicts every image as PNEUMONIA will not be considered a successful model, even if it achieves 100% recall.

## 10. Expected Outcome

The expected outcome is a reliable CNN-based model that can distinguish between NORMAL and PNEUMONIA chest X-ray images.

The project will compare baseline models and neural network architectures, analyze their performance, and select the best-performing model based on validation results.

The system is intended for academic research and is not a replacement for professional medical diagnosis.

## 11. Implementation Status

The project currently includes an initial custom CNN architecture and a configurable data preprocessing pipeline.

The final Model B and Model C architectures, baseline comparisons, and experimental results will be documented after implementation and evaluation.
