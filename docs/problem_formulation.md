# Problem Formulation

## 1. Project Title
**Chest X-Ray Pneumonia Classification Using Neural Networks**

## 2. Problem Statement
Pneumonia is a lung infection that can cause serious health complications if it is not detected and treated early.

Chest X-ray images are commonly used by medical professionals to help identify pneumonia. However, manually examining a large number of X-ray images can be time-consuming.

The goal of this project is to develop a deep learning model that analyzes chest X-ray images and classifies them into two categories:

- **NORMAL:** The X-ray image shows no signs of pneumonia.
- **PNEUMONIA:** The X-ray image indicates the presence of pneumonia.

The project aims to investigate how neural networks can support the classification process by providing predictions from chest X-ray images.

## 3. Project Objectives
The main objectives are:

1. Collect and prepare a chest X-ray image dataset.
2. Explore and understand the dataset through Exploratory Data Analysis (EDA).
3. Clean the dataset and handle duplicate or corrupted images.
4. Create training, validation, and testing datasets while avoiding data leakage.
5. Build and evaluate two baseline models.
6. Design and train neural network architectures for image classification.
7. Apply data augmentation and regularization techniques to improve model performance.
8. Experiment with different hyperparameters and compare the results.
9. Evaluate the final model using appropriate classification metrics.

## 4. Dataset Description
The project uses the **Chest X-Ray Pneumonia Dataset** from Hugging Face.

**Dataset source:** https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia

The dataset contains grayscale chest X-ray images belonging to two classes:

| Class | Label |
|---|---:|
| NORMAL | 0 |
| PNEUMONIA | 1 |

After data cleaning and splitting, the dataset contains:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,078 | 3,086 | 4,164 |
| Validation | 270 | 772 | 1,042 |
| Testing | 231 | 387 | 618 |
| **Total** | **1,579** | **4,245** | **5,824** |

All images are resized to **224 × 224** pixels and converted to grayscale with normalized pixel values.

## 5. Machine Learning Task
This project is a **Supervised Binary Image Classification** problem.

The input is a chest X-ray image, and the output is a binary prediction:

- 0 → NORMAL
- 1 → PNEUMONIA

The neural network learns patterns and features from labeled training images to predict the class of previously unseen images.

## 6. Primary Evaluation Metric
The primary evaluation metric is **Pneumonia Recall**.

Recall measures the proportion of actual pneumonia cases correctly identified by the model.

\[
Recall = \frac{TP}{TP + FN}
\]

Where:
- TP: True Positives.
- FN: False Negatives.

Pneumonia Recall is selected as the primary metric because failing to identify a pneumonia case (False Negative) is particularly important in this application.

Other evaluation metrics include:

- Accuracy
- Precision
- F1-Score
- ROC-AUC
- Confusion Matrix

## 7. Data Preparation and Leakage Prevention
The preprocessing pipeline includes:

- Checking image readability and removing corrupted images.
- Detecting duplicate images using image hashes.
- Checking for conflicting labels among duplicates.
- Splitting the data into training, validation, and testing sets.
- Using patient-based grouping where possible to reduce the risk of patient leakage.
- Resizing and normalizing images.
- Applying data augmentation to training images only.

The test set is kept separate from model training and hyperparameter tuning to provide an unbiased final evaluation.

## 8. Project Scope
The project focuses on:

- Binary classification of chest X-ray images.
- Comparing baseline approaches with neural network models.
- Improving classification performance through experiments.
- Evaluating models using appropriate metrics.
- Selecting a final model based on experimental results.

The project is an academic research and development task. It is **not intended to replace professional medical diagnosis** or provide independent clinical decisions.

## 9. Expected Outcome
The expected outcome is a trained neural network model capable of distinguishing between NORMAL and PNEUMONIA chest X-ray images.

The project will compare multiple models and experiments, analyze their performance, and select the most suitable model with particular attention to Pneumonia Recall.
