# Dataset Research

## 1. Research Objective

The purpose of this research is to identify and evaluate a suitable dataset for the Chest X-Ray Pneumonia Classification project.

The selected dataset must contain labeled chest X-ray images that can be used to train and evaluate neural network models for binary image classification.

## 2. Selected Dataset

**Dataset Name:** Chest X-Ray Pneumonia Dataset

**Source:** Hugging Face

**Dataset Repository:** https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia

**Task:** Binary Image Classification

**Classes:**
- NORMAL
- PNEUMONIA

## 3. Why Was This Dataset Selected?

The dataset was selected for the following reasons:

1. It is directly related to the project's objective.
2. It contains real chest X-ray images.
3. Each image has a corresponding class label.
4. It provides two clearly defined categories: NORMAL and PNEUMONIA.
5. It is suitable for supervised learning and neural network experiments.
6. It is accessible through the Hugging Face Datasets library.
7. Its image format makes it suitable for CNN-based image classification.

## 4. Dataset Features

The dataset contains two main features:

| Feature | Description |
|---|---|
| image | Chest X-ray image |
| label | Class label associated with the image |

The label feature uses the following mapping:

| Label ID | Class |
|---:|---|
| 0 | NORMAL |
| 1 | PNEUMONIA |

## 5. Original Dataset Structure

The dataset is provided with three splits:

- Train
- Validation
- Test

The original dataset contains:

| Split | Number of Images |
|---|---:|
| Train | 5,216 |
| Validation | 16 |
| Test | 624 |
| **Total** | **5,856** |

The original validation split is very small, so the project applies a new training/validation split to obtain a more suitable validation set while keeping the original test split separate.

## 6. Initial Class Distribution

The original dataset has the following class distribution:

### Training Split

| Class | Number of Images |
|---|---:|
| NORMAL | 1,341 |
| PNEUMONIA | 3,875 |
| **Total** | **5,216** |

### Validation Split

| Class | Number of Images |
|---|---:|
| NORMAL | 8 |
| PNEUMONIA | 8 |
| **Total** | **16** |

### Test Split

| Class | Number of Images |
|---|---:|
| NORMAL | 234 |
| PNEUMONIA | 390 |
| **Total** | **624** |

## 7. Dataset Challenges

During dataset investigation, several challenges were identified:

### 7.1 Class Imbalance
The number of PNEUMONIA images is higher than the number of NORMAL images.

This may cause a model to favor the majority class.

**Proposed solution:** Calculate class weights from the training data and monitor multiple evaluation metrics.

### 7.2 Small Validation Set
The original validation split contains only 16 images, which is insufficient for reliable model selection.

**Proposed solution:** Create a new validation split from the available training data while preserving the original test set.

### 7.3 Duplicate Images
Duplicate images may exist across different dataset splits.

**Proposed solution:** Use SHA-256 image hashes to identify duplicates and remove duplicate copies according to the project's split-priority policy.

### 7.4 Corrupted Images
Some image files may be unreadable or invalid.

**Proposed solution:** Check image readability during data cleaning and exclude corrupted images.

### 7.5 Data Leakage
Images from the same patient appearing in both training and validation sets may lead to overly optimistic evaluation results.

**Proposed solution:** Extract patient identifiers from filenames and use patient-based grouping when creating the training and validation splits. Check image hash overlap as well.

## 8. Suitability for Neural Networks

The dataset is suitable for CNN-based classification because:

- Images contain visual patterns that can help distinguish the two classes.
- Images can be resized to a consistent input shape.
- Grayscale images reduce input complexity.
- Labels are available for supervised training.
- The dataset supports experiments with different CNN architectures and hyperparameters.

## 9. Research Decision

After reviewing the dataset's content, labels, structure, and limitations, the team selected the Chest X-Ray Pneumonia Dataset from Hugging Face.

The dataset was considered appropriate for the project's binary image classification task.

The team will use a cleaning and preprocessing pipeline to improve data quality and reduce the risk of data leakage before training the models.

## 10. References

1. Hugging Face Dataset Repository: https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia
2. TensorFlow Documentation: https://www.tensorflow.org/
3. Hugging Face Datasets Documentation: https://huggingface.co/docs/datasets/

---

**Note:** The original dataset statistics in this document describe the dataset before cleaning. The final cleaned dataset statistics are documented separately in `dataset_info.md`.
